# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Linux-socket IMX219 player with SCRFD face detection (HSB 3.0 / HSDK5 Graph API).
# Refer to fpga/efinix/Ti375N1156/README.md for the demo setup.
#
# Captures IMX219 1920x1080 RAW10 over UDP via LinuxReceiverOp, runs SCRFD face
# detection, and overlays green rectangles on the live Holoviz video. The boxes layer is an
# RGBA image overlay (HSDK5 GeometryLayer has no per-layer color). With --servo, the
# largest-face center also drives pan/tilt PWM through the Efinix apb_servo_slave APB
# registers. Default model is examples/500m.onnx. Exposure / letterbox / normalize /
# inference / postprocessor / servo / visualizer settings come from
# examples/linux_imx219_face_detection.yaml.
#
#   LinuxReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> ExposureOp
#     -> (a) HolovizOp "video" image layer
#     -> (b) FpsOp
#     -> (c) LetterboxOp (640x640) -> NormalizeOp -> InferenceOp (SCRFD) -> ScrfdPostprocessOp
#          -> (c1) HolovizOp "boxes" RGBA image overlay
#          -> (c2) ServoControlOp  (optional, --servo)
#
# Usage:
#   linux_imx219_face_detection.py [OPTIONS]
#
# Options:
#   --hololink IP         IP address of Hololink board (default: 192.168.0.2)
#   --camera-mode N       IMX219 mode (default: 0 = 1920x1080 RAW10)
#   --i2c-bus N           I2C bus override (default: sensor driver default)
#   --configuration PATH  YAML config (default: examples/linux_imx219_face_detection.yaml)
#   --model PATH          SCRFD ONNX/engine path (default: examples/500m.onnx)
#   --frame-limit N       Number of frames to capture (default: unlimited)
#   --headless            Run without opening a window (still renders, presents nowhere)
#   --fullscreen          Run in fullscreen mode
#   --servo               Enable Hololink APB servo writes (off by default)
#   --log-level N         Python logging level (default: 20 = INFO)

import argparse
import logging
import os
import signal
import sys
import time
from datetime import timedelta

import cupy as cp
import hololink.sensors.imx219
import holoscan as hs
import holoscan_vision
import holoscan_visualization as hz  # noqa: F401 -- registers "holoscan.visualization.HolovizOp"
import numpy as np
import yaml
from cuda.bindings import driver as cuda_driver

import hololink as hl

SCRFD_MODEL_NAME = "500m.onnx"

_SCRFD_OUTPUT_PORTS = (
    "score_8",
    "score_16",
    "score_32",
    "bbox_8",
    "bbox_16",
    "bbox_32",
)

# Bounded-window pairing for InferenceOp's six sibling SCRFD output ports, which it always
# emits together from the same compute() activation.
_INFERENCE_OUTPUT_PAIRING_WINDOW = timedelta(microseconds=500)


def scrfd_model_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), SCRFD_MODEL_NAME))


def _cuda_check(result):
    err = result[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        _, description = cuda_driver.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error: {description}")
    if len(result) == 2:
        return result[1]
    return result[1:]


def import_tensor(tensor, stream):
    # Borrow a read-only GPU view; callers finish reads before compute returns.
    return cp.from_dlpack(
        tensor.__dlpack__(stream=stream.ptr, max_version=(1, 0), copy=False)
    )


def box_overlay_style(viz_cfg):
    # Read box overlay color/line_width from YAML (RGBA 0..1 -> 0..255).
    color = list(viz_cfg.get("box_color", [0.0, 1.0, 0.0, 1.0]))
    line_width = int(viz_cfg.get("box_line_width", 3))
    while len(color) < 4:
        color.append(1.0)
    rgba = tuple(max(0, min(255, int(round(channel * 255.0)))) for channel in color[:4])
    return rgba, max(1, line_width)


def emit_host_tensor(port, values, options):
    values = np.ascontiguousarray(values, dtype=np.float32).ravel()
    port.emit(hs.Tensor.as_tensor(values), options=options)


def import_host_tensor(tensor):
    return np.from_dlpack(tensor)


# C1156HSB / Ti375N1156 apb_servo_slave on Hololink external APB slot 1.
SERVO_APB_BASE = 0x20000000
SERVO_REG_CTRL = SERVO_APB_BASE + 0x00
SERVO_REG_PAN = SERVO_APB_BASE + 0x04
SERVO_REG_TILT = SERVO_APB_BASE + 0x08
SERVO_REG_PERIOD = SERVO_APB_BASE + 0x0C
SERVO_CTRL_ENABLE = 0x1
SERVO_CTRL_PAN_EN = 0x2
SERVO_CTRL_TILT_EN = 0x4


def wait_for_completion(session):
    # Wait for the session to finish and report whether any runtime errors occurred.
    try:
        session.wait()
    except KeyboardInterrupt:
        session.request_stop()
        session.wait()
    return any(
        diag.severity == hs.diagnostics.DiagnosticSeverity.ERROR
        for diag in session.diagnostics
    )


def _load_configuration(config_path):
    if not config_path or not os.path.isfile(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class ExposureOp(hs.Operator):
    # Convert demosaiced uint16 RGB to uint8 and apply exposure/white balance.

    input = hs.Input("input", hs.schema.Image)
    output = hs.Output("output", hs.schema.Image)

    exposure = hs.Param(1.0)
    red_gain = hs.Param(1.05)
    green_gain = hs.Param(0.95)
    blue_gain = hs.Param(1.05)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logged = False
        self._gains = None

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)
        spec.output(self.output).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=3,
            ),
            hs.tensor.byte_span(1920 * 1080 * 3),
            field="data",
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        with context.stream():
            self._compute(context, cp.cuda.get_current_stream())

    def _compute(self, context, stream):
        sample = self.input.receive()
        image_msg = sample.data
        image = import_tensor(image_msg.data, stream)

        if image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(f"Expected HWC RGB/RGBA, got shape {image.shape}")

        if self._gains is None:
            self._gains = cp.asarray(
                [self.red_gain, self.green_gain, self.blue_gain], dtype=cp.float32
            )

        rgb = image[:, :, :3]
        if rgb.dtype == cp.uint16:
            # ImageProcessorOp stores RAW10-derived data in the upper bits.
            rgb = (rgb >> 8).astype(cp.float32)
        else:
            rgb = rgb.astype(cp.float32)

        scaled = cp.clip(rgb * self.exposure * self._gains, 0, 255).astype(cp.uint8)

        if not self._logged:
            self._logged = True
            print(
                "ExposureOp first frame:",
                f"input_shape={image.shape}",
                f"input_dtype={image.dtype}",
                f"output_shape={scaled.shape}",
                f"min={int(scaled.min())}",
                f"max={int(scaled.max())}",
            )

        descriptor = hs.schema.Image(
            height=image_msg.height,
            width=image_msg.width,
            encoding=hs.schema.image.ImageEncoding.RGB8,
        )
        request = hs.tensor.LoanRequest(
            shape=scaled.shape, dtype=hs.tensor.DLDataType(1, 8, 1)
        )
        provenance = hs.sensor_io.ProvenanceAccumulator()
        provenance.observe(sample.metadata)
        options = provenance.emit_options(hs.sensor_io.CaptureTimePolicy.EARLIEST)
        with self.output.allocate_tensors(
            descriptor, {"data": request}, options=options
        ) as loan:
            tensor = loan.tensors["data"]
            memory = cp.cuda.UnownedMemory(tensor.data, scaled.nbytes, tensor)
            destination = cp.ndarray(
                scaled.shape,
                dtype=cp.uint8,
                memptr=cp.cuda.MemoryPointer(memory, 0),
            )
            cp.copyto(destination, scaled)


class FpsOp(hs.Operator):
    # Report end-to-end display FPS every two seconds.

    input = hs.Input("input", hs.schema.Image)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._count = 0
        self._start = None

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()
        now = time.perf_counter()
        if self._start is None:
            self._start = now
        self._count += 1
        elapsed = now - self._start
        if elapsed >= 2.0:
            print(f"end-to-end FPS: {self._count / elapsed:.1f}")
            self._count = 0
            self._start = now


class LetterboxOp(hs.Operator):
    # Scale 1920x1080 RGB to 640x360 and place it in a 640x640 canvas.

    input = hs.Input("input", hs.schema.Image)
    output = hs.Output("output", hs.schema.Image)

    width = hs.Param(640)
    height = hs.Param(640)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logged = False

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)
        spec.output(self.output).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=3,
            ),
            hs.tensor.byte_span(self.width * self.height * 3),
            field="data",
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        with context.stream():
            self._compute(context, cp.cuda.get_current_stream())

    def _compute(self, context, stream):
        sample = self.input.receive()
        rgb = import_tensor(sample.data.data, stream)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"Expected HWC RGB, got shape {rgb.shape}")

        src_height, src_width = rgb.shape[:2]
        image_ratio = src_height / src_width
        model_ratio = self.height / self.width
        if image_ratio > model_ratio:
            new_height = self.height
            new_width = int(new_height / image_ratio)
        else:
            new_width = self.width
            new_height = int(new_width * image_ratio)

        # The bring-up path is 1920x1080 -> 640x360. Average each 3x3 source block
        # on the GPU, avoiding an OpenCV host round-trip.
        if src_width == new_width * 3 and src_height == new_height * 3:
            resized = (
                rgb.reshape(new_height, 3, new_width, 3, 3)
                .mean(axis=(1, 3))
                .astype(cp.uint8)
            )
        else:
            x_index = cp.minimum(
                cp.arange(new_width) * src_width // new_width, src_width - 1
            )
            y_index = cp.minimum(
                cp.arange(new_height) * src_height // new_height, src_height - 1
            )
            resized = rgb[y_index[:, None], x_index[None, :]]

        scale = new_height / src_height
        if not self._logged:
            self._logged = True
            print(
                "LetterboxOp first frame:",
                f"src={src_width}x{src_height}",
                f"resized={new_width}x{new_height}",
                f"canvas={self.width}x{self.height}",
                f"scale={scale:.6f}",
            )

        canvas_shape = (self.height, self.width, 3)
        descriptor = hs.schema.Image(
            height=self.height,
            width=self.width,
            encoding=hs.schema.image.ImageEncoding.RGB8,
        )
        request = hs.tensor.LoanRequest(
            shape=canvas_shape, dtype=hs.tensor.DLDataType(1, 8, 1)
        )
        provenance = hs.sensor_io.ProvenanceAccumulator()
        provenance.observe(sample.metadata)
        options = provenance.emit_options(hs.sensor_io.CaptureTimePolicy.EARLIEST)
        with self.output.allocate_tensors(
            descriptor, {"data": request}, options=options
        ) as loan:
            tensor = loan.tensors["data"]
            memory = cp.cuda.UnownedMemory(
                tensor.data, self.height * self.width * 3, tensor
            )
            destination = cp.ndarray(
                canvas_shape,
                dtype=cp.uint8,
                memptr=cp.cuda.MemoryPointer(memory, 0),
            )
            destination.fill(0)
            destination[:new_height, :new_width] = resized


class NormalizeOp(hs.Operator):
    # Convert uint8 HWC RGB to contiguous float32 NCHW tensor to meet the input requirements of the SCRFD model.

    input = hs.Input("input", hs.schema.Image)
    output = hs.Output("output", hs.payload.Tensor)

    width = hs.Param(640)
    height = hs.Param(640)
    mean = hs.Param(127.5)
    std = hs.Param(128.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logged = False

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)
        spec.output(self.output).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(2, 32, 1),
                rank=4,
            ),
            hs.tensor.byte_span(1 * 3 * self.height * self.width * 4),
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        with context.stream():
            self._compute(context, cp.cuda.get_current_stream())

    def _compute(self, context, stream):
        sample = self.input.receive()
        image = import_tensor(sample.data.data, stream).astype(cp.float32, copy=False)
        normalized = (image - self.mean) / self.std
        model_input = cp.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])

        if not self._logged:
            self._logged = True
            print(
                "NormalizeOp first frame:",
                f"shape={model_input.shape}",
                f"dtype={model_input.dtype}",
                f"min={float(model_input.min()):.3f}",
                f"max={float(model_input.max()):.3f}",
            )

        provenance = hs.sensor_io.ProvenanceAccumulator()
        provenance.observe(sample.metadata)
        options = provenance.emit_options(hs.sensor_io.CaptureTimePolicy.EARLIEST)
        with self.output.allocate_tensor(
            shape=model_input.shape, dtype="float32", options=options
        ) as loan:
            memory = cp.cuda.UnownedMemory(
                loan.tensor.data, model_input.nbytes, loan.tensor
            )
            destination = cp.ndarray(
                model_input.shape,
                dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(memory, 0),
            )
            cp.copyto(destination, model_input)


class ScrfdPostprocessOp(hs.Operator):
    # Decode six-output SCRFD tensors, run NMS, and emit a green box overlay.

    in_score_8 = hs.Input("in_score_8", hs.payload.Tensor)
    in_score_16 = hs.Input("in_score_16", hs.payload.Tensor)
    in_score_32 = hs.Input("in_score_32", hs.payload.Tensor)
    in_bbox_8 = hs.Input("in_bbox_8", hs.payload.Tensor)
    in_bbox_16 = hs.Input("in_bbox_16", hs.payload.Tensor)
    in_bbox_32 = hs.Input("in_bbox_32", hs.payload.Tensor)
    boxes = hs.Output("boxes", hs.schema.Image)
    face_center = hs.Output("face_center", hs.payload.Tensor)

    model_width = hs.Param(640)
    model_height = hs.Param(640)
    image_width = hs.Param(1920)
    image_height = hs.Param(1080)
    letterbox_scale = hs.Param(1.0 / 3.0)
    score_threshold = hs.Param(0.5)
    nms_threshold = hs.Param(0.4)
    max_detections = hs.Param(100)
    line_width = hs.Param(3)
    box_r = hs.Param(0)
    box_g = hs.Param(255)
    box_b = hs.Param(0)
    box_a = hs.Param(255)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._anchor_centers = {}
        self._logged = False
        self._logged_detection = False
        self._logged_center = False
        self._box_rgba = None

    def setup(self, spec):
        for port in (
            self.in_score_8,
            self.in_score_16,
            self.in_score_32,
            self.in_bbox_8,
            self.in_bbox_16,
            self.in_bbox_32,
        ):
            spec.input(port, queue_depth=4)
        spec.output(self.face_center)
        spec.output(self.boxes).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=3,
            ),
            hs.tensor.byte_span(self.image_width * self.image_height * 4),
            field="data",
        )

    def contract(self):
        ports = [
            self.in_score_8,
            self.in_score_16,
            self.in_score_32,
            self.in_bbox_8,
            self.in_bbox_16,
            self.in_bbox_32,
        ]
        contract = hs.Contract().trigger(
            hs.OnAligned(
                ports=ports,
                window=_INFERENCE_OUTPUT_PAIRING_WINDOW,
            )
        )
        for port in ports:
            contract = contract.input(port, hs.Exact())
        return contract

    def _centers(self, stride):
        if stride not in self._anchor_centers:
            height = self.model_height // stride
            width = self.model_width // stride
            centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
                np.float32
            )
            centers = (centers * stride).reshape(-1, 2)
            centers = np.stack([centers] * 2, axis=1).reshape(-1, 2)
            self._anchor_centers[stride] = centers
        return self._anchor_centers[stride]

    def _nms(self, detections):
        x1, y1, x2, y2, scores = detections.T
        areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0 and len(keep) < self.max_detections:
            index = order[0]
            keep.append(index)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[index], x1[rest])
            yy1 = np.maximum(y1[index], y1[rest])
            xx2 = np.minimum(x2[index], x2[rest])
            yy2 = np.minimum(y2[index], y2[rest])
            width = np.maximum(0.0, xx2 - xx1 + 1)
            height = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width * height
            union = areas[index] + areas[rest] - intersection
            order = rest[(intersection / np.maximum(union, 1e-6)) <= self.nms_threshold]

        return detections[keep]

    def compute(self, context):
        with context.stream():
            self._compute(context, cp.cuda.get_current_stream())

    def _compute(self, context, stream):
        samples = {
            "score_8": self.in_score_8.receive(),
            "score_16": self.in_score_16.receive(),
            "score_32": self.in_score_32.receive(),
            "bbox_8": self.in_bbox_8.receive(),
            "bbox_16": self.in_bbox_16.receive(),
            "bbox_32": self.in_bbox_32.receive(),
        }
        provenance = hs.sensor_io.ProvenanceAccumulator()
        for sample in samples.values():
            provenance.observe(sample.metadata)
        options = provenance.emit_options(hs.sensor_io.CaptureTimePolicy.EARLIEST)

        candidates = []
        for stride in (8, 16, 32):
            scores = import_tensor(samples[f"score_{stride}"].data, stream)[0, :, 0]
            distances = (
                import_tensor(samples[f"bbox_{stride}"].data, stream)[0] * stride
            )
            positive = cp.where(scores >= self.score_threshold)[0]
            if positive.size == 0:
                continue

            positive_cpu = cp.asnumpy(positive)
            centers = self._centers(stride)[positive_cpu]
            scores_cpu = cp.asnumpy(scores[positive])
            distances_cpu = cp.asnumpy(distances[positive])
            boxes = np.stack(
                (
                    centers[:, 0] - distances_cpu[:, 0],
                    centers[:, 1] - distances_cpu[:, 1],
                    centers[:, 0] + distances_cpu[:, 2],
                    centers[:, 1] + distances_cpu[:, 3],
                ),
                axis=1,
            )
            candidates.append(np.column_stack((boxes, scores_cpu)))

        if candidates:
            detections = np.concatenate(candidates).astype(np.float32, copy=False)
            detections = detections[detections[:, 4].argsort()[::-1]]
            detections = self._nms(detections)

            model_detections = detections.copy()
            detections[:, :4] /= self.letterbox_scale
            detections[:, [0, 2]] = np.clip(detections[:, [0, 2]], 0, self.image_width)
            detections[:, [1, 3]] = np.clip(detections[:, [1, 3]], 0, self.image_height)
            valid = (detections[:, 2] > detections[:, 0]) & (
                detections[:, 3] > detections[:, 1]
            )
            model_detections = model_detections[valid]
            detections = detections[valid]

            if detections.size and not self._logged_detection:
                self._logged_detection = True
                print(
                    "SCRFD first detection:",
                    f"score={detections[0, 4]:.3f}",
                    f"model_box={model_detections[0, :4].round(1).tolist()}",
                    f"image_box={detections[0, :4].round(1).tolist()}",
                )
        else:
            detections = np.empty((0, 5), dtype=np.float32)

        if not self._logged:
            self._logged = True
            print(
                "ScrfdPostprocessOp first frame:",
                f"detections={detections.shape[0]}",
                f"score_threshold={self.score_threshold}",
                f"nms_threshold={self.nms_threshold}",
            )

        # Largest-box center in normalized image coords for ServoControlOp.
        face_center = np.empty((0,), dtype=np.float32)
        if detections.size:
            areas = (detections[:, 2] - detections[:, 0]) * (
                detections[:, 3] - detections[:, 1]
            )
            best_idx = int(np.argmax(areas))
            x1, y1, x2, y2, score = detections[best_idx]
            cx = (x1 + x2) * 0.5 / self.image_width
            cy = (y1 + y2) * 0.5 / self.image_height
            face_center = np.asarray([cx, cy], dtype=np.float32)
            if not self._logged_center:
                self._logged_center = True
                print(
                    "Scrfd face center:",
                    f"score={score:.3f}",
                    f"area_px={(x2 - x1) * (y2 - y1):.1f}",
                    f"center_px=({(x1 + x2) * 0.5:.1f}, {(y1 + y2) * 0.5:.1f})",
                    f"center_norm=({cx:.4f}, {cy:.4f})",
                )
        emit_host_tensor(self.face_center, face_center, options)

        if self._box_rgba is None:
            self._box_rgba = cp.asarray(
                [self.box_r, self.box_g, self.box_b, self.box_a], dtype=cp.uint8
            )

        overlay_shape = (self.image_height, self.image_width, 4)
        descriptor = hs.schema.Image(
            height=self.image_height,
            width=self.image_width,
            encoding=hs.schema.image.ImageEncoding.RGBA8,
        )
        request = hs.tensor.LoanRequest(
            shape=overlay_shape, dtype=hs.tensor.DLDataType(1, 8, 1)
        )
        with self.boxes.allocate_tensors(
            descriptor, {"data": request}, options=options
        ) as loan:
            tensor = loan.tensors["data"]
            memory = cp.cuda.UnownedMemory(
                tensor.data, self.image_height * self.image_width * 4, tensor
            )
            overlay = cp.ndarray(
                overlay_shape,
                dtype=cp.uint8,
                memptr=cp.cuda.MemoryPointer(memory, 0),
            )
            overlay.fill(0)
            thickness = max(1, int(self.line_width))
            for detection in detections:
                x0 = max(0, min(self.image_width - 1, int(detection[0])))
                y0 = max(0, min(self.image_height - 1, int(detection[1])))
                x1 = max(0, min(self.image_width, int(detection[2])))
                y1 = max(0, min(self.image_height, int(detection[3])))
                if x1 <= x0 or y1 <= y0:
                    continue
                top = min(y0 + thickness, y1)
                bottom = max(y1 - thickness, y0)
                left = min(x0 + thickness, x1)
                right = max(x1 - thickness, x0)
                overlay[y0:top, x0:x1] = self._box_rgba
                overlay[bottom:y1, x0:x1] = self._box_rgba
                overlay[y0:y1, x0:left] = self._box_rgba
                overlay[y0:y1, right:x1] = self._box_rgba


class ServoControlOp(hs.Operator):
    # Use largest-face center error to update pan/tilt PWM setpoints.
    #
    # face_center is normalized Holoviz/image coords in [0, 1]. Image center is (0.5, 0.5).
    # Positive error means the face is right/down of center; invert_* flips the command.

    input = hs.Input("input", hs.payload.Tensor)

    def __init__(
        self,
        *args,
        hololink=None,
        enabled=True,
        enable_pan=True,
        enable_tilt=False,
        center_pulse=150_000,
        tilt_center_pulse=60_000,
        min_pulse=50_000,
        max_pulse=250_000,
        tilt_min_pulse=50_000,
        tilt_max_pulse=150_000,
        period_cycles=2_000_000,
        control_hz=15.0,
        velocity_filter_alpha=0.8,
        command_amplifier=0.9,
        max_error_amplifier=3.5,
        pan_step_gain=5_000.0,
        pan_velocity_gain=2_000.0,
        pan_min_step=500,
        pan_max_step=8_000,
        tilt_step_gain=5_000.0,
        tilt_velocity_gain=2_000.0,
        tilt_min_step=500,
        tilt_max_step=8_000,
        deadband=0.05,
        invert_pan=True,
        invert_tilt=True,
        write_period=True,
        **kwargs,
    ):
        self._hololink = hololink
        self._enabled = enabled
        self._enable_pan = bool(enable_pan)
        self._enable_tilt = bool(enable_tilt)
        self._center_pulse = int(center_pulse)
        self._tilt_center_pulse = int(tilt_center_pulse)
        self._min_pulse = int(min_pulse)
        self._max_pulse = int(max_pulse)
        self._tilt_min_pulse = int(tilt_min_pulse)
        self._tilt_max_pulse = int(tilt_max_pulse)
        self._period_cycles = int(period_cycles)
        self._control_interval_s = 1.0 / float(control_hz)
        self._velocity_filter_alpha = float(velocity_filter_alpha)
        self._command_amplifier = float(command_amplifier)
        self._max_error_amplifier = float(max_error_amplifier)
        self._pan_step_gain = float(pan_step_gain)
        self._pan_velocity_gain = float(pan_velocity_gain)
        self._pan_min_step = int(pan_min_step)
        self._pan_max_step = int(pan_max_step)
        self._tilt_step_gain = float(tilt_step_gain)
        self._tilt_velocity_gain = float(tilt_velocity_gain)
        self._tilt_min_step = int(tilt_min_step)
        self._tilt_max_step = int(tilt_max_step)
        self._deadband = float(deadband)
        self._invert_pan = bool(invert_pan)
        self._invert_tilt = bool(invert_tilt)
        self._write_period = bool(write_period)
        self._period_written = False
        self._last_pan = None
        self._last_tilt = None
        self._logged = False
        self._pan_setpoint = self._center_pulse
        self._tilt_setpoint = self._tilt_center_pulse
        self._last_measurement_time = None
        self._last_control_time = None
        self._last_error_x = 0.0
        self._last_error_y = 0.0
        self._filtered_velocity_x = 0.0
        self._filtered_velocity_y = 0.0
        # After the one-shot home in start(), first face / reacquire must continue from the
        # live pose — never snap back to center.
        self._awaiting_reacquire = True
        self._skip_next_step = False
        self._homed_at_start = False
        super().__init__(*args, **kwargs)

    def _ctrl_bits(self):
        ctrl = SERVO_CTRL_ENABLE
        if self._enable_pan:
            ctrl |= SERVO_CTRL_PAN_EN
        if self._enable_tilt:
            ctrl |= SERVO_CTRL_TILT_EN
        return ctrl

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def start(self):
        # Move to center once at app start; not on first face detection.
        if not self._enabled or self._homed_at_start:
            return
        self._pan_setpoint = self._center_pulse
        self._tilt_setpoint = self._tilt_center_pulse
        self._write_regs(self._ctrl_bits(), self._pan_setpoint, self._tilt_setpoint)
        self._homed_at_start = True
        print(
            "ServoControlOp home at start:",
            f"pan={self._pan_setpoint}",
            f"tilt={self._tilt_setpoint}",
        )

    @staticmethod
    def _clip_pulse(value, min_pulse, max_pulse):
        return int(np.clip(value, min_pulse, max_pulse))

    def _step_setpoint(
        self,
        error,
        velocity,
        invert,
        step_gain,
        velocity_gain,
        min_step,
        max_step,
        setpoint,
        min_pulse,
        max_pulse,
    ):
        # Apply a bounded proportional + velocity step to a setpoint.
        if abs(error) < self._deadband:
            return setpoint

        signed = -error if invert else error
        signed_velocity = -velocity if invert else velocity
        # Increase the multiplier smoothly from the base value just outside the deadband to
        # max_error_amplifier at an image edge (|error|=0.5).
        normalized_error = np.clip(
            (abs(error) - self._deadband) / (0.5 - self._deadband), 0.0, 1.0
        )
        adaptive_amplifier = self._command_amplifier + (
            self._max_error_amplifier - self._command_amplifier
        ) * normalized_error
        raw_step = (signed * step_gain + signed_velocity * velocity_gain) * adaptive_amplifier
        step = int(round(raw_step))
        if abs(step) < min_step:
            direction = raw_step if raw_step != 0 else signed
            step = min_step if direction >= 0 else -min_step
        step = int(np.clip(step, -max_step, max_step))
        return self._clip_pulse(setpoint + step, min_pulse, max_pulse)

    def _step_pan(self, error, velocity):
        self._pan_setpoint = self._step_setpoint(
            error,
            velocity,
            self._invert_pan,
            self._pan_step_gain,
            self._pan_velocity_gain,
            self._pan_min_step,
            self._pan_max_step,
            self._pan_setpoint,
            self._min_pulse,
            self._max_pulse,
        )
        return self._pan_setpoint

    def _step_tilt(self, error, velocity):
        self._tilt_setpoint = self._step_setpoint(
            error,
            velocity,
            self._invert_tilt,
            self._tilt_step_gain,
            self._tilt_velocity_gain,
            self._tilt_min_step,
            self._tilt_max_step,
            self._tilt_setpoint,
            self._tilt_min_pulse,
            self._tilt_max_pulse,
        )
        return self._tilt_setpoint

    def _write_regs(self, ctrl, pan, tilt):
        if self._write_period and not self._period_written:
            self._hololink.write_uint32(SERVO_REG_PERIOD, self._period_cycles)
            self._period_written = True
        if pan != self._last_pan:
            self._hololink.write_uint32(SERVO_REG_PAN, int(pan))
            self._last_pan = pan
        if tilt != self._last_tilt:
            self._hololink.write_uint32(SERVO_REG_TILT, int(tilt))
            self._last_tilt = tilt
        self._hololink.write_uint32(SERVO_REG_CTRL, int(ctrl))

    def _sync_setpoints_from_hardware(self):
        # Seed software setpoints from the live PWM widths so first/reacquire tracking does
        # not snap pan/tilt home.
        try:
            pan = int(self._hololink.read_uint32(SERVO_REG_PAN))
            tilt = int(self._hololink.read_uint32(SERVO_REG_TILT))
            self._pan_setpoint = self._clip_pulse(pan, self._min_pulse, self._max_pulse)
            self._tilt_setpoint = self._clip_pulse(
                tilt, self._tilt_min_pulse, self._tilt_max_pulse
            )
            self._last_pan = self._pan_setpoint
            self._last_tilt = self._tilt_setpoint
        except Exception as exc:
            print(f"ServoControlOp: keep software setpoints ({exc})")

    def _begin_acquire(self, error_x, error_y):
        # Prepare tracking at the current pose; do not snap to center.
        self._sync_setpoints_from_hardware()
        self._last_measurement_time = None
        self._filtered_velocity_x = 0.0
        self._filtered_velocity_y = 0.0
        self._last_error_x = error_x
        self._last_error_y = error_y
        self._awaiting_reacquire = False
        self._skip_next_step = True

    def compute(self, context):
        sample = self.input.receive()
        if not self._enabled:
            return

        face_center = np.asarray(import_host_tensor(sample.data), dtype=np.float32).reshape(-1)
        if face_center.size < 2:
            # Hold the last APB pulse widths and leave PWM enabled. A missed face-detection
            # frame must not recenter or disable the turret.
            self._awaiting_reacquire = True
            return

        cx, cy = float(face_center[0]), float(face_center[1])
        error_x = cx - 0.5
        error_y = cy - 0.5
        now = time.monotonic()

        if self._awaiting_reacquire:
            self._begin_acquire(error_x, error_y)

        if self._last_measurement_time is not None:
            dt = max(now - self._last_measurement_time, 1e-3)
            raw_velocity_x = (error_x - self._last_error_x) / dt
            raw_velocity_y = (error_y - self._last_error_y) / dt
            alpha = self._velocity_filter_alpha
            self._filtered_velocity_x = (
                alpha * self._filtered_velocity_x + (1.0 - alpha) * raw_velocity_x
            )
            self._filtered_velocity_y = (
                alpha * self._filtered_velocity_y + (1.0 - alpha) * raw_velocity_y
            )

        self._last_measurement_time = now
        self._last_error_x = error_x
        self._last_error_y = error_y

        # Keep motion independent of inference FPS and avoid excess APB writes.
        if (
            self._last_control_time is not None
            and now - self._last_control_time < self._control_interval_s
        ):
            return
        self._last_control_time = now

        # First control tick after acquire: enable PWM at the current pose only.
        if self._skip_next_step:
            self._skip_next_step = False
            self._write_regs(self._ctrl_bits(), self._pan_setpoint, self._tilt_setpoint)
            return

        pan = (
            self._step_pan(error_x, self._filtered_velocity_x)
            if self._enable_pan
            else self._center_pulse
        )
        tilt = (
            self._step_tilt(error_y, self._filtered_velocity_y)
            if self._enable_tilt
            else self._tilt_center_pulse
        )

        self._write_regs(self._ctrl_bits(), pan, tilt)

        if not self._logged:
            self._logged = True
            print(
                "ServoControlOp first command:",
                f"center=({cx:.4f}, {cy:.4f})",
                f"error=({error_x:.4f}, {error_y:.4f})",
                f"pan={pan}",
                f"tilt={tilt}",
                f"pan_step_gain={self._pan_step_gain}",
                f"pan_velocity_gain={self._pan_velocity_gain}",
                f"pan_min_step={self._pan_min_step}",
                f"pan_max_step={self._pan_max_step}",
                f"tilt_step_gain={self._tilt_step_gain}",
                f"tilt_velocity_gain={self._tilt_velocity_gain}",
                f"tilt_min_step={self._tilt_min_step}",
                f"tilt_max_step={self._tilt_max_step}",
                f"enable_pan={self._enable_pan}",
                f"enable_tilt={self._enable_tilt}",
            )


class DiscardImageOp(hs.Operator):
    # With --headless, HolovizOp renders into a CUDA image on its output instead of a window;
    # This Op acts as the sink for the unused CUDA image.

    input = hs.Input("input", hs.schema.Image)

    def setup(self, spec):
        spec.input(self.input, queue_depth=1)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        self.input.receive_data()


def run(
    hololink_ip,
    camera_mode,
    i2c_bus,
    model_path,
    config_path,
    frame_limit,
    headless,
    fullscreen,
    servo_enabled,
):
    import holoscan_inference  # noqa: F401 -- deferred so --help works without TensorRT

    config = _load_configuration(config_path)
    exposure_cfg = config.get("exposure", {})
    letterbox_cfg = config.get("letterbox", {})
    normalize_cfg = config.get("normalize", {})
    inference_cfg = config.get("inference", {})
    postproc_cfg = config.get("postprocessor", {})
    servo_cfg = config.get("servo", {})
    viz_cfg = config.get("visualizer", {})

    cuda_device_ordinal = 0

    print("Initializing CUDA...")
    _cuda_check(cuda_driver.cuInit(0))
    cu_device = _cuda_check(cuda_driver.cuDeviceGet(cuda_device_ordinal))
    cu_context = _cuda_check(cuda_driver.cuDevicePrimaryCtxRetain(cu_device))
    cuda_driver.cuCtxSetCurrent(cu_context)

    print(f"Enumerating HSB board at {hololink_ip}...")
    metadata = hl.find_channel(hololink_ip)
    channel = hl.DataChannel(metadata)

    hololink_dev = channel.hololink()
    try:
        hololink_dev.start()
        hololink_dev.reset()

        mode_enum = hl.sensors.imx219.Imx219_Mode(camera_mode)
        camera = hl.sensors.imx219.Imx219Cam(channel, i2c_bus=i2c_bus, mode=mode_enum)
        camera.configure(mode_enum)

        pixel_format = camera.pixel_format()
        bayer_format = camera.bayer_format()
        frame_width = camera._width
        frame_height = camera._height
        geometry = hl.resolve_generic_geometry_from_dimensions(
            int(pixel_format), frame_width, frame_height
        )

        model_width = int(letterbox_cfg.get("width", 640))
        model_height = int(letterbox_cfg.get("height", 640))
        max_detections = int(postproc_cfg.get("max_detections", 100))
        show_boxes = bool(viz_cfg.get("show_boxes", True))
        box_rgba, box_line_width = box_overlay_style(viz_cfg)
        window_width = int(viz_cfg.get("width", frame_width))
        window_height = int(viz_cfg.get("height", frame_height))

        graph = hs.Graph("linux_imx219_face_detection")

        rx = graph.op(
            hl.operators.LinuxReceiverOp,
            "rx",
            data_channel=channel,
            frame_context=int(cu_context),
            device_start=camera.start,
            device_stop=camera.stop,
            frame_size=(
                geometry["start_byte"]
                + geometry["bytes_per_line"] * geometry["pixel_height"]
                + geometry["trailing_bytes"]
            ),
            frame_limit=frame_limit or 0,
            frame_limit_ends_run=True,
        )

        csi = graph.op(
            hl.operators.CsiToBayerOp,
            "csi_to_bayer",
            cuda_device_ordinal=cuda_device_ordinal,
            geometry=geometry,
            bayer_format=bayer_format.name.lower(),
        )
        graph.add_flow(
            rx.tensor,
            csi.in_tensor,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )
        graph.add_flow(
            rx.metadata,
            csi.in_metadata,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        image_processor = graph.op(
            hl.operators.ImageProcessorOp,
            "image_processor",
            pixel_format=int(pixel_format),
            pixel_width=geometry["pixel_width"],
            pixel_height=geometry["pixel_height"],
            bayer_format=int(bayer_format),
            optical_black=64,
            cuda_device_ordinal=cuda_device_ordinal,
        )
        graph.add_flow(
            csi.tensor,
            image_processor.in_tensor,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )
        graph.add_flow(
            csi.metadata,
            image_processor.in_metadata,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        demosaic = graph.op(
            holoscan_vision.operators.BayerDemosaicOp,
            "bayer_demosaic",
            bayer_grid_pos=bayer_format.name.lower(),
            generate_alpha=False,
        )
        graph.add_flow(
            image_processor.tensor,
            demosaic.input,
            queue_depth=4,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        exposure = graph.op(
            ExposureOp,
            "exposure",
            exposure=float(exposure_cfg.get("exposure", 1.0)),
            red_gain=float(exposure_cfg.get("red_gain", 1.05)),
            green_gain=float(exposure_cfg.get("green_gain", 0.95)),
            blue_gain=float(exposure_cfg.get("blue_gain", 1.05)),
        )
        graph.add_flow(
            demosaic.output,
            exposure.input,
            queue_depth=1,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        fps = graph.op(FpsOp, "fps")
        graph.add_flow(
            exposure.output,
            fps.input,
            queue_depth=1,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        letterbox = graph.op(
            LetterboxOp,
            "letterbox",
            width=model_width,
            height=model_height,
        )
        graph.add_flow(
            exposure.output,
            letterbox.input,
            queue_depth=1,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        normalize = graph.op(
            NormalizeOp,
            "normalize",
            width=model_width,
            height=model_height,
            mean=float(normalize_cfg.get("mean", 127.5)),
            std=float(normalize_cfg.get("std", 128.0)),
        )
        graph.add_flow(
            letterbox.output,
            normalize.input,
            queue_depth=1,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        inference = graph.op(
            "holoscan.inference.InferenceOp",
            "inference",
            model_path_map={"face_detector": model_path},
            input_map={"face_detector": ["source_video"]},
            output_map={"face_detector": list(_SCRFD_OUTPUT_PORTS)},
            backend=str(inference_cfg.get("backend", "trt")),
            input_on_cuda=bool(inference_cfg.get("input_on_cuda", True)),
            is_engine_path=bool(inference_cfg.get("is_engine_path", False)),
            enable_fp16=bool(inference_cfg.get("enable_fp16", True)),
        )
        graph.add_flow(
            normalize.output,
            inference.source_video,
            queue_depth=1,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
        )

        postprocessor = graph.op(
            ScrfdPostprocessOp,
            "postprocessor",
            model_width=int(postproc_cfg.get("model_width", model_width)),
            model_height=int(postproc_cfg.get("model_height", model_height)),
            image_width=int(postproc_cfg.get("image_width", frame_width)),
            image_height=int(postproc_cfg.get("image_height", frame_height)),
            letterbox_scale=float(postproc_cfg.get("letterbox_scale", 1.0 / 3.0)),
            score_threshold=float(postproc_cfg.get("score_threshold", 0.5)),
            nms_threshold=float(postproc_cfg.get("nms_threshold", 0.4)),
            max_detections=max_detections,
            line_width=box_line_width,
            box_r=box_rgba[0],
            box_g=box_rgba[1],
            box_b=box_rgba[2],
            box_a=box_rgba[3],
        )
        for port_name in _SCRFD_OUTPUT_PORTS:
            graph.add_flow(
                getattr(inference, port_name),
                getattr(postprocessor, f"in_{port_name}"),
                queue_depth=4,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
            )

        if headless:
            sink = {
                "kind": "cuda",
                "width": window_width,
                "height": window_height,
                "format": "rgba8_srgb",
            }
        else:
            sink = {
                "kind": "window",
                "framebuffer_srgb": True,
                "title": f"Holoviz IMX219 Face Detection - {hololink_ip}",
                "width": window_width,
                "height": window_height,
                "fullscreen": fullscreen,
                "close_on_escape": True,
            }

        layers = [
            {
                "port_name": "video",
                "kind": "image",
                "format": "rgb8_unorm",
                "max_width": frame_width,
                "max_height": frame_height,
            },
        ]
        if show_boxes:
            layers.append(
                {
                    "port_name": "boxes",
                    "kind": "image",
                    "format": "rgba8_unorm",
                    "max_width": frame_width,
                    "max_height": frame_height,
                }
            )

        viz = graph.op(
            "holoscan.visualization.HolovizOp",
            "holoviz",
            layers=layers,
            sink=sink,
            trigger={"kind": "layer", "layer": 0},
        )
        graph.add_flow(
            exposure.output,
            viz.video,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )
        if show_boxes:
            graph.add_flow(
                postprocessor.boxes,
                viz.boxes,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )
        else:
            discard_boxes = graph.op(DiscardImageOp, "discard_boxes")
            graph.add_flow(
                postprocessor.boxes,
                discard_boxes.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

        servo = graph.op(
            ServoControlOp,
            "servo",
            hololink=hololink_dev,
            enabled=servo_enabled,
            enable_pan=bool(servo_cfg.get("enable_pan", True)),
            enable_tilt=bool(servo_cfg.get("enable_tilt", True)),
            center_pulse=int(servo_cfg.get("center_pulse", 150000)),
            tilt_center_pulse=int(servo_cfg.get("tilt_center_pulse", 90000)),
            min_pulse=int(servo_cfg.get("min_pulse", 50000)),
            max_pulse=int(servo_cfg.get("max_pulse", 250000)),
            tilt_min_pulse=int(servo_cfg.get("tilt_min_pulse", 50000)),
            tilt_max_pulse=int(servo_cfg.get("tilt_max_pulse", 150000)),
            period_cycles=int(servo_cfg.get("period_cycles", 2000000)),
            control_hz=float(servo_cfg.get("control_hz", 15.0)),
            velocity_filter_alpha=float(servo_cfg.get("velocity_filter_alpha", 0.95)),
            command_amplifier=float(servo_cfg.get("command_amplifier", 0.9)),
            max_error_amplifier=float(servo_cfg.get("max_error_amplifier", 5.0)),
            pan_step_gain=float(servo_cfg.get("pan_step_gain", 4000.0)),
            pan_velocity_gain=float(servo_cfg.get("pan_velocity_gain", 1500.0)),
            pan_min_step=int(servo_cfg.get("pan_min_step", 10)),
            pan_max_step=int(servo_cfg.get("pan_max_step", 8000)),
            tilt_step_gain=float(servo_cfg.get("tilt_step_gain", 4000.0)),
            tilt_velocity_gain=float(servo_cfg.get("tilt_velocity_gain", 1500.0)),
            tilt_min_step=int(servo_cfg.get("tilt_min_step", 10)),
            tilt_max_step=int(servo_cfg.get("tilt_max_step", 10000)),
            deadband=float(servo_cfg.get("deadband", 0.04)),
            invert_pan=bool(servo_cfg.get("invert_pan", True)),
            invert_tilt=bool(servo_cfg.get("invert_tilt", True)),
            write_period=bool(servo_cfg.get("write_period", True)),
        )
        graph.add_flow(
            postprocessor.face_center,
            servo.input,
            queue_policy=hs.QueuePolicy.DROP_OLDEST,
            queue_depth=1,
        )

        if headless:
            discard = graph.op(DiscardImageOp, "discard")
            graph.add_flow(
                viz.output,
                discard.input,
                queue_policy=hs.QueuePolicy.DROP_OLDEST,
                queue_depth=1,
            )

        clock_key = graph.add_clock(hs.RealtimeClock, "clock")
        graph.set_default_clock(clock_key)

        options = hs.CompileOptions()
        for operator_path in ("rx", "csi_to_bayer", "image_processor"):
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="tensor",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )
        for operator_path in (
            "bayer_demosaic",
            "exposure",
            "letterbox",
            "normalize",
        ):
            options.deployment.bind_tensor_output_device(
                operator_path=operator_path,
                output_port="output",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )
        for output_port in _SCRFD_OUTPUT_PORTS:
            options.deployment.bind_tensor_output_device(
                operator_path="inference",
                output_port=output_port,
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )
        options.deployment.bind_tensor_output_device(
            operator_path="postprocessor",
            output_port="boxes",
            device=hs.execution.DeviceId(cuda_device_ordinal),
        )
        if headless:
            options.deployment.bind_tensor_output_device(
                operator_path="holoviz",
                output_port="output",
                device=hs.execution.DeviceId(cuda_device_ordinal),
            )

        plan = hs.compile(graph, options)
        if not plan.ok:
            print("Graph compilation failed:", file=sys.stderr)
            for diag in plan.diagnostics:
                print(f"  {diag}", file=sys.stderr)
            return 1

        print(
            f"Capturing IMX219 {frame_width}x{frame_height} -- running (Ctrl+C to stop)..."
        )

        with hs.run_async(plan) as session:
            failed = wait_for_completion(session)

        if failed:
            print("Run failed; see runtime diagnostics above.", file=sys.stderr)
            return 1
        print("Done.")
        return 0

    finally:
        try:
            if servo_enabled:
                try:
                    hololink_dev.write_uint32(SERVO_REG_CTRL, 0)
                except Exception as exc:
                    print(f"Servo disable on exit failed: {exc}")
            hololink_dev.stop()
        finally:
            _cuda_check(cuda_driver.cuDevicePrimaryCtxRelease(cu_device))


def main():
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    default_configuration = os.path.join(
        os.path.dirname(__file__), "linux_imx219_face_detection.yaml"
    )
    default_model = scrfd_model_path()

    parser = argparse.ArgumentParser(
        description="IMX219 Linux-socket Holoviz player with SCRFD face detection"
    )
    modes = hl.sensors.imx219.Imx219_Mode
    mode_choices = [mode.value for mode in modes]
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=mode_choices[0],
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--configuration", default=default_configuration)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--hololink", default="192.168.0.2")
    parser.add_argument("--log-level", type=int, default=20)
    parser.add_argument("--i2c-bus", type=int, default=None)
    parser.add_argument(
        "--servo",
        action="store_true",
        help="Enable Hololink APB servo writes (off by default for camera bring-up)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        parser.error(f"SCRFD model not found: {args.model}")

    logging.basicConfig(level=args.log_level)
    logging.info("Initializing IMX219 face-detection player.")

    return run(
        args.hololink,
        args.camera_mode,
        args.i2c_bus,
        os.path.abspath(args.model),
        args.configuration,
        args.frame_limit,
        args.headless,
        args.fullscreen,
        args.servo,
    )


if __name__ == "__main__":
    sys.exit(main())
