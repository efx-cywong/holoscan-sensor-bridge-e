# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IMX219 FuSa CoE player with SCRFD face detection.

FPGA path (CoE-capable bitstream): MIPI -> packetizer -> CoE.
Host follows stock fusa_coe_* pattern (configure_converter; packetizer in start()).

Pipeline:
  FusaCoeCapture -> PackedFormatConverter -> ImageProcessor -> BayerDemosaic -> Exposure
    |-> FPS -> Holoviz video
    `-> 640x640 letterbox -> Normalize -> SCRFD -> boxes -> Holoviz
                                       `-> face_center -> servo APB writes
"""

import argparse
import ctypes
import logging
import os
import time

import cupy as cp
import holoscan
import numpy as np
from holoscan import as_tensor
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.operators import BayerDemosaicOp, HolovizOp, InferenceOp

import hololink as hololink_module

# C1156HSB apb_servo_slave on Hololink external APB slot 1.
SERVO_APB_BASE = 0x20000000
SERVO_REG_CTRL = SERVO_APB_BASE + 0x00
SERVO_REG_PAN = SERVO_APB_BASE + 0x04
SERVO_REG_TILT = SERVO_APB_BASE + 0x08
SERVO_REG_PERIOD = SERVO_APB_BASE + 0x0C
SERVO_CTRL_ENABLE = 0x1
SERVO_CTRL_PAN_EN = 0x2
SERVO_CTRL_TILT_EN = 0x4
SERVO_CTRL_PAN_ONLY = SERVO_CTRL_ENABLE | SERVO_CTRL_PAN_EN
SERVO_CTRL_ALL = SERVO_CTRL_ENABLE | SERVO_CTRL_PAN_EN | SERVO_CTRL_TILT_EN


class ExposureOp(Operator):
    """Convert demosaiced uint16 RGB to uint8 and apply exposure/white balance."""

    def __init__(
        self,
        fragment,
        *args,
        tensor_name="camera",
        exposure=0.9,
        red_gain=1.9,
        green_gain=1.2,
        blue_gain=2.0,
        **kwargs,
    ):
        self.tensor_name = tensor_name
        self.exposure = exposure
        self.gains = cp.asarray(
            [red_gain, green_gain, blue_gain], dtype=cp.float32
        )
        self._logged = False
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        tensor = message.get("")
        if tensor is None:
            tensor = next(iter(message.values()))
        image = cp.asarray(tensor)

        if image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(f"Expected HWC RGB/RGBA, got shape {image.shape}")

        rgb = image[:, :, :3]
        if rgb.dtype == cp.uint16:
            # ImageProcessor stores RAW10-derived data in the upper bits.
            rgb = (rgb >> 8).astype(cp.float32)
        else:
            rgb = rgb.astype(cp.float32)

        output = cp.clip(rgb * self.exposure * self.gains, 0, 255).astype(cp.uint8)

        if not self._logged:
            self._logged = True
            print(
                "ExposureOp first frame:",
                f"input_shape={image.shape}",
                f"input_dtype={image.dtype}",
                f"output_shape={output.shape}",
                f"min={int(output.min())}",
                f"max={int(output.max())}",
            )

        op_output.emit({self.tensor_name: output}, "output")


class FpsOp(Operator):
    """Report end-to-end display FPS every two seconds."""

    def __init__(self, fragment, *args, **kwargs):
        self._count = 0
        self._start = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        now = time.perf_counter()
        if self._start is None:
            self._start = now
        self._count += 1
        elapsed = now - self._start
        if elapsed >= 2.0:
            print(f"end-to-end FPS: {self._count / elapsed:.1f}")
            self._count = 0
            self._start = now
        op_output.emit(message, "output")


class LetterboxOp(Operator):
    """Scale 1920x1080 RGB to 640x360 and place it in a 640x640 canvas."""

    def __init__(
        self,
        fragment,
        *args,
        tensor_name="camera",
        out_tensor_name="source_video",
        width=640,
        height=640,
        **kwargs,
    ):
        self.tensor_name = tensor_name
        self.out_tensor_name = out_tensor_name
        self.width = width
        self.height = height
        self._logged = False
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        rgb = cp.asarray(message[self.tensor_name])
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

        # The bring-up path is exactly 1920x1080 -> 640x360. Average each
        # 3x3 source block on the GPU, avoiding an OpenCV host round-trip.
        if (
            src_width == new_width * 3
            and src_height == new_height * 3
        ):
            resized = (
                rgb.reshape(new_height, 3, new_width, 3, 3)
                .mean(axis=(1, 3))
                .astype(cp.uint8)
            )
        else:
            # Generic nearest-neighbor fallback for other aspect ratios.
            x_index = cp.minimum(
                cp.arange(new_width) * src_width // new_width, src_width - 1
            )
            y_index = cp.minimum(
                cp.arange(new_height) * src_height // new_height, src_height - 1
            )
            resized = rgb[y_index[:, None], x_index[None, :]]

        canvas = cp.zeros((self.height, self.width, 3), dtype=cp.uint8)
        canvas[:new_height, :new_width] = resized
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

        # Only emit the image tensor. Mixing non-tensor values into the
        # message confuses InferenceOp's Entity packing.
        op_output.emit({self.out_tensor_name: canvas}, "output")


class NormalizeOp(Operator):
    """Convert uint8 HWC RGB to contiguous float32 NCHW SCRFD input."""

    def __init__(
        self,
        fragment,
        *args,
        tensor_name="source_video",
        mean=127.5,
        std=128.0,
        **kwargs,
    ):
        self.tensor_name = tensor_name
        self.mean = mean
        self.std = std
        self._logged = False
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        # Keep preprocessing on the GPU. Transpose produces a strided view,
        # so materialize contiguous NCHW data before TensorRT consumes it.
        image = cp.asarray(message[self.tensor_name]).astype(cp.float32, copy=False)
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

        output = Entity(context)
        output.add(as_tensor(model_input), self.tensor_name)
        op_output.emit(output, "output")


class ScrfdPostprocessOp(Operator):
    """Decode six-output SCRFD tensors, run NMS, and emit Holoviz boxes."""

    def __init__(
        self,
        fragment,
        *args,
        model_width=640,
        model_height=640,
        image_width=1920,
        image_height=1080,
        letterbox_scale=1.0 / 3.0,
        score_threshold=0.5,
        nms_threshold=0.4,
        max_detections=100,
        **kwargs,
    ):
        self.model_width = model_width
        self.model_height = model_height
        self.image_width = image_width
        self.image_height = image_height
        self.letterbox_scale = letterbox_scale
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self._anchor_centers = {}
        self._logged = False
        self._logged_detection = False
        self._logged_center = False
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

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
        areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(
            0.0, y2 - y1 + 1
        )
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

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        cuda_stream = op_input.receive_cuda_stream("input")
        candidates = []

        with cp.cuda.ExternalStream(cuda_stream):
            for stride in (8, 16, 32):
                scores = cp.asarray(message[f"score_{stride}"])[0, :, 0]
                distances = cp.asarray(message[f"bbox_{stride}"])[0] * stride
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
            detections[:, [0, 2]] = np.clip(
                detections[:, [0, 2]], 0, self.image_width
            )
            detections[:, [1, 3]] = np.clip(
                detections[:, [1, 3]], 0, self.image_height
            )
            valid = (
                (detections[:, 2] > detections[:, 0])
                & (detections[:, 3] > detections[:, 1])
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

        boxes = np.empty((detections.shape[0] * 2, 2), dtype=np.float32)
        if detections.size:
            boxes[0::2, 0] = detections[:, 0] / self.image_width
            boxes[0::2, 1] = detections[:, 1] / self.image_height
            boxes[1::2, 0] = detections[:, 2] / self.image_width
            boxes[1::2, 1] = detections[:, 3] / self.image_height

        # Emit a single point at the chosen target center so it's
        # unambiguous what we will drive the servos from.
        #
        # Target selection rule:
        # - if multiple faces are detected, trace the one with the
        #   largest bounding-box area.
        face_center = np.empty((0, 2), dtype=np.float32)
        if detections.size:
            # detections columns: [x1, y1, x2, y2, score] in pixel coords
            areas = (detections[:, 2] - detections[:, 0]) * (
                detections[:, 3] - detections[:, 1]
            )
            best_idx = int(np.argmax(areas))
            x1, y1, x2, y2, score = detections[best_idx]
            cx = (x1 + x2) * 0.5 / self.image_width
            cy = (y1 + y2) * 0.5 / self.image_height
            face_center = np.array([[cx, cy]], dtype=np.float32)

            if not self._logged_center:
                self._logged_center = True
                print(
                    "Scrfd face center:",
                    f"score={score:.3f}",
                    f"area_px={(x2 - x1) * (y2 - y1):.1f}",
                    f"center_px=({(x1 + x2) * 0.5:.1f}, {(y1 + y2) * 0.5:.1f})",
                    f"center_norm=({cx:.4f}, {cy:.4f})",
                )

        if not self._logged:
            self._logged = True
            print(
                "ScrfdPostprocessOp first frame:",
                f"detections={detections.shape[0]}",
                f"score_threshold={self.score_threshold}",
                f"nms_threshold={self.nms_threshold}",
            )

        op_output.emit({"boxes": boxes, "face_center": face_center}, "output")


class ServoControlOp(Operator):
    """Use largest-face center error to update pan/tilt PWM setpoints.

    face_center is normalized Holoviz/image coords in [0, 1].
    Image center is (0.5, 0.5). Positive error means the face is right/down
    of center; invert_* flips the commanded direction for the mount.
    """

    def __init__(
        self,
        fragment,
        hololink,
        *args,
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
        # After the one-shot home in start(), first face / reacquire must
        # continue from the live pose — never snap back to center.
        self._awaiting_reacquire = True
        self._skip_next_step = False
        self._homed_at_start = False
        super().__init__(fragment, *args, **kwargs)

    def _ctrl_bits(self):
        ctrl = SERVO_CTRL_ENABLE
        if self._enable_pan:
            ctrl |= SERVO_CTRL_PAN_EN
        if self._enable_tilt:
            ctrl |= SERVO_CTRL_TILT_EN
        return ctrl

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def start(self):
        """Move to center once at app start; not on first face detection."""
        if not self._enabled or self._homed_at_start:
            return
        self._pan_setpoint = self._center_pulse
        self._tilt_setpoint = self._tilt_center_pulse
        self._write_regs(
            self._ctrl_bits(), self._pan_setpoint, self._tilt_setpoint
        )
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
        """Apply a bounded proportional + velocity step to a setpoint."""
        if abs(error) < self._deadband:
            return setpoint

        signed = -error if invert else error
        signed_velocity = -velocity if invert else velocity
        # Increase the multiplier smoothly from the base value just outside
        # the deadband to max_error_amplifier at an image edge (|error|=0.5).
        normalized_error = np.clip(
            (abs(error) - self._deadband) / (0.5 - self._deadband), 0.0, 1.0
        )
        adaptive_amplifier = (
            self._command_amplifier
            + (self._max_error_amplifier - self._command_amplifier)
            * normalized_error
        )
        raw_step = (
            signed * step_gain + signed_velocity * velocity_gain
        ) * adaptive_amplifier
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
        """Seed software setpoints from the live PWM widths.

        Avoids forcing pan/tilt home (center pulses) when a face is first
        acquired or reacquired after a miss.
        """
        try:
            pan = int(self._hololink.read_uint32(SERVO_REG_PAN))
            tilt = int(self._hololink.read_uint32(SERVO_REG_TILT))
            self._pan_setpoint = self._clip_pulse(
                pan, self._min_pulse, self._max_pulse
            )
            self._tilt_setpoint = self._clip_pulse(
                tilt, self._tilt_min_pulse, self._tilt_max_pulse
            )
            self._last_pan = self._pan_setpoint
            self._last_tilt = self._tilt_setpoint
        except Exception as exc:
            print(f"ServoControlOp: keep software setpoints ({exc})")

    def _begin_acquire(self, error_x, error_y):
        """Prepare tracking at the current pose; do not snap to center.

        Used for first detection and for reacquire after a miss. Centering
        happens only once in start(), not here.
        """
        self._sync_setpoints_from_hardware()
        self._last_measurement_time = None
        self._filtered_velocity_x = 0.0
        self._filtered_velocity_y = 0.0
        self._last_error_x = error_x
        self._last_error_y = error_y
        self._awaiting_reacquire = False
        self._skip_next_step = True

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")
        if not self._enabled:
            return

        face_center = message.get("face_center")
        if face_center is None:
            return
        face_center = np.asarray(face_center)
        if face_center.size == 0:
            # Hold the last APB pulse widths and leave PWM enabled. A missed
            # face-detection frame must not recenter or disable the turret.
            self._awaiting_reacquire = True
            return

        cx, cy = float(face_center[0, 0]), float(face_center[0, 1])
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

        # First control tick after acquire: enable PWM at the current pose
        # only. Correction steps start on the following tick.
        if self._skip_next_step:
            self._skip_next_step = False
            self._write_regs(
                self._ctrl_bits(), self._pan_setpoint, self._tilt_setpoint
            )
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


class HoloscanApplication(holoscan.core.Application):
    """IMX219 HSB viewer with an independent SCRFD inference branch."""

    def __init__(
        self,
        headless,
        fullscreen,
        hololink_channel,
        camera,
        frame_limit,
        model_path,
        hololink,
        servo_enabled,
        interface,
        timeout,
    ):
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._hololink_channel = hololink_channel
        self._camera = camera
        self._frame_limit = frame_limit
        self._model_path = model_path
        self._hololink = hololink
        self._servo_enabled = servo_enabled
        self._interface = interface
        self._timeout = timeout

    def compose(self):
        metadata = self._hololink_channel.enumeration_metadata()
        interface = self._interface or metadata.get("interface")
        hsb_mac = metadata.get("mac_id")
        if not interface:
            raise RuntimeError("CoE interface is unavailable; pass --interface")
        if not hsb_mac:
            raise RuntimeError("Hololink MAC address is unavailable from enumeration")
        hsb_mac_bytes = list(bytes.fromhex(hsb_mac.replace(":", "")))
        logging.info("Using FuSa CoE:")
        logging.info(f"  Interface: {interface}")
        logging.info(f"  HSB MAC:   {hsb_mac}")

        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self, name="count", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()

        fusa_coe_capture = hololink_module.operators.FusaCoeCaptureOp(
            self,
            condition,
            name="fusa_coe_capture",
            interface=interface,
            mac_addr=hsb_mac_bytes,
            timeout=self._timeout,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )
        # Same as stock fusa_coe_* examples: camera.configure_converter sets
        # size/format; FusaCoeCaptureOp.start() programs the HSB packetizer.
        self._camera.configure_converter(fusa_coe_capture)

        packed_format_converter_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="packed_format_converter_pool",
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
        )
        packed_format_converter = hololink_module.operators.PackedFormatConverterOp(
            self,
            name="packed_format_converter",
            allocator=packed_format_converter_pool,
        )
        fusa_coe_capture.configure_converter(packed_format_converter)

        image_processor = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            optical_black=64,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        demosaic_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="demosaic_pool",
            storage_type=1,
            block_size=self._camera._width
            * 3
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
        )
        demosaic = BayerDemosaicOp(
            self,
            name="demosaic",
            pool=demosaic_pool,
            generate_alpha=False,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        exposure = ExposureOp(self, name="exposure", **self.kwargs("exposure"))
        fps = FpsOp(self, name="fps")
        letterbox = LetterboxOp(
            self, name="letterbox", **self.kwargs("letterbox")
        )
        normalize = NormalizeOp(
            self, name="normalize", **self.kwargs("normalize")
        )

        # Working HSB inference examples use UnboundedAllocator, not RMM.
        # RMM pool pressure under dual video+inference paths can leave
        # Expected empty inside transmit_data_per_model.
        inference_pool = holoscan.resources.UnboundedAllocator(
            self, name="inference_pool"
        )
        inference = InferenceOp(
            self,
            name="inference",
            allocator=inference_pool,
            model_path_map={"face_detector": self._model_path},
            **self.kwargs("inference"),
        )
        cam_w = self._camera._width
        cam_h = self._camera._height
        post_args = dict(self.kwargs("postprocessor"))
        model_w = int(post_args.get("model_width", 640))
        model_h = int(post_args.get("model_height", 640))
        post_args["image_width"] = cam_w
        post_args["image_height"] = cam_h
        post_args["letterbox_scale"] = min(model_w / cam_w, model_h / cam_h)
        postprocessor = ScrfdPostprocessOp(
            self, name="postprocessor", **post_args
        )
        servo = ServoControlOp(
            self,
            self._hololink,
            name="servo",
            enabled=self._servo_enabled,
            **self.kwargs("servo"),
        )

        visualizer_args = dict(self.kwargs("visualizer"))
        show_boxes = bool(visualizer_args.pop("show_boxes", True))
        visualizer_args["width"] = cam_w
        visualizer_args["height"] = cam_h
        if not show_boxes:
            visualizer_args["tensors"] = [
                tensor
                for tensor in visualizer_args.get("tensors", [])
                if tensor.get("name") != "boxes"
            ]

        visualizer = HolovizOp(
            self,
            name="visualizer",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            **visualizer_args,
        )

        # Dual Holoviz inputs (camera + boxes) break with default metadata.
        self.enable_metadata(False)

        self.add_flow(
            fusa_coe_capture, packed_format_converter, {("output", "input")}
        )
        self.add_flow(
            packed_format_converter, image_processor, {("output", "input")}
        )
        self.add_flow(image_processor, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, exposure, {("transmitter", "input")})

        self.add_flow(exposure, fps, {("output", "input")})
        self.add_flow(fps, visualizer, {("output", "receivers")})

        self.add_flow(exposure, letterbox, {("output", "input")})
        self.add_flow(letterbox, normalize, {("output", "input")})
        self.add_flow(normalize, inference, {("output", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "input")})
        if show_boxes:
            self.add_flow(postprocessor, visualizer, {("output", "receivers")})
        self.add_flow(postprocessor, servo, {("output", "input")})


def main():
    default_configuration = os.path.join(
        os.path.dirname(__file__), "coe_imx219_face_detection.yaml"
    )
    default_model = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "python",
            "holoscan_py",
            "model",
            "500m.onnx",
        )
    )

    parser = argparse.ArgumentParser(
        description="IMX219 FuSa CoE Holoviz player with SCRFD face detection"
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
        "--interface",
        default=None,
        help="Thor CoE interface; default is the interface found by enumeration",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1500,
        help="FuSa capture timeout in milliseconds",
    )
    parser.add_argument(
        "--servo",
        action="store_true",
        help="Enable Hololink APB servo writes (off by default for camera bring-up)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        parser.error(f"SCRFD model not found: {args.model}")

    hololink_module.logging_level(args.log_level)
    logging.info("Initializing IMX219 FuSa CoE face-detection player.")

    metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    hololink_channel = hololink_module.DataChannel(metadata)
    camera = hololink_module.sensors.imx219.Imx219Cam(
        hololink_channel, i2c_bus=args.i2c_bus
    )

    hololink = hololink_channel.hololink()
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        hololink_channel,
        camera,
        args.frame_limit,
        os.path.abspath(args.model),
        hololink,
        servo_enabled=args.servo,
        interface=args.interface,
        timeout=args.timeout,
    )
    application.config(args.configuration)

    hololink.start()
    try:
        hololink.reset()
        camera.configure()
        application.run()
    finally:
        try:
            hololink.write_uint32(SERVO_REG_CTRL, 0)
        except Exception:
            pass
        hololink.stop()

if __name__ == "__main__":
    main()
