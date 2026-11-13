# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IMX219 Linux-socket player with TAO PeopleNet detection (no servo).

Same PeopleNet ISP/infer path as coe_imx219_tao_peoplenet.py; capture differs:

  CoE:   FusaCoeCapture -> PackedFormatConverter  (packetizer on)
  Linux: LinuxReceiver  -> CsiToBayer             (CSI RAW10 layout)

Pipeline:
  LinuxReceiver -> CsiToBayer -> ImageProcessor -> BayerDemosaic -> Exposure
    |-> FPS -> Holoviz video
    `-> FormatConverter 960x544 [0,1] -> NCHW -> PeopleNet -> person/faces -> Holoviz
"""

import argparse
import ctypes
import logging
import os
import time

import cuda.bindings.driver as cuda
import cupy as cp
import holoscan
from holoscan.operators import BayerDemosaicOp, HolovizOp, InferenceOp
from tao_peoplenet import FormatInferenceInputOp, PostprocessorOp

import hololink as hololink_module


class ExposureOp(holoscan.core.Operator):
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

    def setup(self, spec):
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


class FpsOp(holoscan.core.Operator):
    """Report end-to-end display FPS every two seconds."""

    def __init__(self, fragment, *args, **kwargs):
        self._count = 0
        self._start = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
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


class HoloscanApplication(holoscan.core.Application):
    """IMX219 Linux-socket viewer with an independent PeopleNet inference branch."""

    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        frame_limit,
        model_path,
    ):
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._camera = camera
        self._frame_limit = frame_limit
        self._model_path = model_path
        self.enable_metadata(False)

    def compose(self):
        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self, name="count", count=self._frame_limit
            )
        else:
            condition = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )

        csi_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="csi_pool",
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=4,
        )
        csi_to_bayer = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer)

        receiver = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition,
            name="receiver",
            frame_size=csi_to_bayer.get_csi_length(),
            frame_context=self._cuda_context,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()
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

        pool = holoscan.resources.UnboundedAllocator(self, name="inference_pool")
        preprocessor_args = self.kwargs("preprocessor")
        preprocessor = holoscan.operators.FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            **preprocessor_args,
        )
        format_input = FormatInferenceInputOp(
            self,
            name="transpose",
            pool=pool,
        )
        inference = InferenceOp(
            self,
            name="inference",
            allocator=pool,
            model_path_map={"face_detect": self._model_path},
            **self.kwargs("inference"),
        )
        postprocessor_args = dict(self.kwargs("postprocessor"))
        postprocessor_args["image_width"] = preprocessor_args["resize_width"]
        postprocessor_args["image_height"] = preprocessor_args["resize_height"]
        postprocessor = PostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            **postprocessor_args,
        )

        visualizer_args = dict(self.kwargs("visualizer"))
        show_boxes = bool(visualizer_args.pop("show_boxes", True))
        visualizer_args["width"] = self._camera._width
        visualizer_args["height"] = self._camera._height
        if not show_boxes:
            visualizer_args["tensors"] = [
                tensor
                for tensor in visualizer_args.get("tensors", [])
                if tensor.get("name") in ("", "camera")
            ]

        visualizer = HolovizOp(
            self,
            name="visualizer",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            **visualizer_args,
        )

        self.add_flow(receiver, csi_to_bayer, {("output", "input")})
        self.add_flow(csi_to_bayer, image_processor, {("output", "input")})
        self.add_flow(image_processor, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, exposure, {("transmitter", "input")})

        self.add_flow(exposure, fps, {("output", "input")})
        self.add_flow(fps, visualizer, {("output", "receivers")})

        self.add_flow(exposure, preprocessor, {("output", "")})
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in")})
        if show_boxes:
            self.add_flow(postprocessor, visualizer, {("out", "receivers")})


def main():
    default_configuration = os.path.join(
        os.path.dirname(__file__), "linux_imx219_tao_peoplenet.yaml"
    )
    default_model = os.path.join(
        os.path.dirname(__file__), "models", "resnet34_peoplenet_int8.onnx"
    )
    if not os.path.isfile(default_model):
        default_model = os.path.join(
            os.path.dirname(__file__), "resnet34_peoplenet_int8.onnx"
        )

    parser = argparse.ArgumentParser(
        description="IMX219 Linux-socket Holoviz player with TAO PeopleNet (no servo)"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--frame-limit", type=int, default=None)
    parser.add_argument("--configuration", default=default_configuration)
    parser.add_argument(
        "--engine",
        "--model",
        dest="engine",
        default=default_model,
        help="PeopleNet ONNX / TensorRT engine path",
    )
    parser.add_argument("--hololink", default="192.168.0.2")
    parser.add_argument("--log-level", type=int, default=20)
    parser.add_argument("--i2c-bus", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.engine):
        parser.error(
            f"PeopleNet model not found: {args.engine}\n"
            "Download with:\n"
            "  wget --content-disposition "
            "'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/"
            "pruned_quantized_decrypted_v2.3.3/files?redirect=true"
            "&path=resnet34_peoplenet_int8.onnx' "
            "-O examples/models/resnet34_peoplenet_int8.onnx"
        )

    hololink_module.logging_level(args.log_level)
    logging.info("Initializing IMX219 Linux-socket PeopleNet player.")

    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cuda_device_ordinal = 0
    cu_result, cuda_device = cuda.cuDeviceGet(cuda_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cuda_context = cuda.cuDevicePrimaryCtxRetain(cuda_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    hololink_channel = hololink_module.DataChannel(metadata)
    camera = hololink_module.sensors.imx219.Imx219Cam(
        hololink_channel, i2c_bus=args.i2c_bus
    )

    hololink = hololink_channel.hololink()
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        args.frame_limit,
        os.path.abspath(args.engine),
    )
    application.config(args.configuration)

    hololink.start()
    try:
        hololink.reset()
        camera.configure()
        application.run()
    finally:
        hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cuda_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
