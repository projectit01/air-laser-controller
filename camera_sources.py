"""Camera backends used by the Windows laser tracker."""

from __future__ import annotations

import cv2
import numpy as np


class OpenCVCamera:
    def __init__(self, camera_index: int, width: int, height: int) -> None:
        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open OpenCV camera index {camera_index}")
        # Keep the driver from growing a queue of old frames.  A large
        # buffered queue makes the preview appear to replay/loop delayed
        # camera footage, especially when processing is slower than capture.
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


class RealSenseCamera:
    def __init__(
        self,
        stream: str,
        stream_index: int,
        width: int,
        height: int,
        fps: int,
        disable_emitter: bool,
        exposure: float | None = None,
        gain: float | None = None,
        auto_exposure: bool = False,
        dual_stream: bool = False,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "RealSense backend requires pyrealsense2. "
                "Install it with: python -m pip install pyrealsense2"
            ) from error

        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()

        self.dual_stream = dual_stream
        if stream == "infrared" and dual_stream:
            config.enable_stream(rs.stream.infrared, stream_index, width, height, rs.format.y8, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        elif stream == "infrared":
            config.enable_stream(
                rs.stream.infrared,
                stream_index,
                width,
                height,
                rs.format.y8,
                fps,
            )
        elif stream == "color":
            config.enable_stream(
                rs.stream.color,
                width,
                height,
                rs.format.bgr8,
                fps,
            )
        elif stream == "depth":
            config.enable_stream(
                rs.stream.depth,
                width,
                height,
                rs.format.z16,
                fps,
            )
        else:
            raise ValueError(f"Unsupported RealSense stream: {stream}")

        self.stream = stream
        self.stream_index = stream_index
        self.pipeline.start(config)
        self.aligner = rs.align(rs.stream.infrared) if dual_stream else None

        if disable_emitter:
            self._disable_emitter()
        self._set_manual_exposure(exposure, gain, auto_exposure)

    def _set_manual_exposure(
        self, exposure: float | None, gain: float | None, auto_exposure: bool
    ) -> None:
        device = self.pipeline.get_active_profile().get_device()
        for sensor in device.query_sensors():
            try:
                if sensor.supports(self.rs.option.enable_auto_exposure):
                    sensor.set_option(
                        self.rs.option.enable_auto_exposure,
                        1 if auto_exposure else 0,
                    )
                if not auto_exposure and exposure is not None and sensor.supports(self.rs.option.exposure):
                    sensor.set_option(self.rs.option.exposure, float(exposure))
                if not auto_exposure and gain is not None and sensor.supports(self.rs.option.gain):
                    sensor.set_option(self.rs.option.gain, float(gain))
            except self.rs.error:
                pass

    def _disable_emitter(self) -> None:
        device = self.pipeline.get_active_profile().get_device()
        for sensor in device.query_sensors():
            try:
                if sensor.supports(self.rs.option.emitter_enabled):
                    sensor.set_option(self.rs.option.emitter_enabled, 0)
            except self.rs.error:
                pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        frames = self.pipeline.wait_for_frames()
        if self.aligner is not None:
            frames = self.aligner.process(frames)
        if self.stream == "infrared":
            frame = frames.get_infrared_frame(self.stream_index)
        elif self.stream == "color":
            frame = frames.get_color_frame()
        else:
            frame = frames.get_depth_frame()
        if not frame:
            return False, None
        return True, np.asanyarray(frame.get_data())

    def read_dual(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Read synchronized infrared and color frames when dual_stream is enabled."""
        if not self.dual_stream:
            ok, frame = self.read()
            return ok, frame, None
        frames = self.pipeline.wait_for_frames()
        if self.aligner is not None:
            frames = self.aligner.process(frames)
        infrared = frames.get_infrared_frame(self.stream_index)
        color = frames.get_color_frame()
        if not infrared or not color:
            return False, None, None
        return True, np.asanyarray(infrared.get_data()), np.asanyarray(color.get_data())

    def release(self) -> None:
        self.pipeline.stop()
