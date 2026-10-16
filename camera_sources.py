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

        if stream == "infrared":
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

        if disable_emitter:
            self._disable_emitter()

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
        if self.stream == "infrared":
            frame = frames.get_infrared_frame(self.stream_index)
        elif self.stream == "color":
            frame = frames.get_color_frame()
        else:
            frame = frames.get_depth_frame()
        if not frame:
            return False, None
        return True, np.asanyarray(frame.get_data())

    def release(self) -> None:
        self.pipeline.stop()
