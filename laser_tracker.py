"""Read a camera stream and report the detected laser-point coordinates.

Usage:
    python laser_tracker.py
    python laser_tracker.py --camera 1 --color green

Press Q or Esc in the preview window to stop.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import msvcrt
from pathlib import Path
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from camera_sources import OpenCVCamera, RealSenseCamera


@dataclass(frozen=True)
class ColorRange:
    """HSV threshold range. OpenCV uses H in [0, 179]."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


COLOR_RANGES: dict[str, tuple[ColorRange, ...]] = {
    # Red wraps around the HSV hue range, therefore it needs two masks.
    "red": (
        ColorRange((0, 100, 140), (10, 255, 255)),
        ColorRange((170, 100, 140), (179, 255, 255)),
    ),
    "green": (ColorRange((35, 70, 100), (90, 255, 255)),),
    "blue": (ColorRange((90, 70, 100), (135, 255, 255)),),
    "ir": (),
}

DEFAULT_CONFIG = {
    "camera": 0,
    "source": "opencv",
    "color": "red",
    "camera_width": 1280,
    "camera_height": 720,
    "top_left": [0, 0],
    "bottom_right": [1279, 719],
    "screen_width": 1920,
    "screen_height": 1080,
    "min_area": 4.0,
    "max_area": 2000.0,
    "min_pointer_area": 20.0,
    "max_pointer_aspect": 2.5,
    "min_pointer_circularity": 0.45,
    "max_track_jump": 140.0,
    "max_lost_frames": 4,
    "fusion_max_distance": 40.0,
    "red_min_area": 5.0,
    "red_max_area": 1000.0,
    "background_subtraction": True,
    "background_history": 120,
    "background_var_threshold": 24.0,
    "background_learning_rate": 0.001,
    "smoothing": 0.35,
    "realsense_stream": "infrared",
    "realsense_index": 1,
    "realsense_fps": 30,
    "disable_emitter": True,
    "exposure": 1000,
    "gain": 16,
    "auto_exposure": False,
    "ir_min_value": 180,
    "ir_local_threshold": 40,
    "mouse_control": False,
    "no_preview": False,
}


class MouseController:
    """Move the Windows cursor using the native user32 API."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Direct mouse control is only supported on Windows")
        self.set_cursor_pos = ctypes.windll.user32.SetCursorPos
        self.last_position: tuple[int, int] | None = None

    def move_to(self, point: tuple[int, int] | None) -> None:
        if point is None or point == self.last_position:
            return
        self.set_cursor_pos(int(point[0]), int(point[1]))
        self.last_position = point


def get_cursor_position() -> tuple[int, int] | None:
    """Return the current Windows cursor position, when available."""
    if sys.platform != "win32":
        return None
    point = ctypes.wintypes.POINT()
    if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
        return point.x, point.y
    return None


def load_config(path: str) -> dict:
    config = dict(DEFAULT_CONFIG)
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        if isinstance(loaded, dict):
            config.update(loaded)
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as error:
        print(f"Invalid config file {path}: {error}", file=sys.stderr)
    return config


def save_config(path: str, config: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a laser point from a webcam and print its coordinates."
    )
    parser.add_argument("--camera", type=int, default=None, help="Camera index")
    parser.add_argument(
        "--source",
        choices=("opencv", "realsense"),
        default=None,
        help="Camera backend",
    )
    parser.add_argument(
        "--color",
        choices=sorted(COLOR_RANGES),
        default=None,
        help="Laser color",
    )
    parser.add_argument("--width", type=int, default=None, help="Requested camera width")
    parser.add_argument("--height", type=int, default=None, help="Requested camera height")
    parser.add_argument(
        "--min-area",
        type=float,
        default=None,
        help="Minimum detected blob area in pixels",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=None,
        help="Coordinate smoothing factor from 0 to 1",
    )
    parser.add_argument("--screen-width", type=int, default=None, help="Screen width in pixels")
    parser.add_argument("--screen-height", type=int, default=None, help="Screen height in pixels")
    parser.add_argument("--ir-min-value", type=int, default=None, help="Minimum IR brightness")
    parser.add_argument(
        "--mouse",
        dest="mouse_control",
        action="store_true",
        default=None,
        help="Move the Windows mouse cursor using the mapped screen coordinates",
    )
    parser.add_argument(
        "--no-mouse",
        dest="mouse_control",
        action="store_false",
        help="Disable direct mouse control",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        default=None,
        help="Do not open a preview window; coordinates are still printed",
    )
    return parser.parse_args()


def make_mask(frame: np.ndarray, color: str, ir_min_value: int) -> np.ndarray:
    if color == "ir":
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        # IR projector/screen areas can be broadly saturated.  A white
        # top-hat keeps small local highlights (the laser spot) while
        # suppressing large, uniformly bright regions.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        local_bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        mask = cv2.inRange(local_bright, ir_min_value, 255)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for color_range in COLOR_RANGES[color]:
        lower = np.array(color_range.lower, dtype=np.uint8)
        upper = np.array(color_range.upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    # Remove isolated noise and fill tiny holes in the laser blob.
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def find_red_point(
    frame: np.ndarray, min_area: float = 5.0, max_area: float = 1000.0
) -> tuple[int, int] | None:
    """Find a compact red pointer candidate in an RGB/BGR frame."""
    mask = make_mask(frame, "red", 0)
    return find_laser_point(mask, min_area, max_area, min_area, 2.5)


def find_laser_point(
    mask: np.ndarray, min_area: float, max_area: float,
    min_pointer_area: float, max_pointer_aspect: float,
    brightness: np.ndarray | None = None,
    preferred_point: tuple[int, int] | None = None,
    min_circularity: float = 0.45,
    max_track_jump: float = 140.0,
) -> tuple[int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(min_area, min_pointer_area) or area > max_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width == 0 or height == 0:
            continue
        aspect = max(width / height, height / width)
        if aspect > max_pointer_aspect:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area / (perimeter * perimeter)) if perimeter else 0
        if circularity < min_circularity:
            continue
        candidates.append(contour)
    if not candidates:
        return None
    if preferred_point is not None:
        px, py = preferred_point
        nearby = []
        for candidate in candidates:
            moments = cv2.moments(candidate)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            if (cx - px) ** 2 + (cy - py) ** 2 <= max_track_jump ** 2:
                nearby.append(candidate)
        if nearby:
            candidates = nearby
        else:
            return None
        def candidate_score(candidate: np.ndarray) -> float:
            moments = cv2.moments(candidate)
            cx = moments["m10"] / max(moments["m00"], 1)
            cy = moments["m01"] / max(moments["m00"], 1)
            distance = float(np.hypot(cx - px, cy - py))
            brightness_score = 0.0
            if brightness is not None:
                region = np.zeros_like(mask)
                cv2.drawContours(region, [candidate], -1, 255, -1)
                brightness_score = float(cv2.mean(brightness, mask=region)[0]) / 255.0
            return brightness_score * 2.0 - distance / max_track_jump

        contour = max(candidates, key=candidate_score)
    elif brightness is None:
        contour = max(candidates, key=cv2.contourArea)
    else:
        contour = max(
            candidates,
            key=lambda candidate: cv2.mean(brightness, mask=cv2.drawContours(
                np.zeros_like(mask), [candidate], -1, 255, -1
            ))[0],
        )
    area = cv2.contourArea(contour)

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return x, y


def print_coordinate(
    point: tuple[int, int] | None,
    frame_size: tuple[int, int],
    screen_point: tuple[int, int] | None,
    inside_frame: bool,
    screen_size: tuple[int, int],
    started_at: float,
    motion_px: float = 0.0,
) -> None:
    width, height = frame_size
    screen_width, screen_height = screen_size
    payload = {
        "timestamp": round(time.time() - started_at, 3),
        "detected": point is not None,
        "x": point[0] if point else None,
        "y": point[1] if point else None,
        "screen_x": screen_point[0] if screen_point else None,
        "screen_y": screen_point[1] if screen_point else None,
        "inside_frame": inside_frame,
        "screen_width": screen_width,
        "screen_height": screen_height,
        "frame_width": width,
        "frame_height": height,
        "motion_px": round(motion_px, 2),
        "moving": motion_px >= 1.0,
    }
    cursor = get_cursor_position()
    payload["mouse_x"] = cursor[0] if cursor else None
    payload["mouse_y"] = cursor[1] if cursor else None
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def map_to_screen(
    point: tuple[int, int] | None,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    screen_size: tuple[int, int],
) -> tuple[tuple[int, int] | None, bool]:
    if point is None:
        return None, False

    x1, y1 = top_left
    x2, y2 = bottom_right
    screen_width, screen_height = screen_size
    if x2 <= x1 or y2 <= y1:
        return None, False

    rx = (point[0] - x1) / (x2 - x1)
    ry = (point[1] - y1) / (y2 - y1)
    inside = 0 <= rx <= 1 and 0 <= ry <= 1
    if not inside:
        return None, False

    return (
        round(rx * (screen_width - 1)),
        round(ry * (screen_height - 1)),
    ), True


def has_gui_support() -> bool:
    """Return whether this OpenCV build can create windows."""
    build_info = cv2.getBuildInformation()
    gui_section = build_info.split("GUI:", 1)
    if len(gui_section) < 2:
        return False
    gui_line = gui_section[1].splitlines()[0].strip().upper()
    return gui_line not in {"", "NONE"}


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    args.camera = config["camera"] if args.camera is None else args.camera
    args.source = config["source"] if args.source is None else args.source
    args.color = config["color"] if args.color is None else args.color
    args.width = config["camera_width"] if args.width is None else args.width
    args.height = config["camera_height"] if args.height is None else args.height
    args.min_area = config["min_area"] if args.min_area is None else args.min_area
    max_area = float(config.get("max_area", 2000.0))
    min_pointer_area = float(config.get("min_pointer_area", 20.0))
    max_pointer_aspect = float(config.get("max_pointer_aspect", 2.5))
    min_circularity = float(config.get("min_pointer_circularity", 0.45))
    max_track_jump = float(config.get("max_track_jump", 140.0))
    args.smoothing = config["smoothing"] if args.smoothing is None else args.smoothing
    args.ir_min_value = config["ir_min_value"] if args.ir_min_value is None else args.ir_min_value
    mouse_enabled = (
        config["mouse_control"] if args.mouse_control is None else args.mouse_control
    )
    no_preview = config["no_preview"] if args.no_preview is None else args.no_preview
    screen_width = config["screen_width"] if args.screen_width is None else args.screen_width
    screen_height = config["screen_height"] if args.screen_height is None else args.screen_height
    try:
        top_left = (int(config["top_left"][0]), int(config["top_left"][1]))
        bottom_right = (
            int(config["bottom_right"][0]),
            int(config["bottom_right"][1]),
        )
    except (KeyError, TypeError, ValueError, IndexError):
        print("top_left and bottom_right must be [x, y] in config.json", file=sys.stderr)
        return 2
    if args.color not in COLOR_RANGES:
        print(f"Unsupported color in config: {args.color}", file=sys.stderr)
        return 2
    # A run must use exactly one imaging mode.  RealSense must also expose
    # the stream that matches the selected detector; never start IR and color
    # streams together or silently process the wrong one.
    realsense_stream = str(config.get("realsense_stream", "infrared")).lower()
    if args.source == "realsense":
        expected_stream = "infrared" if args.color == "ir" else "color"
        if realsense_stream != expected_stream:
            print(
                f"Color mode '{args.color}' requires RealSense stream "
                f"'{expected_stream}', got '{realsense_stream}'",
                file=sys.stderr,
            )
            return 2
    if screen_width <= 0 or screen_height <= 0:
        print("Screen width and height must be greater than 0", file=sys.stderr)
        return 2
    if bottom_right[0] <= top_left[0] or bottom_right[1] <= top_left[1]:
        print("bottom_right must be lower-right of top_left", file=sys.stderr)
        return 2
    if not 0 < args.smoothing <= 1:
        print("--smoothing must be greater than 0 and at most 1", file=sys.stderr)
        return 2

    mouse = None
    if mouse_enabled:
        try:
            mouse = MouseController()
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            return 2
        print("Direct mouse control: ENABLED", file=sys.stderr)

    mouse_enabled_runtime = mouse is not None

    try:
        if args.source == "realsense":
            camera = RealSenseCamera(
                stream=realsense_stream,
                stream_index=int(config["realsense_index"]),
                width=args.width,
                height=args.height,
                fps=int(config["realsense_fps"]),
                disable_emitter=bool(config["disable_emitter"]),
                exposure=float(config["exposure"]) if config.get("exposure") is not None else None,
                gain=float(config["gain"]) if config.get("gain") is not None else None,
                auto_exposure=bool(config.get("auto_exposure", False)),
                dual_stream=bool(config.get("dual_stream", False)),
            )
        else:
            camera = OpenCVCamera(args.camera, args.width, args.height)
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    dual_stream = isinstance(camera, RealSenseCamera) and bool(config.get("dual_stream", False))
    if dual_stream:
        ok, first_frame, first_rgb_frame = camera.read_dual()
    else:
        ok, first_frame = camera.read()
        first_rgb_frame = None
    if not ok or first_frame is None:
        camera.release()
        print("Cannot read first frame from camera", file=sys.stderr)
        return 1

    actual_height, actual_width = first_frame.shape[:2]
    preview_enabled = not no_preview and has_gui_support()
    if not no_preview and not preview_enabled:
        print(
            "OpenCV không có GUI support; chuyển sang chế độ không preview. "
            "Dùng opencv-python thay cho opencv-python-headless để xem cửa sổ.",
            file=sys.stderr,
        )

    print(
        f"Camera opened: {actual_width}x{actual_height}. "
        "Press Q or Esc to quit.",
        file=sys.stderr,
    )

    smoothed: tuple[float, float] | None = None
    previous_point: tuple[int, int] | None = None
    lost_frames = 0
    calibration_points: list[tuple[int, int]] = []
    print("Calibration: aim at top-left and press \\. Then aim at bottom-right and press \\.", file=sys.stderr)
    started_at = time.time()
    background_subtractor = None
    if args.color == "ir" and bool(config.get("background_subtraction", True)):
        background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=int(config.get("background_history", 120)),
            varThreshold=float(config.get("background_var_threshold", 24.0)),
            detectShadows=False,
        )

    try:
        while True:
            if first_frame is not None:
                ok, frame = True, first_frame
                rgb_frame = first_rgb_frame
                first_frame = None
                first_rgb_frame = None
            else:
                if dual_stream:
                    ok, frame, rgb_frame = camera.read_dual()
                else:
                    ok, frame = camera.read()
                    rgb_frame = None
            if not ok:
                print("Cannot read frame from camera", file=sys.stderr)
                break

            height, width = frame.shape[:2]
            threshold = (
                config.get("ir_local_threshold", 40)
                if args.color == "ir"
                else args.ir_min_value
            )
            mask = make_mask(frame, args.color, threshold)
            brightness = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.ndim == 3
                else frame
            )
            if background_subtractor is not None:
                foreground = background_subtractor.apply(
                    brightness,
                    learningRate=float(config.get("background_learning_rate", 0.001)),
                )
                mask = cv2.bitwise_and(mask, foreground)
            ir_point = find_laser_point(
                mask, args.min_area, max_area,
                min_pointer_area, max_pointer_aspect,
                brightness,
                smoothed and (round(smoothed[0]), round(smoothed[1])),
                min_circularity,
                max_track_jump,
            )
            red_point = None
            if rgb_frame is not None:
                red_point = find_red_point(
                    rgb_frame,
                    float(config.get("red_min_area", 5.0)),
                    float(config.get("red_max_area", 1000.0)),
                )
            point = ir_point
            if dual_stream and bool(config.get("fusion_require_red", True)):
                if ir_point is None or red_point is None:
                    point = None
                elif np.hypot(ir_point[0] - red_point[0], ir_point[1] - red_point[1]) > float(config.get("fusion_max_distance", 40.0)):
                    point = None

            if point is not None:
                lost_frames = 0
                if smoothed is None:
                    smoothed = (float(point[0]), float(point[1]))
                else:
                    alpha = args.smoothing
                    smoothed = (
                        alpha * point[0] + (1 - alpha) * smoothed[0],
                        alpha * point[1] + (1 - alpha) * smoothed[1],
                    )
                output_point = (round(smoothed[0]), round(smoothed[1]))
            else:
                lost_frames += 1
                if smoothed is not None and lost_frames <= int(config.get("max_lost_frames", 4)):
                    output_point = (round(smoothed[0]), round(smoothed[1]))
                else:
                    smoothed = None
                    output_point = None

            screen_point, inside_frame = map_to_screen(
                output_point,
                top_left,
                bottom_right,
                (screen_width, screen_height),
            )
            print_coordinate(
                output_point,
                (width, height),
                screen_point,
                inside_frame,
                (screen_width, screen_height),
                started_at,
                (
                    float(np.hypot(output_point[0] - previous_point[0], output_point[1] - previous_point[1]))
                    if output_point is not None and previous_point is not None
                    else 0.0
                ),
            )
            previous_point = output_point
            if mouse is not None:
                mouse.move_to(screen_point if inside_frame else None)

            # Console calibration works with --no-preview as well. The key is
            # deliberately handled only when a valid laser point is present.
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key.lower() == "m":
                    if mouse_enabled_runtime:
                        mouse_enabled_runtime = False
                        mouse = None
                    else:
                        try:
                            mouse = MouseController()
                            mouse_enabled_runtime = True
                        except RuntimeError as error:
                            print(str(error), file=sys.stderr)
                    print(
                        f"Mouse control: {'ON' if mouse_enabled_runtime else 'OFF'}",
                        file=sys.stderr,
                    )
                if key == "\\" and output_point is not None:
                    calibration_points.append(output_point)
                    if len(calibration_points) == 1:
                        config["top_left"] = list(output_point)
                        print(f"Mapped top-left: {output_point}. Aim bottom-right and press \\.", file=sys.stderr)
                    elif len(calibration_points) == 2:
                        config["bottom_right"] = list(output_point)
                        save_config(args.config, config)
                        print(f"Mapped bottom-right: {output_point}. Calibration saved.", file=sys.stderr)
                        calibration_points.clear()

            if preview_enabled:
                preview = (
                    cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if frame.ndim == 2
                    else frame.copy()
                )
                cv2.circle(preview, top_left, 8, (0, 255, 0), -1)
                cv2.circle(preview, bottom_right, 8, (0, 0, 255), -1)
                cv2.rectangle(preview, top_left, bottom_right, (0, 255, 255), 2)
                cv2.putText(
                    preview,
                    "TOP LEFT",
                    (top_left[0] + 10, top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    preview,
                    "BOTTOM RIGHT",
                    (bottom_right[0] - 160, bottom_right[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )
                if output_point is not None:
                    cv2.circle(preview, output_point, 12, (0, 255, 0), 2)
                    cv2.drawMarker(
                        preview,
                        output_point,
                        (0, 255, 0),
                        cv2.MARKER_CROSS,
                        24,
                        2,
                    )
                    cv2.putText(
                        preview,
                        f"x={output_point[0]} y={output_point[1]}",
                        (output_point[0] + 15, output_point[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                try:
                    preview_title = "IR Preview" if args.color == "ir" else "RGB Preview"
                    cv2.imshow(preview_title, preview)
                    cv2.imshow("Laser Mask", mask)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("m"), ord("M")):
                        if mouse_enabled_runtime:
                            mouse_enabled_runtime = False
                            mouse = None
                        else:
                            try:
                                mouse = MouseController()
                                mouse_enabled_runtime = True
                            except RuntimeError as error:
                                print(str(error), file=sys.stderr)
                        print(
                            f"Mouse control: {'ON' if mouse_enabled_runtime else 'OFF'}",
                            file=sys.stderr,
                        )
                    if key in (ord("q"), 27):
                        break
                except cv2.error:
                    preview_enabled = False
                    print(
                        "Không thể mở cửa sổ OpenCV; tiếp tục ở chế độ không preview.",
                        file=sys.stderr,
                    )
    finally:
        camera.release()
        if preview_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
