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
import queue
import sys
import threading
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
    "screen_corners": [[0, 0], [1279, 0], [1279, 719], [0, 719]],
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
    "mouse_click_debug": False,
    "no_preview": False,
    "control_window": {
        "enabled": False,
        "title": "Laser Mouse Control",
    },
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

class ControlPanel:
    """Small Windows control window for headless camera operation."""

    def __init__(
        self,
        title: str,
        mouse_enabled: bool,
        screen_corners: list[list[int]],
        screen_width: int,
        screen_height: int,
        preview_enabled: bool,
    ):
        self.title = title
        self.mouse_enabled = mouse_enabled
        self.screen_corners = screen_corners
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.preview_enabled = preview_enabled
        self.status = "Starting..."
        self.coordinates = "-"
        self.last_mouse_event = "No mouse click received"
        self.actions: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        self.root = None
        self.tk = None

    def start(self) -> None:
        self._create_ui()

    def _create_ui(self) -> None:
        try:
            import tkinter as tk
        except ImportError:
            with self.lock:
                self.status = "Tkinter is not available"
            return

        self.tk = tk
        root = tk.Tk()
        self.root = root
        root.title(self.title)
        root.resizable(False, False)
        root.attributes("-topmost", True)

        status_var = tk.StringVar(value="Starting...")
        coordinate_var = tk.StringVar(value="Coordinates: -")
        click_var = tk.StringVar(value="Last click: -")
        mouse_var = tk.BooleanVar(value=self.mouse_enabled)
        preview_var = tk.BooleanVar(value=self.preview_enabled)

        def set_mouse(value: bool) -> None:
            with self.lock:
                self.mouse_enabled = value
            status_var.set("Mouse: ON" if value else "Mouse: OFF")

        def toggle_mouse() -> None:
            set_mouse(not mouse_var.get())
            mouse_var.set(self.mouse_enabled)

        def emergency_stop() -> None:
            set_mouse(False)
            mouse_var.set(False)
            self.actions.put("stop")

        def exit_program() -> None:
            set_mouse(False)
            mouse_var.set(False)
            self.actions.put("exit")

        def save_settings() -> None:
            try:
                values = []
                for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
                    values.append([
                        int(entries[name][0].get()),
                        int(entries[name][1].get()),
                    ])
                settings = {
                    "screen_corners": values,
                    "screen_width": int(width_entry.get()),
                    "screen_height": int(height_entry.get()),
                }
                self.actions.put(("settings", settings))
                status_var.set("Calibration settings queued")
            except ValueError:
                status_var.set("Invalid calibration values")

        def toggle_preview() -> None:
            self.actions.put(("preview", bool(preview_var.get())))

        tk.Label(root, text="Laser Mouse Control", font=("Segoe UI", 12, "bold")).pack(
            padx=14, pady=(12, 4)
        )
        tk.Label(root, textvariable=status_var).pack(padx=14, pady=2)
        tk.Label(root, textvariable=coordinate_var).pack(padx=14, pady=2)
        tk.Label(root, textvariable=click_var).pack(padx=14, pady=2)
        tk.Checkbutton(
            root,
            text="Mouse tracking",
            variable=mouse_var,
            command=lambda: set_mouse(mouse_var.get()),
        ).pack(padx=14, pady=4)

        calibration_frame = tk.LabelFrame(root, text="Screen calibration")
        calibration_frame.pack(padx=14, pady=4, fill="x")
        headers = ("Point", "X", "Y")
        for column, header in enumerate(headers):
            tk.Label(calibration_frame, text=header).grid(row=0, column=column, padx=3, pady=2)
        entries = {}
        corner_names = ("top_left", "top_right", "bottom_right", "bottom_left")
        corner_labels = ("Top-left", "Top-right", "Bottom-right", "Bottom-left")
        for row, (name, label) in enumerate(zip(corner_names, corner_labels), start=1):
            tk.Label(calibration_frame, text=label).grid(row=row, column=0, sticky="w", padx=3)
            fields = []
            point = self.screen_corners[row - 1]
            for column, value in ((1, point[0]), (2, point[1])):
                entry = tk.Entry(calibration_frame, width=7)
                entry.insert(0, str(value))
                entry.grid(row=row, column=column, padx=3, pady=2)
                fields.append(entry)
            entries[name] = fields
        tk.Label(calibration_frame, text="Screen size").grid(row=5, column=0, sticky="w", padx=3, pady=2)
        width_entry = tk.Entry(calibration_frame, width=7)
        width_entry.insert(0, str(self.screen_width))
        width_entry.grid(row=5, column=1, padx=3)
        height_entry = tk.Entry(calibration_frame, width=7)
        height_entry.insert(0, str(self.screen_height))
        height_entry.grid(row=5, column=2, padx=3)
        tk.Button(calibration_frame, text="Save calibration", command=save_settings).grid(
            row=6, column=0, columnspan=3, padx=3, pady=4, sticky="ew"
        )
        tk.Checkbutton(
            root,
            text="Camera preview",
            variable=preview_var,
            command=toggle_preview,
        ).pack(padx=14, pady=4)

        tk.Button(root, text="EMERGENCY STOP", fg="white", bg="#b00020", command=emergency_stop).pack(
            padx=14, pady=(4, 12), fill="x"
        )
        tk.Button(root, text="EXIT PROGRAM", command=exit_program).pack(
            padx=14, pady=(0, 12), fill="x"
        )

        root.bind("<Escape>", lambda _event: emergency_stop())
        root.bind("m", lambda _event: toggle_mouse())
        root.protocol("WM_DELETE_WINDOW", exit_program)

        def refresh() -> None:
            with self.lock:
                status = self.status
                coordinates = self.coordinates
                last_mouse_event = self.last_mouse_event
                mouse_state = self.mouse_enabled
            status_var.set(status)
            coordinate_var.set(f"Coordinates: {coordinates}")
            click_var.set(f"Last click: {last_mouse_event}")
            mouse_var.set(mouse_state)
            root.after(100, refresh)

        refresh()

    def process_events(self) -> None:
        """Pump Tk events from the main thread."""
        if self.root is None:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.root = None
            self.actions.put("exit")

    def close(self) -> None:
        """Destroy Tk objects on the same thread that created them."""
        if self.root is not None:
            try:
                self.root.destroy()
            except self.tk.TclError:
                pass
            self.root = None

    def set_status(self, status: str, coordinates: str = "-") -> None:
        with self.lock:
            self.status = status
            self.coordinates = coordinates

    def notify_mouse_click(self, button: str) -> None:
        with self.lock:
            self.last_mouse_event = f"{button.upper()} CLICK"

    def is_mouse_enabled(self) -> bool:
        with self.lock:
            return self.mouse_enabled

    def consume_actions(self) -> list[str]:
        actions = []
        while True:
            try:
                actions.append(self.actions.get_nowait())
            except queue.Empty:
                return actions


class MouseClickMonitor:
    """Observe OS-level mouse clicks, including Bluetooth HID mouse events."""

    def __init__(self, panel: ControlPanel | None):
        self.panel = panel
        self.listener = None

    def start(self) -> None:
        try:
            from pynput import mouse as pynput_mouse
        except ImportError as error:
            raise RuntimeError(
                "Mouse click debug requires pynput. "
                "Install it with: python -m pip install pynput"
            ) from error

        def on_click(_x: int, _y: int, button: object, pressed: bool) -> None:
            if not pressed:
                return
            name = str(button).split(".")[-1]
            if name in ("left", "right") and self.panel is not None:
                self.panel.notify_mouse_click(name)

        self.listener = pynput_mouse.Listener(on_click=on_click)
        self.listener.daemon = True
        self.listener.start()

    def stop(self) -> None:
        if self.listener is not None:
            self.listener.stop()


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
    transform: np.ndarray | None,
    screen_corners: list[list[int]],
    screen_size: tuple[int, int],
) -> tuple[tuple[int, int] | None, bool]:
    if point is None or transform is None:
        return None, False
    screen_width, screen_height = screen_size
    if len(screen_corners) != 4:
        return None, False
    inside = cv2.pointPolygonTest(
        np.asarray(screen_corners, dtype=np.float32),
        (float(point[0]), float(point[1])),
        False,
    ) >= 0
    if not inside:
        return None, False
    mapped = cv2.perspectiveTransform(
        np.asarray([[[point[0], point[1]]]], dtype=np.float32), transform
    )[0, 0]
    return (
        max(0, min(screen_width - 1, round(float(mapped[0])))),
        max(0, min(screen_height - 1, round(float(mapped[1])))),
    ), True


def make_screen_transform(
    screen_corners: list[list[int]], screen_size: tuple[int, int]
) -> np.ndarray | None:
    if len(screen_corners) != 4:
        return None
    try:
        source = np.asarray(screen_corners, dtype=np.float32)
        if cv2.contourArea(source.reshape((-1, 1, 2))) <= 0:
            return None
        width, height = screen_size
        destination = np.asarray(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(source, destination)
    except (TypeError, ValueError, cv2.error):
        return None


def has_gui_support() -> bool:
    """Return whether this OpenCV build can create windows."""
    build_info = cv2.getBuildInformation()
    gui_section = build_info.split("GUI:", 1)
    if len(gui_section) < 2:
        return False
    gui_line = gui_section[1].splitlines()[0].strip().upper()
    return gui_line not in {"", "NONE"}


def close_preview_windows() -> None:
    """Close OpenCV windows without interrupting the camera loop."""
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except cv2.error:
        pass


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
    control_config = config.get("control_window", {})
    if not isinstance(control_config, dict):
        control_config = {}
    control_window_enabled = bool(control_config.get("enabled", False))
    control_title = str(control_config.get("title", "Laser Mouse Control"))
    screen_width = config["screen_width"] if args.screen_width is None else args.screen_width
    screen_height = config["screen_height"] if args.screen_height is None else args.screen_height
    try:
        screen_corners = [
            [int(point[0]), int(point[1])] for point in config["screen_corners"]
        ]
    except (KeyError, TypeError, ValueError, IndexError):
        print("screen_corners must contain four [x, y] points", file=sys.stderr)
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
    transform = make_screen_transform(screen_corners, (screen_width, screen_height))
    if transform is None:
        print("screen_corners must define a valid four-point quadrilateral", file=sys.stderr)
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
    control_panel = None
    if control_window_enabled:
        control_panel = ControlPanel(
            control_title,
            mouse_enabled_runtime,
            screen_corners,
            screen_width,
            screen_height,
            not no_preview,
        )
        control_panel.start()
        print("Control panel: ENABLED", file=sys.stderr)
    click_monitor = None
    if control_panel is not None and bool(config.get("mouse_click_debug", False)):
        try:
            click_monitor = MouseClickMonitor(control_panel)
            click_monitor.start()
            print("Mouse click debug: ENABLED", file=sys.stderr)
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
    should_exit = False

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
                transform,
                screen_corners,
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
            if control_panel is not None:
                control_panel.process_events()
            if control_panel is not None:
                requested_mouse = control_panel.is_mouse_enabled()
                if requested_mouse and mouse is None:
                    try:
                        mouse = MouseController()
                        mouse_enabled_runtime = True
                    except RuntimeError as error:
                        control_panel.set_status(str(error))
                elif not requested_mouse and mouse is not None:
                    mouse = None
                    mouse_enabled_runtime = False

            if mouse is not None:
                mouse.move_to(screen_point if inside_frame else None)

            if control_panel is not None:
                status = "Mouse: ON" if mouse is not None else "Mouse: OFF"
                if output_point is None:
                    status += " | Laser lost"
                elif not inside_frame:
                    status += " | Outside frame"
                control_panel.set_status(
                    status,
                    f"camera=({output_point[0]}, {output_point[1]}) "
                    f"screen=({screen_point[0]}, {screen_point[1]})"
                    if output_point is not None and screen_point is not None
                    else "-",
                )
                for action in control_panel.consume_actions():
                    if isinstance(action, tuple):
                        action_name, payload = action
                        if action_name == "settings":
                            new_corners = payload["screen_corners"]
                            new_width = payload["screen_width"]
                            new_height = payload["screen_height"]
                            new_transform = make_screen_transform(
                                new_corners, (new_width, new_height)
                            )
                            if new_transform is None:
                                control_panel.set_status("Invalid 4-point calibration")
                            else:
                                screen_corners = new_corners
                                screen_width = new_width
                                screen_height = new_height
                                transform = new_transform
                                config["screen_corners"] = screen_corners
                                config["screen_width"] = screen_width
                                config["screen_height"] = screen_height
                                save_config(args.config, config)
                                control_panel.set_status("Calibration saved")
                        elif action_name == "preview":
                            preview_enabled = bool(payload) and has_gui_support()
                            if not preview_enabled:
                                close_preview_windows()
                        continue
                    if action == "stop":
                        mouse = None
                        mouse_enabled_runtime = False
                        continue
                    if action == "exit":
                        mouse = None
                        mouse_enabled_runtime = False
                        should_exit = True
                        continue
                if should_exit:
                    break

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
            if preview_enabled:
                preview = (
                    cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if frame.ndim == 2
                    else frame.copy()
                )
                polygon = np.asarray(screen_corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(preview, [polygon], True, (0, 255, 255), 2)
                for index, corner in enumerate(screen_corners):
                    color = (0, 255, 0) if index == 0 else (255, 100, 0)
                    cv2.circle(preview, tuple(corner), 8, color, -1)
                    cv2.putText(
                        preview,
                        ("TL", "TR", "BR", "BL")[index],
                        (corner[0] + 10, corner[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
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
                    close_preview_windows()
                    print(
                        "Không thể mở cửa sổ OpenCV; tiếp tục ở chế độ không preview.",
                        file=sys.stderr,
                    )
    finally:
        if click_monitor is not None:
            click_monitor.stop()
        if control_panel is not None:
            control_panel.close()
        camera.release()
        close_preview_windows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
