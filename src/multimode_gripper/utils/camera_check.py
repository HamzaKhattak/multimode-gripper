"""Standalone camera helper for Linux.

This module enumerates available V4L2 cameras under /dev/video*, captures one frame from
each camera, and overlays the detected serial number on the image.

Only OpenCV and Python standard library modules are required.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

VIDEO_DEVICE_PATTERN = "/dev/video*"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)
TEXT_BG_COLOR = (0, 0, 0)
TEXT_MARGIN = 10


def list_video_devices() -> List[str]:
    """Return a sorted list of available V4L2 device paths."""
    devices = sorted(glob.glob(VIDEO_DEVICE_PATTERN))
    return [device for device in devices if Path(device).is_char_device()]


def _resolve_video_sysfs_directory(device_path: str) -> Optional[Path]:
    """Resolve the sysfs entry for a video device path."""
    video_name = Path(device_path).name
    sysfs_video = Path("/sys/class/video4linux") / video_name
    if not sysfs_video.exists():
        return None

    try:
        resolved = sysfs_video.resolve()
    except OSError:
        return None

    return resolved if resolved.exists() else None


def _read_file_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None


def _parse_uevent_for_serial(uevent_path: Path) -> Optional[str]:
    text = _read_file_text(uevent_path)
    if not text:
        return None

    for line in text.splitlines():
        if line.startswith("SERIAL=") or line.startswith("ID_SERIAL="):
            _, value = line.split("=", 1)
            if value:
                return value.strip()
    return None


def _find_serial_in_sysfs(sysfs_path: Path) -> Optional[str]:
    """Search parent sysfs directories for a serial number."""
    serial_file_names = ["serial", "idSerial", "serialnumber", "usbserial"]

    for directory in [sysfs_path] + list(sysfs_path.parents):
        for candidate_name in serial_file_names:
            candidate = directory / candidate_name
            if candidate.is_file():
                serial = _read_file_text(candidate)
                if serial:
                    return serial

        uevent_file = directory / "uevent"
        if uevent_file.is_file():
            serial = _parse_uevent_for_serial(uevent_file)
            if serial:
                return serial

    return None


def _find_camera_name_in_sysfs(sysfs_path: Path) -> Optional[str]:
    """Search sysfs for manufacturer/product/name when serial is unavailable."""
    names: List[str] = []
    candidate_names = ["product", "manufacturer", "name"]

    for directory in [sysfs_path] + list(sysfs_path.parents):
        for candidate_name in candidate_names:
            candidate = directory / candidate_name
            if candidate.is_file():
                text = _read_file_text(candidate)
                if text and text not in names:
                    names.append(text)
        if names:
            break

    return " ".join(names) if names else None


def get_camera_serial(device_path: str) -> str:
    """Return a camera serial number for the given Linux video device path."""
    sysfs_path = _resolve_video_sysfs_directory(device_path)
    if sysfs_path is None:
        return f"unknown serial ({device_path})"

    serial = _find_serial_in_sysfs(sysfs_path)
    if serial:
        return serial

    name = _find_camera_name_in_sysfs(sysfs_path)
    if name:
        return f"{name} (no serial)"

    return f"unknown serial ({device_path})"


def capture_camera_frame(
    device_path: str,
    width: int = 1280,
    height: int = 720,
    attempts: int = 5,
) -> Optional[np.ndarray]:
    """Capture one frame from a video device and return the BGR image."""
    flags = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
    capture = cv2.VideoCapture(device_path, flags)
    if not capture.isOpened():
        return None

    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(attempts):
            ret, frame = capture.read()
            if ret and frame is not None and frame.size > 0:
                return frame
    finally:
        capture.release()

    return None


def annotate_serial_on_image(image: np.ndarray, serial_text: str) -> np.ndarray:
    """Draw the serial number as text on top of an image."""
    if image is None:
        raise ValueError("Image must not be None")

    text = f"Serial: {serial_text}"
    text_size, baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    text_width, text_height = text_size
    box_tl = (TEXT_MARGIN, TEXT_MARGIN)
    box_br = (TEXT_MARGIN + text_width + 8, TEXT_MARGIN + text_height + baseline + 8)

    annotated = image.copy()
    cv2.rectangle(annotated, box_tl, box_br, TEXT_BG_COLOR, cv2.FILLED)
    cv2.putText(
        annotated,
        text,
        (TEXT_MARGIN + 4, TEXT_MARGIN + text_height + 2),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return annotated


def capture_all_camera_images(show_images: bool = False, wait_ms: int = 1000) -> Dict[str, np.ndarray]:
    """Capture one frame from every connected Linux camera and overlay serial text."""
    device_paths = list_video_devices()
    results: Dict[str, np.ndarray] = {}

    if not device_paths:
        print("No /dev/video* camera devices found.")
        return results

    for device_path in device_paths:
        serial = get_camera_serial(device_path)
        print(f"{device_path}: {serial}")

        frame = capture_camera_frame(device_path)
        if frame is None:
            print(f"Failed to capture frame from {device_path}")
            continue

        annotated = annotate_serial_on_image(frame, serial)
        results[device_path] = annotated

        if show_images:
            window_name = f"Camera {device_path}"
            cv2.imshow(window_name, annotated)

    if show_images and results:
        print("Press any key in one of the image windows to close.")
        cv2.waitKey(wait_ms)
        cv2.destroyAllWindows()

    return results


if __name__ == "__main__":
    capture_all_camera_images(show_images=True, wait_ms=2000)
