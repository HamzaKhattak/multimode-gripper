"""Standalone Ubuntu camera discovery and capture helpers.

The module scans /dev/video* nodes, groups multiple nodes that belong to the same
physical camera, ignores Intel RealSense devices, prints each camera serial number,
and overlays that serial number on captured frames.

Only OpenCV, NumPy, and Python standard library modules are required.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import stat
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

VIDEO_DEVICE_PATTERN = "/dev/video*"
SYSFS_VIDEO_ROOT = Path("/sys/class/video4linux")
REALSENSE_MARKERS = ("realsense", "intel realsense", "intel(r) realsense")
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30.0
FRAME_READ_ATTEMPTS = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)
TEXT_BG_COLOR = (0, 0, 0)
TEXT_MARGIN = 10


@dataclass(frozen=True)
class VideoNodeInfo:
    device_path: str
    sysfs_path: Path
    node_name: str
    node_index: int
    camera_name: str
    serial_number: str
    physical_key: str


@dataclass(frozen=True)
class CameraInfo:
    serial_number: str
    camera_name: str
    primary_device: str
    video_devices: tuple[str, ...]


def list_video_device_nodes() -> List[str]:
    """Return sorted /dev/video* character devices present on the system."""
    devices = sorted(glob.glob(VIDEO_DEVICE_PATTERN))
    return [device for device in devices if _is_character_device(Path(device))]


def _is_character_device(path: Path) -> bool:
    try:
        return stat.S_ISCHR(path.stat().st_mode)
    except OSError:
        return False


def _read_file_text(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    return text or None


def _resolve_video_sysfs_directory(device_path: str) -> Optional[Path]:
    video_name = Path(device_path).name
    sysfs_video = SYSFS_VIDEO_ROOT / video_name
    if not sysfs_video.exists():
        return None

    try:
        resolved = sysfs_video.resolve()
    except OSError:
        return None

    return resolved if resolved.exists() else None


def _iter_sysfs_directories(start_path: Path) -> Iterable[Path]:
    current = start_path
    while True:
        yield current
        if current.parent == current:
            break
        current = current.parent


def _parse_uevent_value(uevent_path: Path, keys: Iterable[str]) -> Optional[str]:
    text = _read_file_text(uevent_path)
    if not text:
        return None

    prefixes = tuple(f"{key}=" for key in keys)
    for line in text.splitlines():
        if line.startswith(prefixes):
            _, value = line.split("=", 1)
            value = value.strip()
            if value:
                return value
    return None


def _read_first_existing(directory: Path, file_names: Iterable[str]) -> Optional[str]:
    for file_name in file_names:
        value = _read_file_text(directory / file_name)
        if value:
            return value
    return None


def _find_serial_in_sysfs(sysfs_path: Path) -> Optional[str]:
    serial_file_names = ("serial", "idSerial", "serialnumber", "usbserial")
    for directory in _iter_sysfs_directories(sysfs_path):
        serial = _read_first_existing(directory, serial_file_names)
        if serial:
            return serial

        serial = _parse_uevent_value(directory / "uevent", ("SERIAL", "ID_SERIAL_SHORT", "ID_SERIAL"))
        if serial:
            return serial
    return None


def _find_camera_name_in_sysfs(sysfs_path: Path) -> str:
    names: List[str] = []
    candidate_names = ("name", "product", "manufacturer", "interface")

    for directory in _iter_sysfs_directories(sysfs_path):
        for candidate_name in candidate_names:
            text = _read_file_text(directory / candidate_name)
            if text and text not in names:
                names.append(text)
        if names:
            break

    return " ".join(names) if names else sysfs_path.name


def _find_physical_camera_key(sysfs_path: Path) -> str:
    for directory in _iter_sysfs_directories(sysfs_path):
        if (directory / "busnum").is_file() and (directory / "devnum").is_file():
            return str(directory)
        if (directory / "serial").is_file():
            return str(directory)
    return str(sysfs_path.parent)


def _read_node_index(sysfs_path: Path) -> int:
    text = _read_file_text(sysfs_path / "index")
    if text is None:
        return 1_000_000

    try:
        return int(text)
    except ValueError:
        return 1_000_000


def _is_realsense_camera(sysfs_path: Path, camera_name: str) -> bool:
    metadata: List[str] = [camera_name]
    for directory in _iter_sysfs_directories(sysfs_path):
        for file_name in ("name", "product", "manufacturer", "interface"):
            text = _read_file_text(directory / file_name)
            if text:
                metadata.append(text)

    metadata_text = " ".join(metadata).lower()
    return any(marker in metadata_text for marker in REALSENSE_MARKERS)


def _build_video_node_info(device_path: str) -> Optional[VideoNodeInfo]:
    sysfs_path = _resolve_video_sysfs_directory(device_path)
    if sysfs_path is None:
        return None

    camera_name = _find_camera_name_in_sysfs(sysfs_path)
    if _is_realsense_camera(sysfs_path, camera_name):
        return None

    serial_number = _find_serial_in_sysfs(sysfs_path) or f"unknown:{Path(device_path).name}"
    return VideoNodeInfo(
        device_path=device_path,
        sysfs_path=sysfs_path,
        node_name=_read_file_text(sysfs_path / "name") or Path(device_path).name,
        node_index=_read_node_index(sysfs_path),
        camera_name=camera_name,
        serial_number=serial_number,
        physical_key=_find_physical_camera_key(sysfs_path),
    )


def _open_capture(device_path: str, width: int, height: int, fps: float) -> cv2.VideoCapture:
    backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
    capture = cv2.VideoCapture(device_path, backend)
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _read_frame(capture: cv2.VideoCapture, attempts: int = FRAME_READ_ATTEMPTS) -> Optional[np.ndarray]:
    for _ in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None and frame.size > 0:
            return frame
    return None


def _device_supports_capture(device_path: str, width: int, height: int, fps: float) -> bool:
    capture = _open_capture(device_path, width, height, fps)
    if not capture.isOpened():
        capture.release()
        return False

    try:
        return _read_frame(capture) is not None
    finally:
        capture.release()


def list_available_cameras(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: float = DEFAULT_FPS,
) -> List[CameraInfo]:
    """Return physical non-RealSense cameras that can be opened with OpenCV on Ubuntu."""
    grouped_nodes: Dict[str, List[VideoNodeInfo]] = {}

    for device_path in list_video_device_nodes():
        node_info = _build_video_node_info(device_path)
        if node_info is None:
            continue
        grouped_nodes.setdefault(node_info.physical_key, []).append(node_info)

    cameras: List[CameraInfo] = []
    for node_group in grouped_nodes.values():
        sorted_nodes = sorted(node_group, key=lambda node: (node.node_index, node.device_path))
        usable_nodes = [
            node for node in sorted_nodes if _device_supports_capture(node.device_path, width, height, fps)
        ]
        if not usable_nodes:
            continue

        primary_node = usable_nodes[0]
        cameras.append(
            CameraInfo(
                serial_number=primary_node.serial_number,
                camera_name=primary_node.camera_name,
                primary_device=primary_node.device_path,
                video_devices=tuple(node.device_path for node in sorted_nodes),
            )
        )

    cameras.sort(key=lambda camera: (camera.camera_name.lower(), camera.serial_number.lower()))
    return cameras


def print_available_cameras(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: float = DEFAULT_FPS,
) -> List[CameraInfo]:
    """Print the non-RealSense cameras that can be used with OpenCV."""
    cameras = list_available_cameras(width=width, height=height, fps=fps)
    if not cameras:
        print("No non-RealSense capture cameras found under /dev/video*.")
        return cameras

    for camera in cameras:
        print(
            f"serial={camera.serial_number} name={camera.camera_name} "
            f"primary={camera.primary_device} nodes={', '.join(camera.video_devices)}"
        )
    return cameras


def open_camera_by_serial(
    serial_number: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: float = DEFAULT_FPS,
) -> cv2.VideoCapture:
    """Open a non-RealSense camera by serial number with the requested settings."""
    for camera in list_available_cameras(width=width, height=height, fps=fps):
        if camera.serial_number != serial_number:
            continue

        capture = _open_capture(camera.primary_device, width, height, fps)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Camera with serial '{serial_number}' was found but could not be opened on {camera.primary_device}."
            )
        return capture

    raise ValueError(f"No non-RealSense camera found for serial '{serial_number}'.")


def capture_camera_frame_by_serial(
    serial_number: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: float = DEFAULT_FPS,
) -> np.ndarray:
    """Open a camera by serial number, grab one frame, and return it."""
    capture = open_camera_by_serial(serial_number, width=width, height=height, fps=fps)
    try:
        frame = _read_frame(capture)
    finally:
        capture.release()

    if frame is None:
        raise RuntimeError(f"Failed to capture a frame from camera '{serial_number}'.")
    return frame


def annotate_serial_on_image(image: np.ndarray, serial_number: str) -> np.ndarray:
    """Overlay the camera serial number on the image."""
    text = f"Serial: {serial_number}"
    text_size, baseline = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    text_width, text_height = text_size
    box_top_left = (TEXT_MARGIN, TEXT_MARGIN)
    box_bottom_right = (TEXT_MARGIN + text_width + 8, TEXT_MARGIN + text_height + baseline + 8)

    annotated = image.copy()
    cv2.rectangle(annotated, box_top_left, box_bottom_right, TEXT_BG_COLOR, cv2.FILLED)
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


def capture_all_camera_images(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: float = DEFAULT_FPS,
    show_images: bool = False,
    wait_ms: int = 0,
) -> Dict[str, np.ndarray]:
    """Capture one annotated frame from every non-RealSense camera."""
    results: Dict[str, np.ndarray] = {}
    cameras = print_available_cameras(width=width, height=height, fps=fps)

    for camera in cameras:
        frame = capture_camera_frame_by_serial(
            camera.serial_number,
            width=width,
            height=height,
            fps=fps,
        )
        annotated = annotate_serial_on_image(frame, camera.serial_number)
        results[camera.serial_number] = annotated

        if show_images:
            window_name = f"{camera.camera_name} [{camera.primary_device}]"
            cv2.imshow(window_name, annotated)

    if show_images and results:
        cv2.waitKey(wait_ms)
        cv2.destroyAllWindows()

    return results


if __name__ == "__main__":
    capture_all_camera_images(show_images=True)
