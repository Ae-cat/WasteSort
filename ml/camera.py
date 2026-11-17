"""Capture one camera frame for the WasteSort prediction pipeline."""
from pathlib import Path
import sys


def load_opencv():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from exc
    return cv2


def camera_error(message):
    if sys.platform == "darwin":
        message += (
            ". On macOS, grant Camera access to the terminal application in "
            "System Settings > Privacy & Security > Camera, then restart it"
        )
    return RuntimeError(message)


def read_frame(camera, camera_index=0):
    ok, frame = camera.read()
    if not ok or frame is None:
        raise camera_error(f"Could not read a frame from camera {camera_index}")
    return frame


def save_frame(frame, output_path, cv2):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not save camera frame to {output_path}")
    return output_path


def capture_frame(output_path, camera_index=0):
    cv2 = load_opencv()

    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise camera_error(f"Could not open camera {camera_index}")
        return save_frame(read_frame(camera, camera_index), output_path, cv2)
    finally:
        camera.release()
