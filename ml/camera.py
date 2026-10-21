"""Capture one camera frame for the WasteSort prediction pipeline."""
from pathlib import Path


def capture_frame(output_path, camera_index=0):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError(f"Could not read a frame from camera {camera_index}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), frame):
            raise RuntimeError(f"Could not save camera frame to {output_path}")
        return output_path
    finally:
        camera.release()
