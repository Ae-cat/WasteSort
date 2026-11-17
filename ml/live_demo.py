"""Preview laptop webcam 0 and classify a fresh capture on Space, using CPU."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

from camera import camera_error, load_opencv, read_frame, save_frame
from model import load_checkpoint, predict_image


WINDOW = "WasteSort | Space: predict | Q: quit"


def run_demo(checkpoint, capture_output=Path("captures/latest.jpg"), *, timing=False):
    cv2 = load_opencv()
    camera = None
    try:
        model, classes = load_checkpoint(checkpoint, "cpu")
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise camera_error("Could not open camera 0")
        for _ in range(5):
            read_frame(camera)
        print("Center the item in good light. Focus the preview window. Space: predict; Q: quit.")
        while True:
            cv2.imshow(WINDOW, read_frame(camera))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" "):
                # Capture after the key press so preview/key wait is not latency.
                started = perf_counter()
                frame = read_frame(camera)
                captured = perf_counter()
                image_path = save_frame(frame, capture_output, cv2)
                saved = perf_counter()
                timings = {} if timing else None
                result = predict_image(model, classes, image_path, "cpu", timings=timings)
                finished = perf_counter()
                print(
                    f"prediction={result['label']} "
                    f"disposal_category={result['disposal_category']} "
                    f"confidence={result['confidence']:.1%}",
                    flush=True,
                )
                if timing:
                    timings.update(camera_read=captured - started, file_write=saved - captured)
                    total = finished - started
                    timings["other"] = max(0.0, total - sum(timings.values()))
                    timings["total"] = total
                    print("timing_ms " + " ".join(
                        f"{name}={seconds * 1000:.3f}" for name, seconds in timings.items()
                    ), flush=True)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except cv2.error as exc:
        raise camera_error(f"OpenCV camera or preview operation failed: {exc}") from exc
    finally:
        handling_error = sys.exc_info()[0] is not None
        cleanup_error = None
        try:
            if camera is not None:
                camera.release()
        except cv2.error as exc:
            cleanup_error = exc
        try:
            cv2.destroyAllWindows()
        except cv2.error as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if cleanup_error is not None and not handling_error:
            raise camera_error(f"OpenCV resource cleanup failed: {cleanup_error}") from cleanup_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet18_realwaste.pt"))
    parser.add_argument("--capture-output", type=Path, default=Path("captures/latest.jpg"))
    parser.add_argument("--timing", action="store_true", help="Print CPU stage durations in milliseconds")
    args = parser.parse_args()
    try:
        run_demo(args.checkpoint, args.capture_output, timing=args.timing)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
