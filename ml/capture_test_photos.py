"""Capture labeled webcam test photos with the Pi camera.

Run on the Pi with the camera connected:

    python ml/capture_test_photos.py                  (Pi camera)
    python ml/capture_test_photos.py --camera opencv  (laptop webcam)

Pick a class number, then press Enter to take each photo. Photos go to
data/external_unseen/photos/ and each one is added to
data/external_unseen/manifest.csv (the format ml/test_images.py expects).
These photos are for TESTING only: never train on them.
"""
import argparse
import csv
import time
from datetime import date
from pathlib import Path

from PIL import Image

CLASSES = [
    "Cardboard", "Food Organics", "Glass", "Metal", "Miscellaneous Trash",
    "Paper", "Plastic", "Textile Trash", "Vegetation",
]
ROOT = Path("data/external_unseen")
PHOTOS = ROOT / "photos"
MANIFEST = ROOT / "manifest.csv"
SOURCE = "Pi Camera Module 3 photo by Ariella on " + date.today().isoformat() + ", demo mounting position"


def next_index(slug):
    existing = [int(p.stem.rsplit("_", 1)[1]) for p in PHOTOS.glob(slug + "_*.jpg")
                if p.stem.rsplit("_", 1)[1].isdigit()]
    return max(existing, default=0) + 1


def add_manifest_row(path, label):
    new_file = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["image_path", "true_label", "source"])
        writer.writerow([path, label, SOURCE])


def open_camera(kind):
    """Return (grab, close): grab() gives an RGB numpy frame."""
    if kind == "opencv":
        import cv2

        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            raise RuntimeError("Could not open camera 0")
        for _ in range(5):  # let auto-exposure settle
            capture.read()

        def grab():
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            return frame[:, :, ::-1]  # BGR -> RGB

        return grab, capture.release

    from picamera2 import Picamera2

    camera = Picamera2()
    camera.configure(camera.create_still_configuration(main={"size": (1280, 720), "format": "RGB888"}))
    camera.start()
    time.sleep(2)
    return (lambda: camera.capture_array()[:, :, ::-1]), camera.stop  # RGB888 is BGR in memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", choices=["picamera2", "opencv"], default="picamera2",
                        help="picamera2 = Pi camera (default), opencv = laptop webcam")
    args = parser.parse_args()
    global SOURCE
    if args.camera == "opencv":
        SOURCE = "Laptop webcam photo on " + date.today().isoformat()

    PHOTOS.mkdir(parents=True, exist_ok=True)
    grab, close_camera = open_camera(args.camera)

    label = None
    print("Classes:")
    for number, name in enumerate(CLASSES, 1):
        print(f"  {number}. {name}")
    print("Type a class number to choose it, Enter to take a photo, q to quit.")
    while True:
        prompt = f"[{label or 'no class chosen'}] > "
        answer = input(prompt).strip().lower()
        if answer == "q":
            break
        if answer.isdigit() and 1 <= int(answer) <= len(CLASSES):
            label = CLASSES[int(answer) - 1]
            print("Now photographing:", label)
            continue
        if answer != "":
            print("Type 1-9, Enter, or q.")
            continue
        if label is None:
            print("Choose a class number first.")
            continue
        slug = label.lower().replace(" ", "_")
        name = f"{slug}_{next_index(slug):02d}.jpg"
        frame = grab()
        Image.fromarray(frame).save(PHOTOS / name, quality=95)
        add_manifest_row("photos/" + name, label)
        print("Saved", name)
    close_camera()
    print("Photos:", len(list(PHOTOS.glob("*.jpg"))), "- manifest:", MANIFEST)


if __name__ == "__main__":
    main()
