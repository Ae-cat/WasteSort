"""Time capture, preprocessing, and inference on this machine.

Uses an untrained ResNet18 with 9 outputs (same speed as the real model), so it
needs no checkpoint. Run on the Pi with the camera connected:

    python ml/pi_speed_test.py --runs 10

Without a camera, add --no-camera to time preprocessing and inference only.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import build_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--no-camera", action="store_true")
    args = parser.parse_args()

    model = build_model(9, pretrained=False).eval()
    transform = ResNet18_Weights.DEFAULT.transforms()

    camera = None
    if not args.no_camera:
        from picamera2 import Picamera2

        camera = Picamera2()
        camera.configure(camera.create_still_configuration(main={"size": (1280, 720), "format": "RGB888"}))
        camera.start()
        time.sleep(2)  # let auto-exposure settle

    capture_ms, prep_ms, infer_ms = [], [], []
    for i in range(args.warmup + args.runs):
        t0 = time.perf_counter()
        if camera is not None:
            frame = camera.capture_array()[:, :, ::-1]  # BGR -> RGB
            image = Image.fromarray(frame)
        else:
            image = Image.new("RGB", (1280, 720), (128, 128, 128))
        t1 = time.perf_counter()
        tensor = transform(image).unsqueeze(0)
        t2 = time.perf_counter()
        with torch.inference_mode():
            model(tensor).softmax(dim=1)
        t3 = time.perf_counter()
        if i >= args.warmup:
            capture_ms.append((t1 - t0) * 1000)
            prep_ms.append((t2 - t1) * 1000)
            infer_ms.append((t3 - t2) * 1000)

    if camera is not None:
        camera.stop()

    def row(name, values):
        return f"| {name} | {statistics.mean(values):.0f} ms | {min(values):.0f}-{max(values):.0f} ms |"

    total = [a + b + c for a, b, c in zip(capture_ms, prep_ms, infer_ms)]
    print(f"\nRuns: {args.runs} (after {args.warmup} warm-up runs), torch {torch.__version__}, "
          f"threads {torch.get_num_threads()}")
    print("| Step | Average | Range |")
    print("|---|---|---|")
    print(row("Camera capture" if camera is not None else "Image creation (no camera)", capture_ms))
    print(row("Preprocessing", prep_ms))
    print(row("Model inference", infer_ms))
    print(row("Total per prediction", total))


if __name__ == "__main__":
    main()
