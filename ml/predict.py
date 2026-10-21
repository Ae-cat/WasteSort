"""Run the end-to-end WasteSort prediction pipeline from an image or camera."""
import argparse
from pathlib import Path

import torch

from camera import capture_frame
from model import load_checkpoint, predict_image


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--camera", type=int, metavar="INDEX")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet18_realwaste.pt"))
    parser.add_argument("--capture-output", type=Path, default=Path("captures/latest.jpg"))
    args = parser.parse_args()

    image_path = args.image if args.image else capture_frame(args.capture_output, args.camera)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, classes = load_checkpoint(args.checkpoint, device)
    result = predict_image(model, classes, image_path, device)
    print(f"image={image_path}")
    print(f"prediction={result['label']}")
    print(f"confidence={result['confidence']:.1%}")
    print(f"disposal_category={result['disposal_category']}")


if __name__ == "__main__":
    main()
