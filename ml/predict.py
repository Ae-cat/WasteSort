"""Run the end-to-end WasteSort prediction pipeline from an image or camera."""
import argparse
from pathlib import Path

import torch

from camera import capture_frame
from model import load_checkpoint, predict_image
from prediction_log import prediction_record, sha256, write_predictions


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--camera", type=int, metavar="INDEX")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet18_realwaste.pt"))
    parser.add_argument("--capture-output", type=Path, default=Path("captures/latest.jpg"))
    parser.add_argument("--log", type=Path, help="New CSV file for this prediction")
    parser.add_argument("--true-label", help="Optional known RealWaste label")
    args = parser.parse_args()
    if args.log and args.log.exists():
        parser.error("Log already exists; choose a new file")

    image_path = args.image if args.image else capture_frame(args.capture_output, args.camera)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, classes = load_checkpoint(args.checkpoint, device)
    if args.true_label and args.true_label not in classes:
        parser.error("Unknown true label")
    result = predict_image(model, classes, image_path, device)
    if args.log:
        write_predictions(args.log, [prediction_record(
            image_path, args.true_label, result["label"], result["confidence"],
            "single_image_unverified", args.checkpoint, sha256(args.checkpoint))])
    print(f"image={image_path}")
    print(f"prediction={result['label']}")
    print(f"confidence={result['confidence']:.1%}")
    print(f"disposal_category={result['disposal_category']}")


if __name__ == "__main__":
    main()
