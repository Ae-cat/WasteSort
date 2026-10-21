import argparse
from pathlib import Path
import random

import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

ROOT = Path("data/realwaste-main/RealWaste")


def prepare_image(path, preprocess):
    with Image.open(path) as image:
        x = preprocess(image.convert("RGB")).unsqueeze(0)
    if x.shape != (1, 3, 224, 224) or x.dtype != torch.float32:
        raise ValueError("Expected a float32 image tensor with shape (1, 3, 224, 224)")
    if not torch.isfinite(x).all():
        raise ValueError("Image tensor contains non-finite values")
    return x


def predict(x, model, labels):
    with torch.inference_mode():
        probs = model(x).softmax(dim=1)[0]
    conf, idx = probs.max(dim=0)
    return labels[idx.item()], conf.item()


def main():
    parser = argparse.ArgumentParser(description="Predict using pretrained ResNet18.")
    parser.add_argument("--image", type=Path, help="Path to an image file")
    args = parser.parse_args()

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    labels = weights.meta["categories"]

    try:
        # Invalid files fail before loading model weights.
        x = prepare_image(args.image, preprocess) if args.image else None
        model = resnet18(weights=weights)
        model.eval()

        if args.image:
            print(f"Input: shape={tuple(x.shape)}, dtype={x.dtype}")
            label, confidence = predict(x, model, labels)
            print(f"{args.image} -> {label} ({confidence:.0%})")
        else:
            random.seed(0)
            for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
                f = random.choice(list(d.glob("*.jpg")))
                x = prepare_image(f, preprocess)
                label, confidence = predict(x, model, labels)
                print(f"{d.name:20s} -> {label} ({confidence:.0%})")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
