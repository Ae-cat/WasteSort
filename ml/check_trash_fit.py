"""Check whether the added trash images fit Miscellaneous Trash: model predictions + a sample grid."""
import argparse
import math
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights

from model import DISPOSAL_MAP, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet18_combined_layer4.pt"))
    parser.add_argument("--folder", type=Path, default=Path("data/combined_trash/Miscellaneous Trash"))
    parser.add_argument("--grid", type=Path, default=Path("reports/trash_fit_sample.png"))
    parser.add_argument("--sample", type=int, default=48)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, classes = load_checkpoint(args.checkpoint, device)
    transform = ResNet18_Weights.DEFAULT.transforms()

    files = sorted(p for p in args.folder.iterdir() if p.name.startswith(("tn_", "g12_")))
    results = []
    with torch.inference_mode():
        for p in files:
            with Image.open(p) as img:
                x = transform(img.convert("RGB")).unsqueeze(0).to(device)
            probs = model(x).softmax(dim=1)[0].cpu()
            conf, idx = probs.max(dim=0)
            results.append((p, classes[idx.item()], conf.item()))

    print(f"{len(results)} added trash images in {args.folder}")
    print("\nPredicted class (current model):")
    for cls, n in Counter(r[1] for r in results).most_common():
        print(f"  {cls:22s} {n:4d}  ({n / len(results):.0%})")
    print("\nPredicted bin:")
    for b, n in Counter(DISPOSAL_MAP[r[1]] for r in results).most_common():
        print(f"  {b:12s} {n:4d}  ({n / len(results):.0%})")
    confident_recyclable = [r for r in results if DISPOSAL_MAP[r[1]] == "Recyclable" and r[2] >= 0.8]
    print(f"\nConfidently predicted Recyclable (>=80%): {len(confident_recyclable)}")

    random.seed(42)
    sample = random.sample(results, min(args.sample, len(results)))
    cols = 8
    rows = math.ceil(len(sample) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (p, cls, conf) in zip(axes.flat, sample):
        with Image.open(p) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{p.name[:16]}\n{cls[:12]} {conf:.0%}", fontsize=7)
    fig.suptitle("Added trash images → current model's prediction")
    fig.tight_layout()
    args.grid.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.grid, dpi=110)
    print(f"\nsaved={args.grid}")


if __name__ == "__main__":
    main()