"""Show test images of one true class that the model predicted as another class."""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.models import ResNet18_Weights

from dataset import WasteDataset
from model import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet18_realwaste.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/realwaste-main/RealWaste"))
    parser.add_argument("--splits", type=Path, default=Path("ml/splits.csv"))
    parser.add_argument("--true-class", default="Miscellaneous Trash")
    parser.add_argument("--pred-class", default="Food Organics")
    parser.add_argument("--output", type=Path, default=Path("reports/misc_as_food_organics.png"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, classes = load_checkpoint(args.checkpoint, device)
    true_index = classes.index(args.true_class)
    pred_index = classes.index(args.pred_class)

    # Same test split twice: one preprocessed for the model, one raw for display
    model_view = WasteDataset("test", ResNet18_Weights.DEFAULT.transforms(), args.splits, args.data_root)
    raw_view = WasteDataset("test", None, args.splits, args.data_root)

    errors = []
    with torch.inference_mode():
        for i in range(len(model_view)):
            image, label = model_view[i]
            if label != true_index:
                continue
            probs = model(image.unsqueeze(0).to(device)).softmax(dim=1)[0].cpu()
            if probs.argmax().item() == pred_index:
                name = Path(model_view.samples[i][0]).name
                errors.append((i, name, probs[pred_index].item(), probs[true_index].item()))

    print(f"{args.true_class} predicted as {args.pred_class}: {len(errors)} images")
    for _, name, pred_conf, true_conf in errors:
        print(f"  {name:30s} {args.pred_class} {pred_conf:.0%}   {args.true_class} {true_conf:.0%}")
    if not errors:
        return

    cols = 4
    rows = math.ceil(len(errors) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (i, name, pred_conf, true_conf) in zip(axes.flat, errors):
        image, _ = raw_view[i]
        ax.imshow(image)
        ax.set_title(f"{name}\npred {pred_conf:.0%} / true {true_conf:.0%}", fontsize=8)
    fig.suptitle(f"True: {args.true_class}  →  Predicted: {args.pred_class}")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()