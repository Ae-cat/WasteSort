"""Measure how a 'Needs review' confidence threshold trades caught errors for flagged correct predictions."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from dataset import WasteDataset
from model import DISPOSAL_MAP, load_checkpoint

THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def collect(model, loader, device):
    confidences, predictions, targets = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            probs = model(images.to(device)).softmax(dim=1).cpu()
            conf, pred = probs.max(dim=1)
            confidences += conf.tolist()
            predictions += pred.tolist()
            targets += labels.tolist()
    return confidences, predictions, targets


def pct(part, whole):
    return f"{part / whole:.0%}" if whole else "n/a"


def sweep(title, rows):
    """rows: list of (confidence, predicted_bin, true_bin)."""
    wrong = [r for r in rows if r[1] != r[2]]
    right = [r for r in rows if r[1] == r[2]]
    print(f"\n{title}: {len(rows)} predictions, {len(wrong)} in the wrong bin")
    print(f"  {'threshold':<10}{'wrong-bin caught':<22}{'right-bin flagged':<22}")
    for t in THRESHOLDS:
        caught = sum(conf < t for conf, _, _ in wrong)
        flagged = sum(conf < t for conf, _, _ in right)
        print(f"  {t:<10.0%}{f'{caught}/{len(wrong)} ({pct(caught, len(wrong))})':<22}"
              f"{f'{flagged}/{len(right)} ({pct(flagged, len(right))})':<22}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet18_realwaste.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/realwaste-main/RealWaste"))
    parser.add_argument("--splits", type=Path, default=Path("ml/splits.csv"))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, classes = load_checkpoint(args.checkpoint, device)
    dataset = WasteDataset(args.split, ResNet18_Weights.DEFAULT.transforms(), args.splits, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    confidences, predictions, targets = collect(model, loader, device)
    bin_of = [DISPOSAL_MAP[name] for name in classes]
    rows = [(c, bin_of[p], bin_of[t]) for c, p, t in zip(confidences, predictions, targets)]

    print(f"checkpoint={args.checkpoint} split={args.split}")
    sweep("All predictions", rows)
    sweep("Predicted Compost only", [r for r in rows if r[1] == "Compost"])


if __name__ == "__main__":
    main()