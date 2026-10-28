"""Measure how a 'Needs review' confidence threshold trades caught errors for flagged correct predictions."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from dataset import WasteDataset
from model import DISPOSAL_MAP, load_checkpoint
from prediction_log import sha256

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


def policy_metrics(rows, threshold):
    """Compost-only abstention, measured without treating review as a correction."""
    flagged = [r for r in rows if r[1] == 'Compost' and r[0] < threshold]
    compost = [r for r in rows if r[1] == 'Compost']
    wrong_compost = sum(p != t for _, p, t in compost)
    right_compost = len(compost) - wrong_compost
    caught = sum(p != t for _, p, t in flagged)
    accepted = [r for r in rows if not (r[1] == 'Compost' and r[0] < threshold)]
    return dict(threshold=threshold, images=len(rows), flagged=len(flagged),
                wrong_compost=wrong_compost, wrong_compost_caught=caught,
                wrong_compost_remaining=wrong_compost-caught,
                correct_compost=right_compost, correct_compost_flagged=len(flagged)-caught,
                catch_rate=caught/wrong_compost if wrong_compost else None,
                correct_compost_flag_rate=(len(flagged)-caught)/right_compost if right_compost else None,
                coverage=len(accepted)/len(rows) if rows else None,
                raw_bin_accuracy=sum(p == t for _, p, t in rows)/len(rows) if rows else None,
                accepted_bin_accuracy=sum(p == t for _, p, t in accepted)/len(accepted) if accepted else None,
                true_compost_total=sum(t == 'Compost' for _, _, t in rows),
                true_compost_flagged=sum(t == 'Compost' for _, _, t in flagged))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet18_realwaste.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/realwaste-main/RealWaste"))
    parser.add_argument("--splits", type=Path, default=Path("ml/splits.csv"))
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument('--threshold', type=float, help='Fixed Compost-only threshold; no sweep')
    parser.add_argument('--output-dir', type=Path, help='New directory for auditable predictions and metrics')
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1')
    if args.output_dir and args.output_dir.exists():
        parser.error('Choose a new output directory')

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, classes = load_checkpoint(args.checkpoint, device)
    dataset = WasteDataset(args.split, ResNet18_Weights.DEFAULT.transforms(), args.splits, args.data_root)
    if dataset.classes != classes:
        raise ValueError('Checkpoint class order does not match the dataset')
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    confidences, predictions, targets = collect(model, loader, device)
    bin_of = [DISPOSAL_MAP[name] for name in classes]
    rows = [(c, bin_of[p], bin_of[t]) for c, p, t in zip(confidences, predictions, targets)]

    print(f"checkpoint={args.checkpoint} split={args.split}")
    thresholds = THRESHOLDS if args.threshold is None else [args.threshold]
    metrics = [policy_metrics(rows, t) for t in thresholds]
    if args.threshold is None:
        sweep("All predictions", rows)
        sweep("Predicted Compost only", [r for r in rows if r[1] == "Compost"])
    else:
        print(json.dumps(metrics[0], indent=2))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = dict(split=args.split, checkpoint=str(args.checkpoint),
                      checkpoint_sha256=sha256(args.checkpoint), splits_sha256=sha256(args.splits),
                      timestamp_utc=datetime.now(timezone.utc).isoformat(), device=device,
                      confidence='maximum class softmax, not summed Compost probability',
                      rule='predicted bin == Compost and confidence < threshold', metrics=metrics)
        (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        with (args.output_dir / 'predictions.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_path','true_label','predicted_label','true_bin','predicted_bin',
                             'confidence','needs_review'])
            for (path, _), c, p, t in zip(dataset.samples, confidences, predictions, targets):
                review = '' if args.threshold is None else bin_of[p] == 'Compost' and c < args.threshold
                writer.writerow([path, classes[t], classes[p], bin_of[t], bin_of[p], c, review])


if __name__ == "__main__":
    main()