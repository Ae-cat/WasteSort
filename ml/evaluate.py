"""Evaluate a trained RealWaste checkpoint on the held-out test split."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from dataset import WasteDataset
from model import load_checkpoint


def calculate_metrics(targets, predictions, classes):
    matrix = confusion_matrix(targets, predictions, labels=range(len(classes)))
    total = matrix.sum()
    overall_accuracy = matrix.trace() / total if total else 0.0
    class_totals = matrix.sum(axis=1)
    per_class_accuracy = {
        class_name: matrix[index, index] / class_totals[index]
        if class_totals[index] else 0.0
        for index, class_name in enumerate(classes)
    }
    return overall_accuracy, per_class_accuracy, matrix


def evaluate(model, loader, device):
    targets = []
    predictions = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return targets, predictions


def save_confusion_matrix(matrix, classes, output_path):
    figure, axis = plt.subplots(figsize=(11, 9))
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False)
    axis.set_title("RealWaste Test-Set Confusion Matrix")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/resnet18_realwaste.pt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/realwaste-main/RealWaste"))
    parser.add_argument("--splits", type=Path, default=Path("ml/splits.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--confusion-matrix",
        type=Path,
        default=Path("reports/baseline_confusion_matrix.png"),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, classes = load_checkpoint(args.checkpoint, device)
    test_set = WasteDataset(
        "test",
        ResNet18_Weights.DEFAULT.transforms(),
        args.splits,
        args.data_root,
    )
    if test_set.classes != classes:
        raise ValueError("Checkpoint class order does not match the test dataset")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

    targets, predictions = evaluate(model, test_loader, device)
    overall_accuracy, per_class_accuracy, matrix = calculate_metrics(
        targets, predictions, classes
    )

    print(f"device={device} test_images={len(test_set)}")
    print(f"overall_accuracy={overall_accuracy:.3%}")
    print("per_class_accuracy:")
    for class_name, accuracy in per_class_accuracy.items():
        print(f"  {class_name:22s} {accuracy:.3%}")
    print("confusion_matrix:")
    print(matrix)

    save_confusion_matrix(matrix, classes, args.confusion_matrix)
    print(f"confusion_matrix_image={args.confusion_matrix}")


if __name__ == "__main__":
    main()
