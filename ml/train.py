"""Train a RealWaste ResNet18 classifier and save the best checkpoint."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from dataset import WasteDataset
from model import build_model


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_examples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/realwaste-main/RealWaste"))
    parser.add_argument("--splits", type=Path, default=Path("ml/splits.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/resnet18_realwaste.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--class-weights", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    transform = ResNet18_Weights.DEFAULT.transforms()
    train_set = WasteDataset("train", transform, args.splits, args.data_root)
    val_set = WasteDataset("val", transform, args.splits, args.data_root)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=0)

    model = build_model(len(train_set.classes), not args.no_pretrained, args.freeze_backbone).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    if args.class_weights:
        labels = torch.tensor([label for _, label in train_set.samples])
        counts = torch.bincount(labels, minlength=len(train_set.classes)).float()
        weights = len(train_set) / (len(train_set.classes) * counts)
        print("class weights:", {c: round(w, 4) for c, w in zip(train_set.classes, weights.tolist())})
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    best_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"device={device} classes={len(train_set.classes)} train={len(train_set)} val={len(val_set)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state": model.state_dict(), "classes": train_set.classes,
                        "val_accuracy": val_accuracy, "epoch": epoch}, args.output)

    print(f"saved={args.output} best_val_accuracy={best_accuracy:.3f}")


if __name__ == "__main__":
    main()
