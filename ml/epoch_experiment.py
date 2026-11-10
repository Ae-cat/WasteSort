"""Paired 5-vs-10 epoch budgets with one shared initialization and training prefix."""
import argparse
import csv
import json
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from dataset import WasteDataset
from model import build_model
from prediction_log import sha256
from train import run_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data-root', type=Path, default=Path('data/realwaste-main/RealWaste'))
    parser.add_argument('--splits', type=Path, default=Path('ml/splits.csv'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.model_dir.exists():
        parser.error('Use new output/model directories to protect previous runs')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    transform = ResNet18_Weights.DEFAULT.transforms()
    train = WasteDataset('train', transform, args.splits, args.data_root)
    val = WasteDataset('val', transform, args.splits, args.data_root)
    if train.classes != val.classes:
        raise ValueError('Class order mismatch')
    # Match train.py: frozen parameters, but model.train() still updates BatchNorm
    # running statistics. Do not change that behavior in this epochs-only experiment.
    model = build_model(len(train.classes), pretrained=True, freeze_backbone=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = DataLoader(train, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=32, num_workers=0)
    args.output_dir.mkdir(parents=True)
    args.model_dir.mkdir(parents=True)
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
                    device='cpu', epoch_budgets=[5, 10], batch_size=32, learning_rate=.001,
                    optimizer='Adam', class_weights=False, freeze_backbone=True,
                    batchnorm_updates=True, pretrained_weights='ResNet18_Weights.IMAGENET1K_V1',
                    train_images=len(train), validation_images=len(val), test_used=False,
                    selection='Highest validation accuracy within each budget; ties keep earliest epoch',
                    classes=train.classes, preprocessing=str(transform),
                    torch=torch.__version__, torchvision=torchvision.__version__,
                    splits_sha256=sha256(args.splits),
                    git_base_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                    source_sha256={p: sha256(p) for p in ['ml/epoch_experiment.py', 'ml/train.py',
                                                         'ml/model.py', 'ml/dataset.py']})
    (args.output_dir / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    best_accuracy, best_epoch = -1., None
    fields = ['epoch','train_loss','train_accuracy','val_loss','val_accuracy','elapsed_seconds']
    with (args.output_dir / 'history.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for epoch in range(1, 11):
            start = time.monotonic()
            train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, 'cpu', optimizer)
            val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, 'cpu')
            row = dict(zip(fields, [epoch, train_loss, train_accuracy, val_loss, val_accuracy,
                                    time.monotonic() - start]))
            writer.writerow(row)
            stream.flush()
            print(json.dumps(row), flush=True)
            if val_accuracy > best_accuracy:
                best_accuracy, best_epoch = val_accuracy, epoch
                checkpoint = dict(model_state=model.state_dict(), classes=train.classes,
                                  val_accuracy=val_accuracy, epoch=epoch, seed=args.seed)
                torch.save(checkpoint, args.model_dir / 'best_10epochs.pt')
                if epoch <= 5:
                    torch.save(checkpoint, args.model_dir / 'best_5epochs.pt')
            if epoch == 5:
                metadata['budget_5'] = dict(best_epoch=best_epoch, val_accuracy=best_accuracy)
    metadata['budget_10'] = dict(best_epoch=best_epoch, val_accuracy=best_accuracy)
    metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['checkpoint_sha256'] = {p.name: sha256(p) for p in sorted(args.model_dir.glob('*.pt'))}
    (args.output_dir / 'config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
