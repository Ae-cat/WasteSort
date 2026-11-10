"""Log held-out RealWaste or separately sourced external images; never combine them."""
import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.models import ResNet18_Weights

from dataset import WasteDataset
from model import DISPOSAL_MAP, load_checkpoint
from prediction_log import prediction_record, sha256, write_predictions


def external_samples(manifest, classes, reference_paths):
    with manifest.open(newline='') as source:
        reader = csv.DictReader(source)
        if not {'image_path', 'true_label', 'source'}.issubset(reader.fieldnames or []):
            raise ValueError('Manifest requires image_path,true_label,source columns')
        rows = list(reader)
    if not rows:
        raise ValueError('External manifest has no images; collect independent photos first')
    # Check ALL splits, not just test: reject exact copies even if renamed.
    reference_hashes = {sha256(p) for p in reference_paths}
    samples, seen = [], set()
    for row in rows:
        path = (manifest.parent / row['image_path']).resolve()
        label = row['true_label'].strip()
        if not row['source'].strip():
            raise ValueError(f'Missing external provenance: {path}')
        if label and label not in classes:
            raise ValueError(f'Unknown true label: {label}')
        digest = sha256(path)
        if digest in reference_hashes:
            raise ValueError(f'External image duplicates RealWaste: {path}')
        if digest in seen:
            raise ValueError(f'Duplicate external image: {path}')
        seen.add(digest)
        samples.append((path, label))
    return samples


def summarize(records, classes):
    labeled = [r for r in records if r['true_label']]
    matrix = [[0 for _ in classes] for _ in classes]
    for row in labeled:
        matrix[classes.index(row['true_label'])][classes.index(row['predicted_label'])] += 1
    per_class = {}
    for i, label in enumerate(classes):
        support = sum(matrix[i])
        per_class[label] = {'support': support, 'correct': matrix[i][i],
                            'recall': matrix[i][i] / support if support else None}
    correct = sum(r['correct'] for r in labeled)
    return {'images': len(records), 'labeled_images': len(labeled), 'correct': correct,
            'accuracy': correct / len(labeled) if labeled else None,
            'disposal_accuracy': sum(DISPOSAL_MAP[r['true_label']] == DISPOSAL_MAP[r['predicted_label']]
                                     for r in labeled) / len(labeled) if labeled else None,
            'per_class': per_class, 'confusion_matrix': matrix}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=['realwaste_held_out_test', 'external_unseen'])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/resnet18_realwaste.pt'))
    parser.add_argument('--splits', type=Path, default=Path('ml/splits.csv'))
    parser.add_argument('--data-root', type=Path, default=Path('data/realwaste-main/RealWaste'))
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing runs are protected')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if (args.kind == 'external_unseen') != bool(args.manifest):
        parser.error('--manifest is required only for external_unseen')
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new run directory')
    model, classes = load_checkpoint(args.checkpoint, args.device)
    if len(classes) != 9 or set(classes) != set(DISPOSAL_MAP):
        raise ValueError('Checkpoint must contain the nine RealWaste classes')
    datasets = [WasteDataset(s, csv_path=args.splits, data_root=args.data_root)
                for s in ['train', 'val', 'test']]
    if any(ds.classes != classes for ds in datasets):
        raise ValueError('Checkpoint class order differs from split labels')
    paths = [p.resolve() for ds in datasets for p, _ in ds.samples]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate paths or split overlap in splits CSV')
    if args.kind == 'external_unseen':
        samples = external_samples(args.manifest, classes, paths)
    else:
        samples = [(p, classes[y]) for p, y in datasets[2].samples]
    transform = ResNet18_Weights.DEFAULT.transforms()
    digest = sha256(args.checkpoint)
    records = []
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = samples[start:start + args.batch_size]
            images = []
            for path, _ in batch:
                with Image.open(path) as image:
                    images.append(transform(image.convert('RGB')))
            probabilities = model(torch.stack(images).to(args.device)).softmax(1)
            confidences, indices = probabilities.max(1)
            for (path, label), confidence, index in zip(batch, confidences.tolist(), indices.tolist()):
                records.append(prediction_record(path, label, classes[index], confidence,
                                                 args.kind, args.checkpoint, digest))
    result = summarize(records, classes)
    result.update(dataset_kind=args.kind, classes=classes, checkpoint_sha256=digest,
                  splits_sha256=sha256(args.splits), device=args.device,
                  preprocessing=str(transform), torch=torch.__version__, torchvision=torchvision.__version__,
                  python=platform.python_version(),
                  git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  manifest_sha256=sha256(args.manifest) if args.manifest else None)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_predictions(args.output_dir / 'predictions.csv', records)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
