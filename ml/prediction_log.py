"""Reusable, auditable prediction records (blank correctness means unlabeled)."""
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ['image_path', 'true_label', 'predicted_label', 'confidence', 'correct',
          'timestamp_utc', 'dataset_kind', 'model', 'checkpoint', 'checkpoint_sha256']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_record(path, true_label, predicted_label, confidence, kind, checkpoint, digest):
    return dict(zip(FIELDS, [str(path), true_label or '', predicted_label, confidence,
                '' if not true_label else true_label == predicted_label,
                datetime.now(timezone.utc).isoformat(), kind, 'resnet18_realwaste_9class',
                str(checkpoint), digest]))


def write_predictions(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
