"""Create a stratified 70/15/15 train/val/test split of RealWaste and save it to ml/splits.csv."""
from pathlib import Path
from collections import defaultdict
import random
import csv

ROOT = Path("data/realwaste-main/RealWaste")
OUT = Path("ml/splits.csv")
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  

random.seed(SEED)

by_class = defaultdict(list)
for class_dir in sorted(ROOT.iterdir()):
    if not class_dir.is_dir():
        continue
    for path in sorted(class_dir.glob("*.jpg")):
        by_class[class_dir.name].append(path)

rows = []
counts = {}
for cls, paths in by_class.items():
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    splits = {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:],
    }
    counts[cls] = {name: len(ps) for name, ps in splits.items()}

    for split_name, ps in splits.items():
        for p in ps:
            rows.append([p.as_posix(), cls, split_name])

with OUT.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label", "split"])
    writer.writerows(rows)

print(f"{'class':22s}{'train':>7s}{'val':>6s}{'test':>6s}{'total':>7s}")
totals = {"train": 0, "val": 0, "test": 0}
for cls, c in counts.items():
    print(f"{cls:22s}{c['train']:7d}{c['val']:6d}{c['test']:6d}{sum(c.values()):7d}")
    for k in totals:
        totals[k] += c[k]
print(f"{'TOTAL':22s}{totals['train']:7d}{totals['val']:6d}{totals['test']:6d}{sum(totals.values()):7d}")
print(f"\nWrote {len(rows)} rows to {OUT}")