"""Build the combined training set: RealWaste + filtered TrashNet + filtered Garbage Classification (12 classes).

- Validation and test rows from ml/splits.csv are kept exactly as they are; new images go to train only.
- New images are mapped to our 9 classes; unmatched folders (battery, trash) are skipped.
- Near-duplicates are removed: any new image close to ANY RealWaste image (all splits, test included),
  and duplicates among the new images (all copies dropped if they disagree on the label).
- Clothes and shoes are capped so Textile Trash does not swamp the other classes.
- Everything is copied into data/combined/<Class>/ so train.py / evaluate.py work with
  --data-root data/combined (dataset.py rebuilds paths as data_root / label / filename).

Deterministic: rerunning on the same inputs gives the same ml/splits_combined.csv.
"""
import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MIN_SIDE = 64          # skip thumbnails / broken tiny images
DIST_REALWASTE = 4     # perceptual-hash distance counted as "same photo" vs RealWaste
DIST_SAME_SOURCE = 2   # same, between two new images from the same dataset
DIST_CROSS_SOURCE = 4  # looser between datasets, since a re-uploaded copy may be resized or recompressed

# (name, root folder, filename prefix, {source folder: (our class, cap or None)})
SOURCES = [
    ("TrashNet", Path("data/trashnet/dataset-resized"), "tn_", {
        "cardboard": ("Cardboard", None),
        "glass": ("Glass", None),
        "metal": ("Metal", None),
        "paper": ("Paper", None),
        "plastic": ("Plastic", None),
        # "trash" skipped: catch-all class, does not clearly match Miscellaneous Trash
    }),
    ("Garbage12", Path("data/garbage12/garbage_classification"), "g12_", {
        "biological": ("Food Organics", None),
        "cardboard": ("Cardboard", None),
        "paper": ("Paper", None),
        "plastic": ("Plastic", None),
        "metal": ("Metal", None),
        "white-glass": ("Glass", None),
        "green-glass": ("Glass", None),
        "brown-glass": ("Glass", None),
        "clothes": ("Textile Trash", 800),
        "shoes": ("Textile Trash", 800),
        # "battery" and "trash" skipped: no matching class
    }),
]


def phash_bits(path):
    """Return the 64-bit perceptual hash as a uint64, or None if the image is unusable."""
    try:
        with Image.open(path) as img:
            img.load()
            if min(img.size) < MIN_SIDE:
                return None
            bits = imagehash.phash(img.convert("RGB")).hash.flatten()
    except Exception:
        return None
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(one, many):
    """Hamming distance between one uint64 hash and an array of uint64 hashes."""
    if len(many) == 0:
        return np.array([], dtype=int)
    x = np.bitwise_xor(many, one)
    return _POPCOUNT[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("ml/splits.csv"))
    parser.add_argument("--realwaste-root", type=Path, default=Path("data/realwaste-main/RealWaste"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/combined"))
    parser.add_argument("--out-csv", type=Path, default=Path("ml/splits_combined.csv"))
    parser.add_argument("--dropped-csv", type=Path, default=Path("reports/combined_dropped.csv"))
    args = parser.parse_args()
    missing_sources = [str(root) for _, root, _, _ in SOURCES if not root.is_dir()]
    if missing_sources:
        raise SystemExit('Missing required dataset directories: ' + ', '.join(missing_sources))
    rng = random.Random(SEED)
    dropped = []  # (source, file, reason)

    # 1. RealWaste: every row, every split (test included) is used for duplicate checks
    with open(args.splits, newline="") as f:
        rw_rows = list(csv.DictReader(f))
    print(f"Hashing {len(rw_rows)} RealWaste images...")
    rw_hashes = []
    for row in rw_rows:
        src = args.realwaste_root / row["label"] / Path(row["path"]).name
        h = phash_bits(src)
        if h is None:
            raise SystemExit(f"Could not read RealWaste image {src} - check --realwaste-root")
        rw_hashes.append(h)
    rw_hashes = np.array(rw_hashes, dtype=np.uint64)

    rw_classes = {r["label"] for r in rw_rows}
    for _, _, _, mapping in SOURCES:
        unknown = {label for label, _ in mapping.values()} - rw_classes
        if unknown:  # a new class would change the label numbering and break the model
            raise SystemExit(f"Mapping uses classes not in RealWaste: {unknown}")

    # 2. Collect candidate new images
    candidates = []  # dict(source, src_path, label, folder, cap, hash)
    for name, root, prefix, mapping in SOURCES:
        if not root.is_dir():
            print(f"WARNING: {root} not found, skipping {name}")
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            if folder.name not in mapping:
                print(f"  {name}/{folder.name}: skipped (no matching class)")
                continue
            label, cap = mapping[folder.name]
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            print(f"  {name}/{folder.name} -> {label}: hashing {len(files)} images...")
            for p in files:
                h = phash_bits(p)
                if h is None:
                    dropped.append((name, str(p), "unreadable or too small"))
                    continue
                candidates.append(dict(source=name, prefix=prefix, src=p, label=label,
                                       folder=f"{name}/{folder.name}", cap=cap, hash=h))

    # 3. Drop anything that matches a RealWaste image (any split)
    kept = []
    for c in candidates:
        d = hamming(c["hash"], rw_hashes)
        if d.size and d.min() <= DIST_REALWASTE:
            dropped.append((c["source"], str(c["src"]), "near-duplicate of a RealWaste image"))
        else:
            kept.append(c)
    candidates = kept

    # 4. Duplicates among the new images (union-find over close pairs)
    hashes = np.array([c["hash"] for c in candidates], dtype=np.uint64)
    source_ids = np.array([c["source"] for c in candidates])
    parent = list(range(len(candidates)))
    for i in range(len(candidates)):
        limit = np.where(source_ids[i + 1:] == source_ids[i], DIST_SAME_SOURCE, DIST_CROSS_SOURCE)
        close = np.nonzero(hamming(hashes[i], hashes[i + 1:]) <= limit)[0] + i + 1
        for j in close:
            parent[find(parent, int(j))] = find(parent, i)
    groups = defaultdict(list)
    for i in range(len(candidates)):
        groups[find(parent, i)].append(i)
    keep_idx = set()
    for members in groups.values():
        labels = {candidates[i]["label"] for i in members}
        if len(members) == 1:
            keep_idx.add(members[0])
        elif len(labels) > 1:
            for i in members:
                dropped.append((candidates[i]["source"], str(candidates[i]["src"]),
                                "duplicate with a different label"))
        else:
            keep_idx.add(members[0])  # TrashNet comes first, so its copy is kept
            for i in members[1:]:
                dropped.append((candidates[i]["source"], str(candidates[i]["src"]),
                                "duplicate of another new image"))
    candidates = [c for i, c in enumerate(candidates) if i in keep_idx]

    # 5. Apply caps per source folder (seeded random sample)
    by_folder = defaultdict(list)
    for c in candidates:
        by_folder[c["folder"]].append(c)
    final = []
    for folder in sorted(by_folder):
        items = by_folder[folder]
        cap = items[0]["cap"]
        if cap is not None and len(items) > cap:
            chosen = set(id(c) for c in rng.sample(items, cap))
            for c in items:
                if id(c) not in chosen:
                    dropped.append((c["source"], str(c["src"]), f"over cap of {cap}"))
            items = [c for c in items if id(c) in chosen]
        final.extend(items)

    # 6. Copy into data/combined/<Class>/ and write the combined split file
    out_rows = []
    for row in rw_rows:
        name = Path(row["path"]).name
        dest = args.out_dir / row["label"] / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(args.realwaste_root / row["label"] / name, dest)
        out_rows.append([dest.as_posix(), row["label"], row["split"]])
    for c in final:
        dest = args.out_dir / c["label"] / f"{c['prefix']}{c['src'].name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(c["src"], dest)
        out_rows.append([dest.as_posix(), c["label"], "train"])

    with args.out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split"])
        writer.writerows(out_rows)
    args.dropped_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.dropped_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "file", "reason"])
        writer.writerows(dropped)

    # 7. Summary
    rw_train = Counter(r["label"] for r in rw_rows if r["split"] == "train")
    added = defaultdict(Counter)
    for c in final:
        added[c["source"]][c["label"]] += 1
    classes = sorted(rw_classes)
    names = [s[0] for s in SOURCES]
    print(f"\n{'class':22s}{'RealWaste':>10s}" + "".join(f"{n:>11s}" for n in names) + f"{'train total':>13s}")
    totals = Counter()
    for cls in classes:
        row_total = rw_train[cls] + sum(added[n][cls] for n in names)
        totals["rw"] += rw_train[cls]
        for n in names:
            totals[n] += added[n][cls]
        totals["all"] += row_total
        print(f"{cls:22s}{rw_train[cls]:>10d}" + "".join(f"{added[n][cls]:>11d}" for n in names) + f"{row_total:>13d}")
    print(f"{'TOTAL':22s}{totals['rw']:>10d}" + "".join(f"{totals[n]:>11d}" for n in names) + f"{totals['all']:>13d}")

    split_counts = Counter(r[2] for r in out_rows)
    orig_counts = Counter(r["split"] for r in rw_rows)
    print(f"\nval:  {split_counts['val']} (original {orig_counts['val']})")
    print(f"test: {split_counts['test']} (original {orig_counts['test']})")
    assert split_counts["val"] == orig_counts["val"] and split_counts["test"] == orig_counts["test"]

    print("\nDropped:")
    for reason, n in Counter(r[2] for r in dropped).most_common():
        print(f"  {n:5d}  {reason}")
    print(f"\nWrote {len(out_rows)} rows to {args.out_csv}")
    print(f"Dropped-file log: {args.dropped_csv}")
    print(f"Images copied to {args.out_dir} (use --data-root {args.out_dir} with train.py / evaluate.py)")


if __name__ == "__main__":
    main()
