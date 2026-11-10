"""Spot-check Household30 real_world photos: a random sample grid, plus near-duplicate pairs side by side."""
import argparse
import random
from pathlib import Path

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/household30/images/images")


def show(pairs_or_files, out, title, cols):
    rows = (len(pairs_or_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (path, label) in zip(axes.flat, pairs_or_files):
        with Image.open(path) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(label, fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved={out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-grid", type=Path, default=Path("reports/h30_sample.png"))
    parser.add_argument("--dupes-grid", type=Path, default=Path("reports/h30_duplicate_pairs.png"))
    args = parser.parse_args()
    random.seed(42)

    folders = sorted(p for p in ROOT.iterdir() if (p / "real_world").is_dir())

    # 1. Random sample: 2 real_world photos from each category
    sample = []
    for f in folders:
        files = sorted((f / "real_world").glob("*.png"))
        for p in random.sample(files, min(2, len(files))):
            sample.append((p, f"{f.name[:22]}\n{p.name}"))
    show(sample, args.sample_grid, "Household30 real_world: 2 random photos per category", cols=10)

    # 2. Near-duplicate pairs (hash distance <= 2) within each category, side by side
    pairs, total_pairs = [], 0
    for f in folders:
        files = sorted((f / "real_world").glob("*.png"))
        hashes = [(p, imagehash.phash(Image.open(p).convert("RGB"))) for p in files]
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                d = hashes[i][1] - hashes[j][1]
                if d <= 2:
                    total_pairs += 1
                    if len(pairs) < 40 and random.random() < 0.15:
                        pairs.append((hashes[i][0], f"{f.name[:18]}\n{hashes[i][0].name} d={d}"))
                        pairs.append((hashes[j][0], f"{f.name[:18]}\n{hashes[j][0].name}"))
    print(f"near-duplicate pairs within categories: {total_pairs}")
    if pairs:
        show(pairs, args.dupes_grid, "Near-duplicate pairs (each pair side by side)", cols=8)


if __name__ == "__main__":
    main()
