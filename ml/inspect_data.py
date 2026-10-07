from pathlib import Path
import random
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path("data/realwaste-main/RealWaste")
random.seed(42)

class_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
for d in class_dirs:
    print(d.name)

total = 0
for d in class_dirs:
    files = list(d.glob("*.jpg"))
    print(f"{d.name}: {len(files)}")
    total += len(files)
print("Total:", total)

all_files = []
for d in class_dirs:
    all_files.extend(d.glob("*.jpg"))

sample = random.sample(all_files, 50)
stats = Counter()
for f in sample:
    img = Image.open(f)
    stats[(img.size, img.mode)] += 1
print(stats)

fig, axes = plt.subplots(nrows=len(class_dirs), ncols=5, figsize=(10, 18))
for row, d in enumerate(class_dirs):
    picks = random.sample(list(d.glob("*.jpg")), 5)
    for col, f in enumerate(picks):
        ax = axes[row][col]
        ax.imshow(Image.open(f))
        ax.axis("off")
    axes[row][0].set_title(d.name, loc="left", fontsize=9)

plt.tight_layout()
plt.savefig("ml/sample_grid.png")
plt.show()