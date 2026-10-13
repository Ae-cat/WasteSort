from pathlib import Path
import hashlib
from collections import defaultdict
from PIL import Image 
import imagehash 

ROOT = Path("data/realwaste-main/RealWaste")

images = []
for class_dir in sorted(ROOT.iterdir()):
    if not class_dir.is_dir():
        continue
    for path in class_dir.glob("*jpg"):
        images.append((path, class_dir.name))

print(len(images))
print(images[0])

unusable = []
exact_groups = defaultdict(list)
phash_groups = defaultdict(list)

for path, cls in images:
    try:
        with Image.open(path) as img:
            img.load()
            p = str(imagehash.phash(img))
    except Exception as e:
        unusable.append((path, cls, str(e)))
        continue

    exact = hashlib.md5(path.read_bytes()).hexdigest()

    exact_groups[exact].append((path, cls))
    phash_groups[p].append((path, cls))

print("Unusable:", len(unusable))

def find_dupes(groups, label):
    dupes = [members for members in groups.values() if len(members) >= 2]
    cross_class = [g for g in dupes if len({cls for _, cls in g}) > 1]

    print(f"\n{label}: {len(dupes)} groups, {sum(len(g) for g in dupes)} images")
    print(f" groups spanning more than one class: {len(cross_class)}")
    for g in dupes[:5]:
        print(" ", [f"{cls}/{path.name}" for path, cls in g])
    return dupes

exact_dupes = find_dupes(exact_groups, "Exact duplicates")
near_dupes = find_dupes(phash_groups, "Near-duplicates")