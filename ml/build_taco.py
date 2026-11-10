"""Download TACO (Trash Annotations in Context) and crop its labeled litter into single-object images.

TACO photos show real litter in real places (streets, grass, beaches), with one or more objects
per photo. This script:
  1. clones the TACO repo into data/taco_repo (annotations only, a few MB) if it is not there yet,
  2. downloads the photos listed in the annotations (resumable; failed links are skipped, not fatal),
  3. crops every labeled object whose box is at least --min-side pixels on its shorter side,
     with a little padding, and saves it under data/taco_crops/<our class>/.

Only TACO categories that clearly match one of our 9 classes are used (see CATEGORY_MAP).
Afterwards, add the crops to training with:
    python ml/build_combined.py --sources taco ...
which applies the usual duplicate checks and keeps validation/test unchanged.

TACO: Proenca & Simoes, "TACO: Trash Annotations in Context for Litter Detection" (2020),
github.com/pedropro/TACO. Code MIT; images CC BY 4.0.
"""
import argparse
import json
import subprocess
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

REPO_URL = "https://github.com/pedropro/TACO.git"

# TACO category name -> our class. Anything not listed is skipped.
CATEGORY_MAP = {
    # Miscellaneous Trash (landfill): film, wrappers, bags, foam, straws, utensils, tissues
    "Plastic film": "Miscellaneous Trash",
    "Other plastic wrapper": "Miscellaneous Trash",
    "Crisp packet": "Miscellaneous Trash",
    "Garbage bag": "Miscellaneous Trash",
    "Single-use carrier bag": "Miscellaneous Trash",
    "Polypropylene bag": "Miscellaneous Trash",
    "Six pack rings": "Miscellaneous Trash",
    "Plastic straw": "Miscellaneous Trash",
    "Styrofoam piece": "Miscellaneous Trash",
    "Foam cup": "Miscellaneous Trash",
    "Foam food container": "Miscellaneous Trash",
    "Plastic utensils": "Miscellaneous Trash",
    "Plastic glooves": "Miscellaneous Trash",
    "Tissues": "Miscellaneous Trash",
    "Rope & strings": "Miscellaneous Trash",
    "Cigarette": "Miscellaneous Trash",
    # Plastic (rigid)
    "Clear plastic bottle": "Plastic",
    "Other plastic bottle": "Plastic",
    "Disposable food container": "Plastic",
    "Spread tub": "Plastic",
    "Tupperware": "Plastic",
    # Metal
    "Drink can": "Metal",
    "Food Can": "Metal",
    "Aerosol": "Metal",
    "Scrap metal": "Metal",
    "Aluminium foil": "Metal",
    # Glass
    "Glass bottle": "Glass",
    "Glass jar": "Glass",
    "Glass cup": "Glass",
    "Broken glass": "Glass",
    # Paper
    "Normal paper": "Paper",
    "Magazine paper": "Paper",
    "Wrapping paper": "Paper",
    "Paper bag": "Paper",
    # Cardboard
    "Corrugated carton": "Cardboard",
    "Meal carton": "Cardboard",
    "Egg carton": "Cardboard",
    "Pizza box": "Cardboard",
    "Toilet tube": "Cardboard",
    # Food Organics / Textile Trash
    "Food waste": "Food Organics",
    "Shoe": "Textile Trash",
    # Skipped on purpose (ambiguous or no matching class): Unlabeled litter, Other plastic,
    # bottle caps, pop tabs, lids, paper cups, plastic cups, drink cartons, other cartons,
    # other plastic containers, paper straws, squeezable tubes, blister packs, batteries.
}


ERRORS = Counter()  # failure reasons, for the summary
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                         "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"}


def download_one(job):
    url, path = job
    if path.exists():
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Some image hosts reject Python's default user agent, so send a normal browser one
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
    except Exception as e:
        ERRORS[f"{urllib.parse.urlparse(url).netloc}: {type(e).__name__} {getattr(e, 'code', '')} {getattr(e, 'reason', '')}".strip()] += 1
        return "failed"
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return "downloaded"


def load_image(path, width, height):
    """Open a TACO photo in the orientation its annotations use."""
    img = Image.open(path)
    rotated = ImageOps.exif_transpose(img)
    if rotated.size == (width, height):
        return rotated
    if img.size == (width, height):
        return img
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("data/taco_repo"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/taco_crops"))
    parser.add_argument("--unofficial", action="store_true",
                        help="also use annotations_unofficial.json (unreviewed by the TACO authors)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--min-side", type=int, default=100, help="skip boxes smaller than this (pixels)")
    parser.add_argument("--pad", type=float, default=0.10, help="padding around each box, as a fraction")
    parser.add_argument("--max-side", type=int, default=512, help="shrink crops so the long side is at most this")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not (args.repo / "data" / "annotations.json").exists():
        print(f"Cloning TACO annotations into {args.repo} ...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(args.repo)], check=True)

    ann_files = [args.repo / "data" / "annotations.json"]
    if args.unofficial:
        ann_files.append(args.repo / "data" / "annotations_unofficial.json")

    stats = Counter()
    per_class = Counter()
    per_category = Counter()
    for ann_file in ann_files:
        tag = "u" if "unofficial" in ann_file.name else "o"
        coco = json.loads(ann_file.read_text())
        images = {im["id"]: im for im in coco["images"]}
        categories = {c["id"]: c["name"] for c in coco["categories"]}
        image_dir = ann_file.parent

        # 1. Download the photos that contain at least one usable object
        def usable(a):
            return categories[a["category_id"]] in CATEGORY_MAP and min(a["bbox"][2], a["bbox"][3]) >= args.min_side

        needed = {a["image_id"] for a in coco["annotations"] if usable(a)}
        print(f"{ann_file.name}: {sum(usable(a) for a in coco['annotations'])} usable objects "
              f"in {len(needed)} photos")
        if not args.skip_download:
            jobs = [(images[i]["flickr_url"], image_dir / images[i]["file_name"]) for i in sorted(needed)]
            print(f"  fetching {len(jobs)} original photos (resumable; this can take a while)...")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for n, result in enumerate(pool.map(download_one, jobs), 1):
                    stats[f"photo {result}"] += 1
                    if n % 100 == 0 or n == len(jobs):
                        print(f"  {n}/{len(jobs)}")

        # 2. Crop each usable object
        by_image = {}
        for a in coco["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        for image_id in sorted(needed):
            meta = images[image_id]
            path = image_dir / meta["file_name"]
            if not path.exists():
                stats["objects skipped: photo not downloaded"] += sum(usable(a) for a in by_image[image_id])
                continue
            try:
                img = load_image(path, meta["width"], meta["height"])
            except Exception:
                img = None
            if img is None:
                stats["photo unreadable or size mismatch"] += 1
                continue
            img = img.convert("RGB")
            for a in by_image[image_id]:
                if not usable(a):
                    continue
                name = categories[a["category_id"]]
                x, y, w, h = a["bbox"]
                label = CATEGORY_MAP[name]
                px, py = w * args.pad, h * args.pad
                box = (max(0, int(x - px)), max(0, int(y - py)),
                       min(img.width, int(x + w + px)), min(img.height, int(y + h + py)))
                crop = img.crop(box)
                crop.thumbnail((args.max_side, args.max_side))
                out = args.out_dir / label / f"taco{tag}_{image_id}_{a['id']}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                crop.save(out, quality=92)
                per_class[label] += 1
                per_category[name] += 1
                stats["objects cropped"] += 1

    print("\nCrops per class:")
    for label, n in sorted(per_class.items()):
        print(f"  {label:22s} {n:5d}")
    print(f"  {'TOTAL':22s} {sum(per_class.values()):5d}")
    print("\nCrops per TACO category:")
    for name, n in per_category.most_common():
        print(f"  {name:26s} {n:5d}  -> {CATEGORY_MAP[name]}")
    print("\nStats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    if ERRORS:
        print("\nDownload failures by reason:")
        for reason, n in ERRORS.most_common():
            print(f"  {n:5d}  {reason}")
    print(f"\nCrops saved under {args.out_dir}/<class>/")
    print("Next: python ml/build_combined.py --sources taco --out-dir data/combined_taco "
          "--out-csv ml/splits_combined_taco.csv --dropped-csv reports/taco_dropped.csv")


if __name__ == "__main__":
    main()
