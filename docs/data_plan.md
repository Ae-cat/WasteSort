# Data Plan

## Goal

Determine what waste objects WasteSort should recognize, and prepare a dataset that is ready for training.

## Dataset

**RealWaste** (UCI): 4,752 images, 524×524 RGB, 9 classes. Each teammate downloads it locally to `data/realwaste-main/RealWaste/` (ignored by Git).

## Classes and disposal categories

| Class | Category |
|---|---|
| Cardboard | Recyclable |
| Food Organics | Compost |
| Glass | Recyclable |
| Metal | Recyclable |
| Miscellaneous Trash | Landfill |
| Paper | Recyclable |
| Plastic | Recyclable (proposed: mostly rigid containers in the data) |
| Textile Trash | Landfill |
| Vegetation | Compost |

## Data checks

- No corrupted, duplicate, or near-duplicate images (`ml/check_duplicates.py`).
- Every photo is overhead on gray concrete, so webcam images may perform worse.
- Classes are imbalanced (Plastic 921 vs. Textile Trash 318), so training should use class weights.

## Split and loading

Stratified 70/15/15 split with seed 42, saved in `ml/splits.csv`: 3,323 train, 710 val, 719 test. Load it with `WasteDataset("train" | "val" | "test")` from `ml/dataset.py`. The test set is for final evaluation only.

## Week 3 verification

- Class counts unchanged: 4,752 images across 9 classes.
- No empty folders, corrupt files, exact duplicates, or near-duplicates (`ml/check_duplicates.py`).
- Splits are separate: every image appears in exactly one split (0 overlaps).
- Splits are reproducible: rerunning `ml/make_split.py` (seed 42) produces an identical `splits.csv`.
- The trained checkpoint's class list matches the dataset classes and the disposal mapping in `ml/model.py`.
- No augmentation is currently used; the same preprocessing is applied to train, val, and test.
- All 9 classes are mapped. Plastic → Recyclable is an assumption: most sampled Plastic images are rigid containers.

## Known limitations

- Every image is overhead on gray concrete, so camera images will look different.
- Material labels don't always match disposal rules (food-soiled cardboard, glass shards, scrap metal).
- Miscellaneous Trash is a mixed "everything else" class. It is the weakest class (57% test accuracy) and is often predicted as Food Organics, which would send it to compost.
- Similar-angle photos of the same object would not be caught by the duplicate check.
- Some objects appear twice from different angles (e.g., Miscellaneous Trash `_465`/`_456` and `_393`/`_423`); the duplicate check does not catch these, so similar shots may span train and test and slightly inflate accuracy.