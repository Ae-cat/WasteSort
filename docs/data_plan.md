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