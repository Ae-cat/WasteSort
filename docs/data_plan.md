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

## Week 5: Combined training set

Script: `ml/build_combined.py` → `ml/splits_combined.csv` (dropped-file log: `reports/combined_dropped.csv`)

Validation (710) and test (719) are unchanged from `ml/splits.csv`. All new images were added to the train split only.

### Sources

- **RealWaste** (base): unchanged.
- **TrashNet** (Thung & Yang, Stanford, 2016): 2,527 images, one object per photo on a plain light background.
- **Garbage Classification (12 classes)** (mostafaabla, Kaggle): 15,150 mostly web-scraped images. Licensed under the Open Database License (ODbL). Attribution: "Contains images from Garbage Classification (12 classes) by mostafaabla on Kaggle, available under the Open Database License (ODbL)."

### Class mapping

| Source folder | Our class |
| --- | --- |
| TrashNet cardboard / glass / metal / paper / plastic | Cardboard / Glass / Metal / Paper / Plastic |
| Garbage12 biological | Food Organics |
| Garbage12 cardboard / paper / plastic / metal | Cardboard / Paper / Plastic / Metal |
| Garbage12 white-glass, green-glass, brown-glass | Glass |
| Garbage12 clothes, shoes (capped at 800 each) | Textile Trash |
| TrashNet trash, Garbage12 trash, Garbage12 battery | Excluded (no clear match) |

### Training images per class

| Class | RealWaste | TrashNet | Garbage12 | Train total |
| --- | ---: | ---: | ---: | ---: |
| Cardboard | 322 | 401 | 484 | 1,207 |
| Food Organics | 287 | 0 | 979 | 1,266 |
| Glass | 294 | 497 | 1,495 | 2,286 |
| Metal | 553 | 404 | 354 | 1,311 |
| Miscellaneous Trash | 346 | 0 | 0 | 346 |
| Paper | 350 | 575 | 456 | 1,381 |
| Plastic | 644 | 477 | 381 | 1,502 |
| Textile Trash | 222 | 0 | 1,600 | 1,822 |
| Vegetation | 305 | 0 | 0 | 305 |
| **Total** | **3,323** | **2,354** | **5,749** | **11,426** |

### Filtering

| Reason | Images dropped |
| --- | ---: |
| Over the 800 cap (clothes, shoes) | 5,681 |
| Duplicate of another new image | 2,446 |
| Duplicate with a different label | 24 |
| Near-duplicate of a RealWaste image (any split) | 9 |

- Garbage12 contains most of TrashNet (e.g., all 594 TrashNet paper images reappear in it). One copy of each was kept.
- Duplicates were found with perceptual hashing: distance ≤ 4 against RealWaste and between datasets, ≤ 2 within a dataset. Copies that disagreed on the label were all dropped.

### Known limitations

- The target was 20,000+ clean images. 11,426 is what remained after keeping only clearly matching classes and removing duplicates.
- Class imbalance is larger than before: Glass has 2,286 training images, Vegetation 305 and Miscellaneous Trash 346 (no new sources). Watch Miscellaneous Trash and Landfill bin accuracy after retraining.
- Garbage12 images are web-scraped; many Textile Trash images are clean product photos rather than worn items. ODbL covers the dataset, not the copyright of individual images, so do not reuse individual images in posters or slides.
- To rebuild `data/combined/` on another machine: unzip TrashNet to `data/trashnet/` and the Kaggle archive to `data/garbage12/`, then run `python ml/build_combined.py`. It produces the identical file.

## Findings: more data (TACO)

To get closer to 20k images, we added 1,991 cropped real-litter photos from TACO (CC BY 4.0) to RealWaste and trained both versions with identical settings (layer4 + fc, seed 42, same machine and starting weights). RealWaste-only scored 86.4% (bin 96.4%); RealWaste + TACO scored 84.6% (bin 94.7%). TACO caught 8 more trash items but sent 12 more recyclables to Landfill (6 to 18 of 466), and Misc Trash recall fell (80% to 73%). Like the earlier added datasets, extra off-domain images did not improve our test set, so we kept the RealWaste-only model. A second candidate dataset (Recyclable and Household Waste Classification) was rejected before training: ~18% exact duplicates and mostly web-search images rather than waste. Results: reports/john_realwaste_layer4_seed42/ and reports/john_taco_layer4_seed42/.
