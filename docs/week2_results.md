# Week 2 ML/Dataset and CV/Camera Results

## Completed

- Preserved the existing dataset inspection, duplicate checking, split generation, and ImageNet baseline scripts.
- Added a portable `WasteDataset` path override so the committed split file works when the dataset is stored elsewhere.
- Added ResNet18 transfer learning with a 9-class output layer, validation metrics, and best-checkpoint saving.
- Added single-image inference and disposal mapping to Recyclable, Compost, or Landfill.
- Added camera capture as an optional input to the same inference command.

## Commands

```bash
python ml/train.py --data-root data/realwaste-main/RealWaste --freeze-backbone
python ml/predict.py --image path/to/test.jpg
python ml/predict.py --camera 0
python -m unittest discover -s tests -v
```

## Test record (2026-10-14)

- Automated dataset and end-to-end untrained-model inference tests: **2 passed** in 0.258 seconds.
- Python compilation and both command-line help checks: **passed**.
- Dataset inspection: **4,752 images**, nine classes, all sampled images were RGB 524 x 524.
- Stratified split loading: **3,323 train / 710 validation / 719 test** images.
- Development dependencies: PyTorch, torchvision, NumPy, pandas, Pillow, matplotlib, scikit-learn, and OpenCV are installed.
- ImageNet ResNet18 baseline: **completed** for one deterministic sample per RealWaste class.
- RealWaste training: not run in this checkout because the dataset is intentionally excluded from Git.
- Physical camera interface: OpenCV loads correctly, but macOS denied camera permission to the current terminal process. Grant Camera permission before the live demo.

## Baseline output

```text
Cardboard            -> cassette (80%)
Food Organics        -> hatchet (26%)
Glass                -> beer bottle (69%)
Metal                -> beer bottle (60%)
Miscellaneous Trash  -> screwdriver (65%)
Paper                -> packet (77%)
Plastic              -> tray (39%)
Textile Trash        -> handkerchief (60%)
Vegetation           -> walking stick (69%)
```

This is an expected baseline: the unmodified model predicts ImageNet labels, which demonstrates why the RealWaste classifier needs transfer learning in the next model-training phase.

The model checkpoint and captured images are local artifacts and must not be committed.

## Current assumptions and limitations

- The MVP maps the broad RealWaste `Plastic` label to `Recyclable`. Real disposal rules vary by plastic type and location, so this rule should be narrowed when the team chooses a target municipality.
- Accuracy cannot be reported until training runs against the locally downloaded RealWaste images.
