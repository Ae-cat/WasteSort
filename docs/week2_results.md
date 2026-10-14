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
- RealWaste training: not run in this checkout because the dataset is intentionally excluded from Git.
- Physical camera capture: not run because this execution environment has no camera access and OpenCV is not installed.

The model checkpoint and captured images are local artifacts and must not be committed.

## Current assumptions and limitations

- The MVP maps the broad RealWaste `Plastic` label to `Recyclable`. Real disposal rules vary by plastic type and location, so this rule should be narrowed when the team chooses a target municipality.
- Accuracy cannot be reported until training runs against the locally downloaded RealWaste images.
