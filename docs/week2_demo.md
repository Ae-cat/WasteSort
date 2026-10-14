# Week 2 Demo Guide

## What to show

1. Open `ml/sample_grid.png` to show five examples from each of the nine RealWaste classes.
2. Explain that the dataset contains 4,752 images and uses a reproducible stratified split:
   - 3,323 training images
   - 710 validation images
   - 719 test images
3. Run the dataset loader:

   ```bash
   python3 ml/dataset.py
   ```

   Point out the `[8, 3, 224, 224]` batch shape and the nine class names.
4. Run the pretrained baseline:

   ```bash
   TORCH_HOME=.torch python3 ml/baseline_predict.py
   ```

   Explain that outputs such as `Cardboard -> cassette` are ImageNet labels. They establish a before-training baseline rather than a finished WasteSort classifier.
5. Show `ml/train.py`, `ml/predict.py`, and `ml/camera.py` as the ready pipeline for the next phase.

## Short team update

> I completed the Week 2 dataset and development setup. RealWaste has 4,752 images across nine classes, and the reproducible split contains 3,323 training, 710 validation, and 719 test images. The PyTorch loader produces normalized ResNet18 batches successfully. I also preserved and ran the pretrained ImageNet baseline, added the transfer-learning and prediction pipeline, and prepared camera capture. The remaining camera check only needs macOS Camera permission, and model accuracy will be measured after training.
