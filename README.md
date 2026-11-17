# WasteSort

WasteSort classifies an image into one of the nine RealWaste classes and maps
the prediction to Recyclable, Compost, or Landfill.

Week 2 setup and test instructions are recorded in
[`docs/week2_results.md`](docs/week2_results.md).
The meeting walkthrough is in [`docs/week2_demo.md`](docs/week2_demo.md).
AI-powered computer vision system for identifying waste and recommending disposal categories.

## Predict waste from an image or laptop webcam

Install the dependencies.
The repository includes trained experimental checkpoints documented in
[`docs/week5_controlled_results.md`](docs/week5_controlled_results.md).
For the laptop demo, pass `--checkpoint reports/week5_controlled_seed42/layer4/best.pt`.
The historical default `models/resnet18_realwaste.pt` is still absent; most weight files are ignored by Git.
The included layer4 checkpoint is an experimental model whose webcam accuracy still needs testing.
The dataset download and pending prediction checks are documented in
[`docs/week3_prediction_log.md`](docs/week3_prediction_log.md).

Run these commands from the repository root:

```sh
python -m pip install -r requirements.txt
python ml/predict.py --image path/to/image.jpg --checkpoint reports/week5_controlled_seed42/layer4/best.pt
python ml/predict.py --camera 0 --checkpoint reports/week5_controlled_seed42/layer4/best.pt
python ml/live_demo.py --timing --checkpoint reports/week5_controlled_seed42/layer4/best.pt
```

The live demo loads the checkpoint once on CPU, opens webcam 0 once, discards five initial frames, and shows a continuous preview.
Focus the preview window and press Space to classify a fresh capture, or Q to exit.
Each Space press overwrites `captures/latest.jpg` and prints a waste class, disposal category, and confidence.
Omit `--timing` for prediction output alone, or use `--checkpoint path/to/team-model.pt` for another trained checkpoint.
Timing definitions and outstanding laptop measurements are in
[`docs/week4_camera_requirements.md`](docs/week4_camera_requirements.md).

On macOS, enable Camera access for the application running Python (Terminal, iTerm, or your editor) in **System Settings > Privacy & Security > Camera**, then restart that application.
A graphical desktop and the full `opencv-python` package are required for the preview.
The written hardware handoff for Ariella is in
[`docs/pi_setup_checklist.md`](docs/pi_setup_checklist.md).

Week 3 prediction logs, reproduced held-out results, and the independent external
image collection/test workflow are in
[`docs/week3_carryover_results.md`](docs/week3_carryover_results.md).

Week 4 work:

- [Baseline error gallery and visual review](docs/week4_error_analysis.md)
- [Controlled 5-versus-10-epoch experiment](docs/week4_epoch_experiment.md)

## Inspect the ImageNet baseline

Install the dependencies, then provide an image path:

```sh
python -m pip install -r requirements.txt
python -m ml.baseline_predict --image path/to/image.jpg
```

The script converts the image to RGB, applies the pretrained ResNet18 preprocessing, and validates a finite float32 tensor of shape `(1, 3, 224, 224)` before prediction.
It prints the input format, predicted ImageNet label, and confidence.
The first prediction downloads model weights if they are not cached.
This baseline uses ImageNet labels, not waste disposal categories.
Run without `--image` to keep the existing dataset sampling workflow.

## Evaluate a trained checkpoint

Evaluate a trained checkpoint on the held-out test split with:

```bash
python3 ml/evaluate.py --checkpoint reports/week5_controlled_seed42/layer4/best.pt
```

The 10-epoch candidate's validation-selected Compost review threshold and fixed
test confirmation are documented in [Week 4 threshold results](docs/week4_10epoch_threshold.md).
