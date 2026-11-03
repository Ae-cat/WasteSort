# WasteSort

WasteSort classifies an image into one of the nine RealWaste classes and maps
the prediction to Recyclable, Compost, or Landfill.

Week 2 setup and test instructions are recorded in
[`docs/week2_results.md`](docs/week2_results.md).
The meeting walkthrough is in [`docs/week2_demo.md`](docs/week2_demo.md).
AI-powered computer vision system for identifying waste and recommending disposal categories.

## Predict waste from an image or laptop webcam

Install dependencies and obtain the team's trained RealWaste checkpoint separately at
`models/resnet18_realwaste.pt` (weights are ignored by Git).
No trained-checkpoint download link is currently recorded in this repository.
The dataset download and pending prediction checks are documented in
[`docs/week3_prediction_log.md`](docs/week3_prediction_log.md).

Run these commands from the repository root:

```sh
python -m pip install -r requirements.txt
python ml/predict.py --image path/to/image.jpg
python ml/predict.py --camera 0
python ml/live_demo.py --timing
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
python3 ml/evaluate.py
```
