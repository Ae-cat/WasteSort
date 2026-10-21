# WasteSort

WasteSort classifies an image into one of the nine RealWaste classes and maps
the prediction to Recyclable, Compost, or Landfill.

Week 2 setup and test instructions are recorded in
[`docs/week2_results.md`](docs/week2_results.md).
The meeting walkthrough is in [`docs/week2_demo.md`](docs/week2_demo.md).
AI-powered computer vision system for identifying waste and recommending disposal categories.

<<<<<<< HEAD
Evaluate a trained checkpoint on the held-out test split with:

```bash
python3 ml/evaluate.py
```
=======
## Predict from an image file

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
>>>>>>> origin/adam
