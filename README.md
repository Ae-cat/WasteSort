# WasteSort
AI-powered computer vision system for identifying waste and recommending disposal categories.

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
