"""Shared model loading, prediction, and disposal-category helpers."""
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


DISPOSAL_MAP = {
    "Cardboard": "Recyclable",
    "Food Organics": "Compost",
    "Glass": "Recyclable",
    "Metal": "Recyclable",
    "Miscellaneous Trash": "Landfill",
    "Paper": "Recyclable",
    "Plastic": "Recyclable",
    "Textile Trash": "Landfill",
    "Vegetation": "Compost",
}


def build_model(num_classes, pretrained=True, freeze_backbone=False):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, classes


def predict_image(model, classes, image_path, device="cpu", *, timings=None):
    """Predict a saved image, optionally recording CPU stage durations in seconds.

    Timing leaves the prediction result unchanged. Restrict it to CPU so async
    accelerator work cannot produce misleading inference measurements.
    """
    if timings is not None and torch.device(device).type != "cpu":
        raise ValueError("Prediction timing is supported only on CPU")
    transform = ResNet18_Weights.DEFAULT.transforms()
    started = perf_counter()
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
    converted = perf_counter()
    image = transform(rgb).unsqueeze(0).to(device)
    preprocessed = perf_counter()
    with torch.inference_mode():
        probabilities = model(image).softmax(dim=1)[0]
    confidence, index = probabilities.max(dim=0)
    inferred = perf_counter()
    if timings is not None:
        timings.update(
            file_read_conversion=converted - started,
            preprocessing=preprocessed - converted,
            inference=inferred - preprocessed,
        )
    label = classes[index.item()]
    return {
        "label": label,
        "confidence": confidence.item(),
        "disposal_category": DISPOSAL_MAP[label],
    }
