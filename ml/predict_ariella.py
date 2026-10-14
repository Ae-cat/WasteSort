"""
Currently have image as a placeholder file
WasteSort prediction pipeline: image -> preprocessing -> model -> class/category/confidence.

Adds to the team pipeline (ml/predict.py + ml/model.py): a confidence threshold, a safe
"Needs review" result for unmapped/low-confidence predictions, and a --repeat consistency check.

Usage (from the repo root; use python3 on a Mac):
    python3 ml/predict_ariella.py path/to/image.jpg
    python3 ml/predict_ariella.py path/to/image.jpg --repeat 5   # check output is consistent

Model checkpoint: models/resnet18_realwaste.pt (saved by ml/train.py, gitignored). Expected format:
  {"model_state": ..., "classes": [...], ...}. A plain state_dict, or a dict using the keys
  "state_dict" / "model_state_dict", is also accepted.
If the checkpoint has no class list, DEFAULT_CLASSES is used. That list MUST match the
order used in training (ml/dataset.py sorts class names alphabetically).
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

MODEL_PATH = Path("models/resnet18_realwaste.pt")

# RealWaste folder names, in alphabetical (ImageFolder) order.
DEFAULT_CLASSES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]

# Dataset class -> WasteSort category. None = not decided (reported as needs review).
# Kept in sync with DISPOSAL_MAP in ml/model.py and docs/data_plan.md.
# Plastic -> Recyclable is a proposed MVP rule; narrow it once a target municipality is chosen.
CATEGORY_MAP = {
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
NEEDS_REVIEW = "Needs review (category not configured)"

# Must match the preprocessing used during training/evaluation (no random augmentation).
_preprocess = ResNet18_Weights.DEFAULT.transforms()

_model = None
_classes = None


def load_model(path=MODEL_PATH):
    """Load the saved checkpoint once and cache it."""
    global _model, _classes
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No model checkpoint at {path}. Train one with ml/train.py or get the .pt file "
            "from the model owner (checkpoints are gitignored, so share it separately)."
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    classes = DEFAULT_CLASSES
    state = ckpt
    for key in ("model_state", "state_dict", "model_state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            classes = ckpt.get("classes", DEFAULT_CLASSES)
            break
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state)
    model.eval()
    _model, _classes = model, list(classes)
    return _model


# Below this confidence the result is reported as "Needs review" instead of a category.
CONFIDENCE_THRESHOLD = 0.60


def predict(image_path):
    """Return the predicted item, category, and confidence."""
    if _model is None:
        load_model()

    image = Image.open(image_path).convert("RGB")
    x = _preprocess(image).unsqueeze(0)

    with torch.no_grad():
        probs = _model(x).softmax(dim=1)[0]

    conf, idx = probs.max(dim=0)
    item = _classes[idx.item()]
    confidence = conf.item()

    if confidence < CONFIDENCE_THRESHOLD:
        category = NEEDS_REVIEW
    else:
        category = CATEGORY_MAP.get(item) or NEEDS_REVIEW

    return {
        "item": item,
        "category": category,
        "confidence": confidence
    }


def format_result(result):
    return (
        f"Detected: {result['item']}\n"
        f"Category: {result['category']}\n"
        f"Confidence: {result['confidence']:.0%}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WasteSort image prediction")
    parser.add_argument("image", help="path to an image file")
    parser.add_argument("--model", default=str(MODEL_PATH), help="checkpoint path")
    parser.add_argument("--repeat", type=int, default=1, help="run N times to check consistency")
    args = parser.parse_args()

    load_model(args.model)
    results = [predict(args.image) for _ in range(args.repeat)]
    print(format_result(results[0]))
    if args.repeat > 1:
        same = all(r == results[0] for r in results)
        print(f"Consistent across {args.repeat} runs: {same}")
