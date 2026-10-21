# Software Plan

## Goal

Connect the camera, ML model, and user interface.

## Planned Pipeline

Camera
↓
Image
↓
ML Model
↓
Prediction
↓
Disposal Category
↓
Display

## Architecture

Each stage is a small, separate piece so teammates can work on them independently
and swap parts later (file input -> laptop webcam -> Pi Camera) without touching the model code.

| Stage | Where it lives | Status |
|-------|----------------|--------|
| Image input | `ml/predict.py` (file path today); OpenCV webcam capture next | file input done |
| Preprocessing | `ml/predict.py` (ResNet18 torchvision transforms, same as training) | done |
| Model | ResNet18 fine-tuned on RealWaste, checkpoint at `models/baseline.pth` (gitignored) | waiting on trained model |
| Prediction -> category | `CATEGORY_MAP` in `ml/predict.py` | Plastic still undecided |
| Output | text now; display + LEDs in Weeks 6-7 | text only |

The single entry point is `predict(image_path)`, which returns the dataset class,
WasteSort category, and confidence. Everything downstream (display, LEDs) only needs that result.
Anything without a configured category is reported as "Needs review" instead of guessing.

## Questions

### How will we capture camera images?

- Now: load an image from a file (the pipeline takes a path).
- Next (Week 4): laptop webcam with OpenCV (`cv2.VideoCapture`), saving a frame and passing it to `predict()`.
- Later (Week 5): Raspberry Pi Camera Module 3 on the Pi 5, replacing the webcam. The rest of the pipeline stays the same.

### How will the software run the model?

- PyTorch/torchvision ResNet18 (transfer learning), loaded once at startup and kept in memory.
- Preprocessing at inference must match training exactly: resize, center crop, ImageNet normalization, no random augmentation.
- Run locally on the laptop first, then on the Raspberry Pi 5. If it is too slow on the Pi, reduce input size or try a smaller model (e.g. MobileNetV3) before buying an accelerator.

### How will the result be displayed?

- Now (Weeks 1-5): plain text in the terminal, e.g. `Detected: Cardboard / Category: Recyclable / Confidence: 91%`.
- Week 6: simple screen on the Pi 5-inch display showing the item, the category in large text with a category color, and confidence.
- Week 7: LEDs mirror the category (Blue = Recyclable, Green = Compost, Red = Landfill). One option is the Pi sending the category to an Arduino over USB serial to drive the LEDs.

## Initial UI Concept

```
+------------------------------+
|   WasteSort                  |
|                              |
|   [ Image preview ]          |
|                              |
|   Item:        Cardboard     |
|   CATEGORY:    RECYCLABLE    |   <- big text, colored background
|   Confidence:  91%           |
|                              |
|   [ Scan again ]  (optional) |
+------------------------------+
```

Rules: category is the largest element; low-confidence or unmapped results show
"Needs review" in a neutral color; no LED lights up in that case.

## Open Items

- Team decision on the Plastic mapping (and confirm the other 8 class mappings).
- Confirm the trained checkpoint format and class order, then drop it in `models/baseline.pth` and run `python ml/predict.py <image>`.
- Choose who builds the Week 4 webcam capture step.

## Week 1 Deliverable

Software team will research the application architecture and
create an initial UI concept.
