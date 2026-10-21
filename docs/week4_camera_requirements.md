# Week 4 Camera Integration: Requirements and Blockers

Owner: Ariella (project lead). Based on the current `ml/` code on `main`. Items marked **Verified** come from reading the code; items marked **To check** are assumptions that need a test.

## 1. What the model expects

- **Input tensor:** 3 x 224 x 224, float, one image per call (a batch dimension is added). **Verified** (`ml/model.py`, `ResNet18_Weights.DEFAULT.transforms()`).
- **Preprocessing (same for training, validation, test, and prediction):** resize shorter side to 256, center-crop 224 x 224, scale to 0-1, normalize with ImageNet mean/std. No random augmentation. **Verified**.
- **Colour order:** RGB. `predict_image` opens the file with PIL and converts to RGB. OpenCV frames are BGR, so a frame passed directly must be converted (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`). **Verified**.
- **Training images:** 524 x 524, overhead, gray concrete background. **Verified** (`data_plan.md`).

## 2. What the camera will send

- **Laptop webcam (Week 4):** typically a 4:3 or 16:9 frame (for example 640 x 480). The center crop keeps only the middle of the frame: for a 640 x 480 frame, about 66% of the width and 87% of the height. **To check** with a real frame.
- **Requirement:** the item must be centered in the frame, with a plain background and steady lighting. Mount or mark a fixed spot.
- **Pi Camera Module 3 (Week 5):** 12 MP, autofocus. Capture at a smaller resolution (the model only uses 224 x 224 after resizing) to save time. **To check**.

## 3. How the prediction function is called with a camera frame

Current flow (works today): `camera.capture_frame()` saves `captures/latest.jpg`, then `predict_image(model, classes, path)` reads the file. Simple and already tested on a laptop.

| Option | How | Pros | Cons |
|---|---|---|---|
| A. File path (current) | capture -> save JPEG -> predict from path | Works now; easy to debug | Extra disk write and JPEG encode/decode; camera opens and closes on every call |
| B. In memory (proposed) | capture -> BGR to RGB -> PIL image -> same transform -> model | Faster; needed for a live loop | Needs a small change to `model.py` (accept a PIL image); agree with Frank |

**Recommendation:** use Option A for the Week 4 laptop loop, and move to Option B only if timing on the Pi shows the file step matters.

For a live loop, also: keep the camera open between captures, discard the first few frames (auto-exposure settling), and load the model once and keep it in memory (currently `predict.py` reloads the checkpoint on every run).

## 4. Latency

Not measured yet. Measure each step separately on the laptop (Week 4) and again on the Pi (Week 5):

| Step | Laptop | Pi 5 |
|---|---|---|
| Camera capture (after warm-up) | to measure | to measure |
| Preprocessing | to measure | to measure |
| Model inference | to measure | to measure |
| Total per prediction | to measure | to measure |

**Proposed target:** about 1-2 seconds per prediction for a "press a key / place an item" demo. This is not real-time video. If the Pi is slower, the first options are a smaller input or a lighter model such as MobileNetV3 (one change at a time).

## 5. Output the camera pipeline must produce

- Dataset class, WasteSort category, and confidence, as in `predict.py` today.
- **Gap:** `ml/model.py` maps with `DISPOSAL_MAP[label]` and has no confidence threshold, so there is no safe "Needs review" result. The threshold (0.60) and safe handling exist only in `ml/predict_ariella.py`. Merge into the main pipeline with Frank before the display and LEDs depend on it.

## 6. Blockers and risks

| Blocker / risk | Impact | Next action |
|---|---|---|
| Hardware not approved or delivered (about 1 week for approval, plus shipping) | Pi work cannot start before about Oct 28-31 | Follow up on approval; keep the laptop webcam + Arduino fallback |
| Training images are overhead on gray concrete | Webcam accuracy may be lower than 80.4% | Collect a labeled webcam test set (Adam and Grecia) and compare |
| Center crop cuts the edges of the frame | Items off-center may be missed | Center the item; add a marked spot or on-screen guide |
| No "Needs review" in the main pipeline | Low-confidence or unknown results are not handled safely | Merge threshold into `model.py` with Frank |
| Miscellaneous Trash is often predicted as Food Organics (11 test images) | Items could be sent to the compost bin | Frank and John investigate; consider a safeguard |
| Camera opened and closed on every capture; model reloaded every run | Slow and unreliable for a live loop | Live loop keeps both in memory (Adam) |
| macOS camera permission denied to the terminal | Webcam capture fails | Grant Camera permission in System Settings |
| Pi Camera Module 3 and OpenCV | OpenCV `VideoCapture` may not read the Pi camera directly; Picamera2 is usually used | **To check** during Pi setup (Adam) |
| Pi 5 needs a 22-pin to 15-pin camera cable | Camera will not connect without it | Included in the hardware order |
| PyTorch install and inference speed on the Pi | Could delay or slow the pipeline | Test early in Week 5; fallback model ready |
| Network access to the Pi (campus Wi-Fi may block SSH) | Remote teammates cannot log in | Test on arrival; use hotspot or in-person sessions |
| One set of hardware, held by one person | Others depend on remote access | Hardware keeper plus two in-person sessions |

## 7. Decisions needed this week

1. Option A (file) or Option B (in memory) for the Week 4 loop. Proposed: Option A.
2. Agree with Frank on merging the confidence threshold into `model.py`.
3. Agree who holds the hardware when it arrives. Proposed: Ariella.