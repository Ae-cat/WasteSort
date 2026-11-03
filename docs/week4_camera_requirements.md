# Week 4 Camera Integration: Requirements and Blockers

Updated November 3, 2026 for the laptop live demo.
Items marked **Verified** come from reading the code; real camera usability and latency remain pending because camera access was denied and the trained checkpoint is missing.

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

The single-capture command uses `camera.capture_frame()` to save `captures/latest.jpg`, then `predict_image(model, classes, path)` reads the file.
The new laptop command, `python ml/live_demo.py --timing`, opens webcam 0 once, loads the checkpoint once on CPU, discards five initial camera frames, and previews continuously.
With the preview focused, Space reads a fresh frame, overwrites `captures/latest.jpg`, and prints one class, category, and confidence; Q exits.
Both commands retain the same file-based prediction and disposal mapping.
Camera release and window cleanup run on normal exit and failures.
The preview pauses while synchronous CPU inference runs; its practical responsiveness needs a real-camera check.

| Option | How | Pros | Cons |
|---|---|---|---|
| A. File path (implemented) | capture -> save JPEG -> predict from path | Reuses existing prediction; easy to debug | Extra disk write and JPEG encode/decode; live command keeps camera open |
| B. In memory (proposed) | capture -> BGR to RGB -> PIL image -> same transform -> model | Faster | Needs a small change to `model.py` (accept a PIL image); agree with Frank |

**Recommendation:** use Option A for the Week 4 laptop loop, and move to Option B only if timing on the Pi shows the file step matters.

`predict.py` retains its existing single-capture behavior and reloads the checkpoint on each invocation.
The live command keeps the model and camera in memory throughout the session.

## 4. Latency

**Pending: zero valid laptop webcam prediction samples.**
The actual camera-open probe returned `opened False`, with OpenCV reporting that camera access was denied.
It released the camera and closed windows without saving images.
The trained model and dataset are also absent; source discovery and deterministic image selection are in [the prediction log](week3_prediction_log.md).
No preview, physical key presses, or end-to-end trained inference could be verified on this laptop session.

| Step | Laptop CPU mean (range), n=10 | Pi 5 CPU mean (range), n=10 |
|---|---|---|
| Camera read (`camera_read`) | Pending: camera permission and checkpoint | Pending: hardware run |
| JPEG encode + disk write (`file_write`) | Pending | Pending |
| File read + JPEG decode + RGB conversion (`file_read_conversion`) | Pending | Pending |
| Tensor preprocessing (`preprocessing`) | Pending | Pending |
| CPU forward pass + output selection (`inference`) | Pending | Pending |
| Remaining prediction overhead (`other`) | Pending | Pending |
| Total per prediction (`total`) | Pending | Pending |

Laptop: MacBook Pro (Mac16,1), Apple M4, 10 CPU cores (4 performance, 6 efficiency), 16 GB RAM, macOS 26.5.1 arm64.
Environment: Python 3.14 virtual environment, PyTorch 2.14.1, torchvision 0.29.1, OpenCV 5.0.0.93, Pillow 12.3.0.
Actual camera resolution, lighting, background, checkpoint SHA-256, and sample statistics: **pending**.
Pi details and values remain pending; no Pi setup commands were executed.

To complete the laptop measurements:

1. Obtain the existing team checkpoint and record its SHA-256.
Grant Camera permission to the application running Python under **System Settings > Privacy & Security > Camera** on macOS, then restart that application.
2. Run `python ml/live_demo.py --timing` from the repository root on a graphical desktop.
Record camera resolution, lighting/background, power mode, and relevant CPU load.
Keep the object centered and the scene steady.
3. The command discards five camera frames automatically.
Press Space once to warm up CPU inference and exclude that result from measurement.
Then press Space ten times, saving each `timing_ms` line and prediction output.
Record all ten values per stage and calculate arithmetic mean plus minimum-maximum range in milliseconds.
4. Verify exactly ten measured results, one camera open, one checkpoint load, useful preview framing, Q and window-close exit, and cleanup after failures.

Stage durations use `time.perf_counter()` and timing is supported only on CPU.
`total` starts immediately before the fresh post-Space camera read and ends after prediction returns, including file write/read, JPEG conversion, tensor preprocessing, inference, and remaining overhead.
It excludes model load, camera startup, frame warm-up, preview/key wait, and console printing.
Thus file overhead is visible and included in total rather than being hidden inside inference.
The default `predict_image` return fields and behavior remain unchanged when timing is omitted.

**Proposed target:** about 1-2 seconds per prediction for a "press a key / place an item" demo. This is not real-time video. If the Pi is slower, the first options are a smaller input or a lighter model such as MobileNetV3 (one change at a time).

## 5. Output the camera pipeline must produce

- Dataset class, WasteSort category, and confidence, as in `predict.py` today.
- **Gap:** `ml/model.py` maps with `DISPOSAL_MAP[label]` and has no confidence threshold, so there is no safe "Needs review" result. The threshold (0.60) and safe handling exist only in `ml/predict_ariella.py`. Merge into the main pipeline with Frank before the display and LEDs depend on it.

## 6. Blockers and risks

| Blocker / risk | Impact | Next action |
|---|---|---|
| Hardware approval and delivery status unverified | Earlier planning estimated arrival around Oct 28-31; actual availability is unverified | Confirm approval and delivery; keep the laptop webcam + Arduino fallback |
| Training images are overhead on gray concrete | Webcam accuracy may be lower than 80.4% | Collect a labeled webcam test set (Adam and Grecia) and compare |
| Center crop cuts the edges of the frame | Items off-center may be missed | Center the item; add a marked spot or on-screen guide |
| No "Needs review" in the main pipeline | Low-confidence or unknown results are not handled safely | Merge threshold into `model.py` with Frank |
| Miscellaneous Trash is often predicted as Food Organics (11 test images) | Items could be sent to the compost bin | Frank and John investigate; consider a safeguard |
| Live preview and ten trained webcam predictions unverified | Real usability and latency unknown | Restore Camera permission and checkpoint, then complete the measurement procedure |
| macOS camera permission denied to the terminal | Webcam capture fails | Grant Camera permission in System Settings |
| Pi Camera Module 3 and OpenCV | OpenCV `VideoCapture` may not read the Pi camera directly; Picamera2 is usually used | **To check** during Pi setup (Adam) |
| Pi 5 needs a 22-pin to 15-pin camera cable | Camera will not connect without it | Included in the hardware order |
| PyTorch install and inference speed on the Pi | Could delay or slow the pipeline | Test early in Week 5; fallback model ready |
| Network access to the Pi (campus Wi-Fi may block SSH) | Remote teammates cannot log in | Test on arrival; use hotspot or in-person sessions |
| One set of hardware, held by one person | Others depend on remote access | Hardware keeper plus two in-person sessions |

## 7. Decisions needed this week

1. Option A (file) is implemented for the Week 4 laptop loop.
Consider Option B only after measured file overhead warrants it.
2. Agree with Frank on merging the confidence threshold into `model.py`.
3. Agree who holds the hardware when it arrives. Proposed: Ariella.
