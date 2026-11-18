"""WasteSort demo for the Raspberry Pi: live camera preview, scan, result screen, LEDs.

Flow: preview screen -> scan (tap Scan, press Space, or auto-detect) -> colour result
screen for a few seconds -> back to preview. The same prediction drives the screen and
the Arduino LEDs.

Run on the Pi (from the repo root, with the virtual environment active):

    python ml/pi_demo.py --checkpoint models/best.pt

Useful options:
    --windowed        run in a normal window instead of full screen (laptop testing)
    --camera opencv   use a laptop webcam instead of the Pi camera
    --image FILE      use one still image instead of a camera (no camera needed)
    --no-led          skip the Arduino
    --auto            start with auto-detect on
    --threshold 0.60  below this confidence the result is "Needs review"

Keys: Space = scan, A = toggle auto-detect, Q or Escape = quit.
"""
import argparse
import csv
import glob
import io
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models import ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DISPOSAL_MAP, load_checkpoint  # noqa: E402

REVIEW = "Needs review"
THEMES = {
    "Recyclable": {"bg": "#185FA5", "dark": "#0C447C", "light": "#B5D4F4", "led": b"R", "hint": "Place in the blue bin"},
    "Compost": {"bg": "#3B6D11", "dark": "#27500A", "light": "#C0DD97", "led": b"C", "hint": "Place in the green bin"},
    "Landfill": {"bg": "#A32D2D", "dark": "#791F1F", "light": "#F7C1C1", "led": b"L", "hint": "Place in the red bin"},
    REVIEW: {"bg": "#5F5E5A", "dark": "#2C2C2A", "light": "#D3D1C7", "led": b"X", "hint": "Not sure. Please ask for help"},
}
PAGE_BG = "#FFFFFF"
INK = "#2C2C2A"
MUTED = "#5F5E5A"


class FrameSource:
    """Returns RGB numpy frames from the Pi camera, a laptop webcam, or one still image."""

    def __init__(self, kind, image_path=None):
        self.kind = kind
        self.camera = None
        if image_path:
            self.kind = "image"
            self.still = np.asarray(Image.open(image_path).convert("RGB"))
        elif kind == "picamera2":
            from picamera2 import Picamera2

            self.camera = Picamera2()
            self.camera.configure(self.camera.create_video_configuration(
                main={"size": (1280, 720), "format": "RGB888"}))
            self.camera.start()
            time.sleep(2)  # let auto-exposure settle
        elif kind == "opencv":
            import cv2

            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise RuntimeError("Could not open camera 0")
            for _ in range(5):
                self.camera.read()
        else:
            raise ValueError("unknown camera kind: " + kind)

    def read(self):
        if self.kind == "image":
            return self.still
        if self.kind == "picamera2":
            return self.camera.capture_array()[:, :, ::-1]  # RGB888 is BGR in memory
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return frame[:, :, ::-1]

    def close(self):
        if self.kind == "picamera2":
            self.camera.stop()
        elif self.kind == "opencv":
            self.camera.release()


class Arduino:
    """Sends one letter per result: R, C, L, or X (off). Does nothing if no Arduino is found."""

    def __init__(self, port=None):
        self.serial = None
        try:
            import serial

            candidates = [port] if port else sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/cu.usbmodem*"))
            for candidate in candidates:
                try:
                    self.serial = serial.Serial(candidate, 9600, timeout=1)
                    time.sleep(2)  # the Arduino resets when the port opens
                    print("LEDs: connected on", candidate)
                    break
                except Exception as exc:  # noqa: BLE001
                    print("LEDs: could not open", candidate, exc)
        except ImportError:
            print("LEDs: pyserial is not installed")
        if self.serial is None:
            print("LEDs: no Arduino found, continuing without LEDs")

    def send(self, letter):
        if self.serial is not None:
            try:
                self.serial.write(letter)
            except Exception as exc:  # noqa: BLE001
                print("LEDs: write failed", exc)

    def close(self):
        self.send(b"X")


class Classifier:
    def __init__(self, checkpoint, threshold):
        self.model, self.classes = load_checkpoint(checkpoint, "cpu")
        self.transform = ResNet18_Weights.DEFAULT.transforms()
        self.threshold = threshold

    def predict(self, frame):
        image = Image.fromarray(np.ascontiguousarray(frame))
        tensor = self.transform(image).unsqueeze(0)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0]
        confidence, index = probabilities.max(dim=0)
        label = self.classes[index.item()]
        confidence = confidence.item()
        category = DISPOSAL_MAP[label]
        return {
            "label": label,
            "confidence": confidence,
            "guess_category": category,
            "category": REVIEW if confidence < self.threshold else category,
        }


def to_photo(image):
    """PIL image -> tkinter PhotoImage without needing ImageTk."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PPM")
    return tk.PhotoImage(data=buffer.getvalue())


def with_guide_box(image):
    """Draw the area the model actually sees (its centre crop) so items are placed inside it."""
    image = image.copy()
    width, height = image.size
    box = (width * 0.25, height * 0.06, width * 0.75, height * 0.94)
    ImageDraw.Draw(image).rectangle(box, outline=(255, 255, 255), width=3)
    return image


class App:
    def __init__(self, root, source, classifier, arduino, args):
        self.root, self.source, self.classifier, self.arduino, self.args = root, source, classifier, arduino, args
        self.width = root.winfo_screenwidth() if args.fullscreen else args.width
        self.height = root.winfo_screenheight() if args.fullscreen else args.height
        self.scale = self.width / 720
        self.last_frame = None
        self.mode = "preview"
        self.auto = args.auto
        self.state = "armed"  # auto-detect states: armed, wait_empty
        self.reference = None
        self.previous = None
        self.candidate_since = None
        self.empty_since = None
        self.return_job = None
        self.log_path = Path("captures/demo_log.csv")
        self.build_preview()
        self.build_result()
        self.show_preview()
        self.tick()

    def font(self, size, bold=False):
        return ("Helvetica", max(8, int(size * self.scale)), "bold" if bold else "normal")

    def button(self, parent, text, command, bg, fg="#FFFFFF", size=22):
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=bg,
                         activeforeground=fg, font=self.font(size, True), relief="flat", bd=0,
                         highlightthickness=0, padx=int(20 * self.scale), pady=int(14 * self.scale))

    def build_preview(self):
        self.preview = tk.Frame(self.root, bg=PAGE_BG)
        tk.Label(self.preview, text="WasteSort", font=self.font(30, True), bg=PAGE_BG, fg="#0C447C").pack(pady=(int(24 * self.scale), 0))
        self.hint = tk.Label(self.preview, text="Place the item inside the box", font=self.font(18), bg=PAGE_BG, fg=MUTED)
        self.hint.pack(pady=(int(6 * self.scale), int(14 * self.scale)))
        self.video = tk.Label(self.preview, bg=PAGE_BG)
        self.video.pack()
        self.status = tk.Label(self.preview, text="", font=self.font(16), bg=PAGE_BG, fg=MUTED)
        self.status.pack(pady=int(12 * self.scale))
        self.scan_button = self.button(self.preview, "Scan item", self.scan, "#185FA5", size=28)
        self.scan_button.pack(fill="x", padx=int(30 * self.scale), pady=(int(10 * self.scale), int(8 * self.scale)))
        self.auto_button = self.button(self.preview, "", self.toggle_auto, "#D3D1C7", fg=INK, size=16)
        self.auto_button.pack(fill="x", padx=int(30 * self.scale), pady=int(6 * self.scale))
        self.refresh_auto_button()

    def build_result(self):
        self.result = tk.Frame(self.root, bg=PAGE_BG)
        self.banner = tk.Frame(self.result)
        self.banner.pack(fill="x")
        self.category_label = tk.Label(self.banner, font=self.font(54, True), fg="#FFFFFF")
        self.category_label.pack(pady=(int(50 * self.scale), int(6 * self.scale)))
        self.bin_label = tk.Label(self.banner, font=self.font(22))
        self.bin_label.pack(pady=(0, int(40 * self.scale)))
        self.card = tk.Frame(self.result, bg=PAGE_BG)
        self.card.pack(fill="both", expand=True, padx=int(30 * self.scale), pady=int(20 * self.scale))
        self.thumb = tk.Label(self.card, bg=PAGE_BG)
        self.thumb.pack(pady=(0, int(14 * self.scale)))
        self.detected_label = tk.Label(self.card, font=self.font(18), bg=PAGE_BG, fg=MUTED)
        self.detected_label.pack()
        self.item_label = tk.Label(self.card, font=self.font(36, True), bg=PAGE_BG, fg=INK)
        self.item_label.pack(pady=(0, int(14 * self.scale)))
        self.confidence_label = tk.Label(self.card, font=self.font(18), bg=PAGE_BG, fg=INK)
        self.confidence_label.pack()
        self.bar = tk.Canvas(self.card, width=int(600 * self.scale), height=int(20 * self.scale), bg="#D3D1C7", highlightthickness=0)
        self.bar.pack(pady=int(8 * self.scale))
        self.next_button = self.button(self.result, "Scan next item", self.show_preview, "#185FA5", size=26)
        self.next_button.pack(fill="x", padx=int(30 * self.scale), pady=int(30 * self.scale), side="bottom")

    def refresh_auto_button(self):
        self.auto_button.config(text="Auto-detect: ON (tap to turn off)" if self.auto else "Auto-detect: OFF (tap to turn on)")

    def toggle_auto(self):
        self.auto = not self.auto
        self.reference = None  # next frame becomes the empty-tray reference
        self.candidate_since = self.empty_since = None
        self.state = "armed"
        self.refresh_auto_button()

    def show_preview(self):
        if self.return_job is not None:
            self.root.after_cancel(self.return_job)
            self.return_job = None
        self.arduino.send(b"X")
        self.result.pack_forget()
        self.preview.pack(fill="both", expand=True)
        self.mode = "preview"
        if self.auto:
            self.state = "wait_empty"  # the item must be taken away before the next scan
            self.empty_since = None

    def tick(self):
        try:
            frame = self.source.read()
            self.last_frame = frame
            if self.mode == "preview":
                shown = with_guide_box(Image.fromarray(np.ascontiguousarray(frame)))
                shown = shown.resize((int(self.width - 40 * self.scale), int((self.width - 40 * self.scale) * shown.height / shown.width)))
                self.video_image = to_photo(shown)
                self.video.config(image=self.video_image)
                if self.auto:
                    self.auto_detect(frame)
                else:
                    self.status.config(text="Tap Scan when the item is in the box")
        except Exception as exc:  # noqa: BLE001
            self.status.config(text="Camera problem: " + str(exc))
        self.root.after(100, self.tick)

    def auto_detect(self, frame):
        small = np.asarray(Image.fromarray(np.ascontiguousarray(frame)).resize((64, 36)).convert("L"), dtype=np.int16)
        if self.reference is None:
            self.reference, self.previous = small, small
            return
        from_reference = float(np.abs(small - self.reference).mean())
        from_previous = float(np.abs(small - self.previous).mean())
        self.previous = small
        now = time.monotonic()
        changed = from_reference > self.args.diff_threshold
        if self.state == "wait_empty":
            self.status.config(text="Take the item away to scan the next one")
            if not changed:
                self.empty_since = self.empty_since or now
                if now - self.empty_since >= 1.0:
                    self.state = "armed"
                    self.candidate_since = None
            else:
                self.empty_since = None
            return
        self.status.config(text="Auto-detect on: place an item in the box")
        if changed:
            if self.candidate_since is None or from_previous > self.args.stable_threshold:
                self.candidate_since = now  # still moving, start the wait again
            elif now - self.candidate_since >= self.args.hold:
                self.scan()
        else:
            self.candidate_since = None

    def scan(self):
        if self.mode != "preview" or self.last_frame is None:
            return
        frame = self.last_frame.copy()
        started = time.perf_counter()
        result = self.classifier.predict(frame)
        elapsed = time.perf_counter() - started
        print(f"{result['label']} {result['confidence']:.1%} -> {result['category']} ({elapsed * 1000:.0f} ms)")
        self.log(frame, result)
        self.show_result(frame, result)

    def log(self, frame, result):
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            photo = self.log_path.parent / f"demo_{stamp}.jpg"
            Image.fromarray(np.ascontiguousarray(frame)).save(photo, quality=90)
            new_file = not self.log_path.exists()
            with self.log_path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                if new_file:
                    writer.writerow(["time", "photo", "predicted_class", "confidence", "shown_category"])
                writer.writerow([datetime.now().isoformat(timespec="seconds"), photo.name, result["label"],
                                 f"{result['confidence']:.4f}", result["category"]])
        except OSError as exc:
            print("Could not write log:", exc)

    def show_result(self, frame, result):
        theme = THEMES[result["category"]]
        review = result["category"] == REVIEW
        self.banner.config(bg=theme["bg"])
        self.category_label.config(text=result["category"], bg=theme["bg"])
        self.bin_label.config(text=theme["hint"], bg=theme["bg"], fg=theme["light"])
        thumb = Image.fromarray(np.ascontiguousarray(frame)).resize((int(320 * self.scale), int(180 * self.scale)))
        self.thumb_image = to_photo(thumb)
        self.thumb.config(image=self.thumb_image)
        self.detected_label.config(text="Best guess" if review else "Detected as")
        self.item_label.config(text=result["label"])
        percent = result["confidence"] * 100
        self.confidence_label.config(text=f"Confidence {percent:.0f}%")
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, int(600 * self.scale * result["confidence"]), int(20 * self.scale), fill=theme["bg"], width=0)
        self.next_button.config(bg=theme["bg"], activebackground=theme["bg"])
        self.preview.pack_forget()
        self.result.pack(fill="both", expand=True)
        self.mode = "result"
        self.arduino.send(theme["led"])
        self.return_job = self.root.after(int(self.args.result_seconds * 1000), self.show_preview)

    def close(self):
        self.arduino.close()
        self.source.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--camera", choices=["picamera2", "opencv"], default="picamera2")
    parser.add_argument("--image", type=Path, help="use one still image instead of a camera")
    parser.add_argument("--port", help="Arduino serial port (found automatically if omitted)")
    parser.add_argument("--no-led", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--auto", action="store_true", help="start with auto-detect on")
    parser.add_argument("--threshold", type=float, default=0.60, help="below this confidence: Needs review")
    parser.add_argument("--result-seconds", type=float, default=6.0)
    parser.add_argument("--diff-threshold", type=float, default=8.0, help="auto-detect: change from the empty tray (0-255)")
    parser.add_argument("--stable-threshold", type=float, default=4.0, help="auto-detect: max change between frames")
    parser.add_argument("--hold", type=float, default=1.0, help="auto-detect: seconds the item must stay still")
    args = parser.parse_args()
    args.fullscreen = not args.windowed

    classifier = Classifier(args.checkpoint, args.threshold)
    source = FrameSource(args.camera, args.image)
    arduino = Arduino.__new__(Arduino)
    if args.no_led:
        arduino.serial = None
    else:
        arduino = Arduino(args.port)

    root = tk.Tk()
    root.title("WasteSort")
    root.configure(bg=PAGE_BG)
    if args.fullscreen:
        root.attributes("-fullscreen", True)
    else:
        root.geometry(f"{args.width}x{args.height}")
    root.update_idletasks()
    app = App(root, source, classifier, arduino, args)
    root.bind("<space>", lambda event: app.scan())
    root.bind("a", lambda event: app.toggle_auto())
    root.bind("q", lambda event: app.close())
    root.bind("<Escape>", lambda event: app.close())
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
