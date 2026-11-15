"""Exercise live lifecycle with a simulated camera, never real hardware."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))

import camera
import live_demo
from model import predict_image


class LiveDemoTest(unittest.TestCase):
    def setUp(self):
        self.camera = Mock()
        self.camera.isOpened.return_value = True
        self.camera.read.return_value = (True, object())
        self.cv2 = Mock()
        self.cv2.error = cv2.error
        self.cv2.VideoCapture.return_value = self.camera
        self.cv2.waitKey.side_effect = [ord(" ")] * 10 + [ord("q")]
        self.cv2.getWindowProperty.return_value = 1
        self.result = {"label": "Glass", "disposal_category": "Recyclable", "confidence": 0.9}

    def run_demo(self, **kwargs):
        return live_demo.run_demo(Path("trained.pt"), Path("captures/latest.jpg"), **kwargs)

    def test_ten_presses_load_and_open_once_then_quit(self):
        with patch.object(live_demo, "load_opencv", return_value=self.cv2), \
                patch.object(live_demo, "load_checkpoint", return_value=(object(), ["Glass"])) as load, \
                patch.object(live_demo, "save_frame", return_value=Path("captures/latest.jpg")) as save, \
                patch.object(live_demo, "predict_image", return_value=self.result) as predict, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.run_demo()
        load.assert_called_once_with(Path("trained.pt"), "cpu")
        self.cv2.VideoCapture.assert_called_once_with(0)
        self.assertEqual(save.call_count, 10)
        self.assertEqual(predict.call_count, 10)
        self.assertEqual(self.camera.read.call_count, 5 + 11 + 10)
        self.assertEqual(output.getvalue().count("prediction=Glass"), 10)
        for call in predict.call_args_list:
            self.assertEqual(call.args[3], "cpu")
            self.assertIsNone(call.kwargs["timings"])
        self.camera.release.assert_called_once()
        self.cv2.destroyAllWindows.assert_called_once()

    def test_camera_open_and_frame_failures_release_resources(self):
        for opened, read_result, message in [
            (False, (True, object()), "Could not open camera 0"),
            (True, (False, None), "Could not read a frame from camera 0"),
            (True, (True, None), "Could not read a frame from camera 0"),
        ]:
            with self.subTest(message=message, opened=opened):
                self.camera.reset_mock()
                self.cv2.destroyAllWindows.reset_mock()
                self.camera.isOpened.return_value = opened
                self.camera.read.return_value = read_result
                with patch.object(live_demo, "load_opencv", return_value=self.cv2), \
                        patch.object(live_demo, "load_checkpoint", return_value=(object(), ["Glass"])), \
                        self.assertRaisesRegex(RuntimeError, message):
                    self.run_demo()
                self.camera.release.assert_called_once()
                self.cv2.destroyAllWindows.assert_called_once()

    def test_prediction_failure_releases_camera_and_windows(self):
        with patch.object(live_demo, "load_opencv", return_value=self.cv2), \
                patch.object(live_demo, "load_checkpoint", return_value=(object(), ["Glass"])), \
                patch.object(live_demo, "save_frame", side_effect=RuntimeError("disk failure")), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "disk failure"):
            self.run_demo()
        self.camera.release.assert_called_once()
        self.cv2.destroyAllWindows.assert_called_once()

    def test_missing_checkpoint_never_opens_camera(self):
        with patch.object(live_demo, "load_opencv", return_value=self.cv2), \
                patch.object(live_demo, "load_checkpoint", side_effect=FileNotFoundError("trained.pt")), \
                self.assertRaises(FileNotFoundError):
            self.run_demo()
        self.cv2.VideoCapture.assert_not_called()
        self.cv2.destroyAllWindows.assert_called_once()

    def test_thrown_read_error_reaches_cli_without_cleanup_masking_it(self):
        self.camera.read.side_effect = cv2.error("original read failure")
        self.cv2.destroyAllWindows.side_effect = cv2.error("cleanup failure")
        with patch.object(live_demo, "load_opencv", return_value=self.cv2), \
                patch.object(live_demo, "load_checkpoint", return_value=(object(), ["Glass"])), \
                patch.object(sys, "argv", ["live_demo.py"]), \
                contextlib.redirect_stderr(io.StringIO()) as output, \
                self.assertRaises(SystemExit) as raised:
            live_demo.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Error: OpenCV camera or preview operation failed: original read failure", output.getvalue())
        self.assertNotIn("cleanup failure", output.getvalue())
        self.camera.release.assert_called_once()
        self.cv2.destroyAllWindows.assert_called_once()

    def test_thrown_display_error_closes_resources(self):
        self.cv2.imshow.side_effect = cv2.error("preview unavailable")
        with patch.object(live_demo, "load_opencv", return_value=self.cv2), \
                patch.object(live_demo, "load_checkpoint", return_value=(object(), ["Glass"])), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "OpenCV camera or preview operation failed: preview unavailable"):
            self.run_demo()
        self.camera.release.assert_called_once()
        self.cv2.destroyAllWindows.assert_called_once()

    def test_cleanup_failure_still_attempts_both_resources(self):
        self.cv2.waitKey.side_effect = [ord("q")]
        self.camera.release.side_effect = cv2.error("release failure")
        self.cv2.destroyAllWindows.side_effect = cv2.error("window cleanup failure")
        with patch.object(live_demo, "load_opencv", return_value=self.cv2), \
                patch.object(live_demo, "load_checkpoint", return_value=(object(), ["Glass"])), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "OpenCV resource cleanup failed: release failure"):
            self.run_demo()
        self.camera.release.assert_called_once()
        self.cv2.destroyAllWindows.assert_called_once()

    def test_timing_total_includes_file_and_other_overhead(self):
        self.cv2.waitKey.side_effect = [ord(" "), ord("q")]

        def predict(*args, timings, **kwargs):
            timings.update(file_read_conversion=0.1, preprocessing=0.2, inference=0.3)
            return self.result

        with patch.object(live_demo, "load_opencv", return_value=self.cv2), \
                patch.object(live_demo, "load_checkpoint", return_value=(object(), ["Glass"])), \
                patch.object(live_demo, "save_frame", return_value=Path("captures/latest.jpg")), \
                patch.object(live_demo, "predict_image", side_effect=predict), \
                patch.object(live_demo, "perf_counter", side_effect=[0, 0.1, 0.3, 1.0]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.run_demo(timing=True)
        self.assertIn("camera_read=100.000", output.getvalue())
        self.assertIn("file_write=200.000", output.getvalue())
        self.assertIn("other=100.000 total=1000.000", output.getvalue())

    def test_macos_permission_hint(self):
        with patch.object(camera.sys, "platform", "darwin"):
            message = str(camera.camera_error("Could not open camera 0"))
        self.assertIn("Privacy & Security > Camera", message)
        self.assertIn("terminal application", message)

    def test_single_capture_still_reads_once_and_releases(self):
        with patch.object(camera, "load_opencv", return_value=self.cv2), \
                patch.object(camera, "save_frame", return_value=Path("capture.jpg")):
            self.assertEqual(camera.capture_frame("capture.jpg"), Path("capture.jpg"))
        self.camera.read.assert_called_once()
        self.camera.release.assert_called_once()

    def test_predict_timing_preserves_fields_and_uses_cpu(self):
        model = Mock(return_value=torch.tensor([[0.0, 1.0]]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.jpg"
            Image.new("L", (256, 256)).save(path)
            timings = {}
            with patch("model.perf_counter", side_effect=[0, 0.1, 0.3, 0.6]):
                result = predict_image(model, ["Glass", "Paper"], path, timings=timings)
            self.assertEqual(set(result), {"label", "confidence", "disposal_category"})
            self.assertEqual(result["label"], "Paper")
            self.assertEqual(model.call_args.args[0].device.type, "cpu")
            self.assertAlmostEqual(timings["file_read_conversion"], 0.1)
            self.assertAlmostEqual(timings["preprocessing"], 0.2)
            self.assertAlmostEqual(timings["inference"], 0.3)
            with self.assertRaisesRegex(ValueError, "only on CPU"):
                predict_image(model, ["Glass", "Paper"], path, "mps", timings={})


if __name__ == "__main__":
    unittest.main()
