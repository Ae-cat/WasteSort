import csv
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "ml"))

from dataset import WasteDataset
from evaluate import calculate_metrics
from model import DISPOSAL_MAP, build_model, predict_image


class PipelineTest(unittest.TestCase):
    def test_dataset_root_override_and_rgb_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "Cardboard" / "sample.jpg"
            image_path.parent.mkdir()
            Image.new("L", (8, 8), 128).save(image_path)
            csv_path = root / "splits.csv"
            with csv_path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["path", "label", "split"])
                writer.writerow(["old/location/sample.jpg", "Cardboard", "train"])
            dataset = WasteDataset("train", csv_path=csv_path, data_root=root)
            image, label = dataset[0]
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(label, 0)

    def test_untrained_model_prediction_has_disposal_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            Image.new("RGB", (32, 32), "green").save(image_path)
            classes = list(DISPOSAL_MAP)
            result = predict_image(build_model(len(classes), pretrained=False), classes, image_path)
            self.assertIn(result["label"], classes)
            self.assertEqual(result["disposal_category"], DISPOSAL_MAP[result["label"]])
            self.assertGreaterEqual(result["confidence"], 0)
            self.assertLessEqual(result["confidence"], 1)

    def test_evaluation_metrics(self):
        targets = [0, 0, 1, 1]
        predictions = [0, 1, 1, 1]
        overall, per_class, matrix = calculate_metrics(
            targets, predictions, ["Cardboard", "Glass"]
        )
        self.assertEqual(overall, 0.75)
        self.assertEqual(per_class["Cardboard"], 0.5)
        self.assertEqual(per_class["Glass"], 1.0)
        self.assertEqual(matrix.tolist(), [[1, 1], [0, 2]])


if __name__ == "__main__":
    unittest.main()
