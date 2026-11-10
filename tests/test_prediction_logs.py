import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'ml'))
from model import DISPOSAL_MAP
from prediction_log import prediction_record, write_predictions
from test_images import external_samples, summarize


class PredictionLogTest(unittest.TestCase):
    def test_missing_labels_and_metrics_denominator(self):
        classes = list(DISPOSAL_MAP)
        records = [prediction_record('a.jpg', 'Metal', 'Metal', .8, 'external_unseen', 'm.pt', 'hash'),
                   prediction_record('b.jpg', '', 'Glass', .9, 'external_unseen', 'm.pt', 'hash'),
                   prediction_record('c.jpg', 'Plastic', 'Metal', .7, 'external_unseen', 'm.pt', 'hash')]
        summary = summarize(records, classes)
        self.assertEqual(summary['accuracy'], .5)
        self.assertEqual(summary['labeled_images'], 2)
        self.assertEqual(summary['disposal_accuracy'], 1)
        self.assertIsNone(summary['per_class']['Glass']['recall'])
        self.assertIsNone(summarize(records[1:2], classes)['accuracy'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'log.csv'
            write_predictions(path, records)
            with path.open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[1]['correct'], '')
            self.assertEqual(rows[0]['correct'], 'True')
            with self.assertRaises(FileExistsError):
                write_predictions(path, records)

    def test_external_provenance_and_copy_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / 'reference.jpg'
            reference.write_bytes(b'realwaste image')
            photo = root / 'photo.jpg'
            photo.write_bytes(reference.read_bytes())
            manifest = root / 'manifest.csv'
            manifest.write_text('image_path,true_label,source\nphoto.jpg,Metal,own camera\n')
            with self.assertRaisesRegex(ValueError, 'duplicates RealWaste'):
                external_samples(manifest, list(DISPOSAL_MAP), [reference])
            photo.write_bytes(b'independent image')
            self.assertEqual(len(external_samples(manifest, list(DISPOSAL_MAP), [reference])), 1)
            manifest.write_text('image_path,true_label,source\nphoto.jpg,Wrong,own camera\n')
            with self.assertRaisesRegex(ValueError, 'Unknown true label'):
                external_samples(manifest, list(DISPOSAL_MAP), [reference])
            manifest.write_text('image_path,true_label,source\nphoto.jpg,Metal,\n')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                external_samples(manifest, list(DISPOSAL_MAP), [reference])
            manifest.write_text('image_path,true_label,source\n')
            with self.assertRaisesRegex(ValueError, 'no images'):
                external_samples(manifest, list(DISPOSAL_MAP), [reference])
