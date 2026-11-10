import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'ml'))
from threshold_sweep import policy_metrics


class ThresholdPolicyTest(unittest.TestCase):
    def test_compost_only_strict_boundary_and_denominators(self):
        rows = [(.59, 'Compost', 'Landfill'), (.6, 'Compost', 'Landfill'),
                (.4, 'Compost', 'Compost'), (.8, 'Compost', 'Compost'),
                (.2, 'Recyclable', 'Landfill'), (.9, 'Recyclable', 'Recyclable'),
                (.1, 'Landfill', 'Compost')]
        m = policy_metrics(rows, .6)
        self.assertEqual(m['flagged'], 2)
        self.assertEqual(m['wrong_compost_caught'], 1)
        self.assertEqual(m['wrong_compost_remaining'], 1)
        self.assertEqual(m['correct_compost_flagged'], 1)
        self.assertEqual(m['catch_rate'], .5)
        self.assertEqual(m['correct_compost_flag_rate'], .5)
        self.assertEqual(m['true_compost_total'], 3)
        self.assertEqual(m['true_compost_flagged'], 1)
        self.assertEqual(m['coverage'], 5/7)
        self.assertEqual(m['raw_bin_accuracy'], 3/7)
        self.assertEqual(m['accepted_bin_accuracy'], 2/5)

    def test_empty_denominators_are_not_perfect_accuracy(self):
        m = policy_metrics([], .6)
        self.assertIsNone(m['coverage'])
        self.assertIsNone(m['accepted_bin_accuracy'])
        self.assertIsNone(m['catch_rate'])
        self.assertIsNone(m['correct_compost_flag_rate'])
        self.assertIsNone(policy_metrics([(.1,'Compost','Compost')], .6)['accepted_bin_accuracy'])
