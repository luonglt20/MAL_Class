import unittest

from malware_hybrid.metrics import (
    average_precision, balanced_accuracy, event_localization_f1,
    multilabel_f1, multilabel_metrics,
)


class MetricsTests(unittest.TestCase):
    def test_tied_scores_do_not_reward_input_order(self):
        self.assertEqual(average_precision([1, 0], [.5, .5]), .5)
        self.assertEqual(average_precision([0, 1], [.5, .5]), .5)
        self.assertIsNone(average_precision([0, 0], [.2, .8]))

    def test_unknown_targets_excluded_and_missing_positives_reported(self):
        result = multilabel_metrics(
            [[1, 0], [0, 0], [0, 0]], [[.9, .5], [.1, .2], [1., .8]],
            [[1, 1], [1, 1], [0, 1]], ["A", "B"],
        )
        self.assertEqual(result["behavior_map"], 1.)
        self.assertEqual(result["behavior_map_labels"], 1)
        self.assertIsNone(result["per_technique"]["B"]["recall"])

    def test_balanced_and_localization(self):
        self.assertEqual(balanced_accuracy([0, 0, 0, 1], [0, 0, 0, 0]), .5)
        self.assertEqual(event_localization_f1([1, 2], [2, 3])["f1"], .5)
        self.assertIsNone(event_localization_f1([], [])["f1"])

    def test_per_label_thresholds_are_applied_independently(self):
        result = multilabel_f1(
            [[1, 0], [0, 1]], [[.2, .1], [.1, .3]], [[1, 1], [1, 1]],
            threshold=[.15, .25],
        )
        self.assertEqual(result["micro_f1"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
