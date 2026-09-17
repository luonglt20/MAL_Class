import unittest

import numpy as np

from malware_hybrid.calibration import (
    DecisionPolicy, MultilabelSplitConformal, SplitConformalClassifier,
    choose_threshold_for_fpr, wilson_interval,
)
from malware_hybrid.evidence import EvidenceMatch


class CalibrationTests(unittest.TestCase):
    def test_high_model_score_without_chain_is_only_suspected(self):
        status, reasons = DecisionPolicy().behavior_status(
            "T1055", 0.99, EvidenceMatch("T1055", "Process Injection", "suspected", 0.5)
        )
        self.assertEqual(status, "suspected")
        self.assertIn("model_high_but_chain_incomplete", reasons)

    def test_threshold_reports_finite_sample_uncertainty(self):
        result = choose_threshold_for_fpr(np.array([0.01, 0.1, 0.2, 0.8]), target_fpr=0.25)
        self.assertIn("wilson_95", result)
        self.assertGreaterEqual(result["threshold"], 0.5)

    def test_split_conformal_abstains_on_ambiguous_set(self):
        calibrator = SplitConformalClassifier.fit(
            ["alpha", "beta"],
            np.array([[.5, .5], [.5, .5], [.5, .5], [.5, .5]]),
            np.array([0, 0, 1, 1]),
            alpha=.2,
        )
        decision = DecisionPolicy(family_abstain_threshold=.0).family_decision(
            ["alpha", "beta"], np.array([.5, .5]), calibrator
        )
        self.assertTrue(decision["abstained"])
        self.assertEqual(set(decision["conformal"]["prediction_set"]), {"alpha", "beta"})

    def test_conformal_checkpoint_payload_round_trip(self):
        calibrator = SplitConformalClassifier(["a", "b"], [.1, .2, .3], .1)
        restored = SplitConformalClassifier.from_dict(calibrator.as_dict())
        self.assertEqual(restored.labels, calibrator.labels)
        self.assertEqual(restored.calibration_scores, calibrator.calibration_scores)

    def test_multilabel_conformal_can_confirm_multiple_families(self):
        calibrator = MultilabelSplitConformal.fit(
            ["alpha", "beta"],
            np.array([
                [.9, .8], [.8, .9], [.85, .85], [.9, .9],
                [.1, .2], [.2, .1], [.15, .15], [.1, .1],
            ]),
            np.array([
                [1, 1], [1, 1], [1, 1], [1, 1],
                [0, 0], [0, 0], [0, 0], [0, 0],
            ]),
            np.ones((8, 2)), alpha=.2,
        )
        decision = DecisionPolicy(family_abstain_threshold=.5).family_multilabel_decision(
            ["alpha", "beta"], np.array([.95, .95]), calibrator
        )
        self.assertEqual(set(decision["confirmed_labels"]), {"alpha", "beta"})
        self.assertFalse(decision["abstained"])

    def test_multilabel_conformal_round_trip(self):
        original = MultilabelSplitConformal(
            ["a"], {"a": [.1, .2]}, {"a": [.3]}, .1
        )
        restored = MultilabelSplitConformal.from_dict(original.as_dict())
        self.assertEqual(restored.negative_scores, original.negative_scores)
        self.assertEqual(restored.positive_scores, original.positive_scores)


if __name__ == "__main__":
    unittest.main()
