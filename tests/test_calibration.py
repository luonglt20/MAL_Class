import unittest

import numpy as np

from malware_hybrid.calibration import (
    DecisionPolicy, MultilabelSplitConformal, SplitConformalClassifier,
    choose_threshold_for_fpr, fit_multilabel_thresholds, wilson_interval,
)
from malware_hybrid.evidence import EvidenceMatch


class CalibrationTests(unittest.TestCase):
    def test_non_alert_is_unknown_and_does_not_suppress_family_classification(self):
        policy = DecisionPolicy(malware_threshold=.9997)
        for probability in (.01, .99):
            verdict = policy.malware_decision(probability)
            self.assertEqual(verdict["label"], "UNKNOWN")
            self.assertEqual(verdict["confidence"], 0.)
        family = policy.family_multilabel_decision(
            ["alpha"], np.array([.9]), malware_probability=.99
        )
        self.assertEqual(family["label"], "UNKNOWN")
        self.assertEqual(family["candidate"], "alpha")
        self.assertTrue(family["abstained"])
        self.assertEqual(family["verdict"], "unknown")

    def test_benign_requires_separate_bound_and_confidence_matches_label(self):
        policy = DecisionPolicy(malware_threshold=.9, malware_benign_threshold=.1)
        self.assertEqual(policy.malware_decision(.05)["label"], "BENIGN")
        self.assertAlmostEqual(policy.malware_decision(.05)["confidence"], .95)
        self.assertEqual(policy.malware_decision(.5)["label"], "UNKNOWN")
        self.assertEqual(policy.malware_decision(.95)["label"], "MALICIOUS")
        self.assertAlmostEqual(policy.malware_decision(.95)["confidence"], .95)
        with self.assertRaises(ValueError):
            DecisionPolicy(malware_threshold=.9, malware_benign_threshold=.95).malware_decision(.5)

    def test_high_model_score_without_chain_is_only_suspected(self):
        status, reasons = DecisionPolicy().behavior_status(
            "T1055", 0.99, EvidenceMatch("T1055", "Process Injection", "suspected", 0.5)
        )
        self.assertEqual(status, "suspected")
        self.assertIn("model_high_but_chain_incomplete", reasons)

    def test_complete_chain_requires_an_available_model_score_to_confirm(self):
        match = EvidenceMatch("T1055", "Process Injection", "confirmed", 1.0)
        policy = DecisionPolicy()
        status, reasons = policy.behavior_status(
            "T1055", 1.0, match, model_score_available=False,
        )
        self.assertEqual(status, "suspected")
        self.assertIn("behavior_model_score_unavailable", reasons)
        self.assertEqual(policy.behavior_status("T1055", 0.99, match)[0], "confirmed")
        self.assertEqual(policy.behavior_status("T1055", 0.8, match)[0], "suspected")

    def test_threshold_reports_finite_sample_uncertainty(self):
        result = choose_threshold_for_fpr(np.array([0.01, 0.1, 0.2, 0.8]), target_fpr=0.25)
        self.assertIn("wilson_95", result)
        self.assertGreaterEqual(result["threshold"], 0.5)
        self.assertTrue(result["target_met"])

    def test_threshold_reports_unattainable_empirical_fpr(self):
        result = choose_threshold_for_fpr(np.array([1.0, 1.0]), target_fpr=0.01)
        self.assertFalse(result["target_met"])
        self.assertEqual(result["fpr"], 1.0)

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

    def test_joint_conformal_abstains_on_ambiguous_out_of_distribution_scores(self):
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
        decision = DecisionPolicy().family_multilabel_decision(
            ["alpha", "beta"], np.array([.55, .56]), calibrator
        )
        self.assertEqual(decision["label"], "UNKNOWN")
        self.assertTrue(decision["abstained"])
        self.assertEqual(decision["confirmed_labels"], [])
        self.assertEqual(
            decision["uncertainty_method"],
            "simultaneous-multilabel-split-conformal",
        )

    def test_legacy_marginal_conformal_is_never_presented_as_simultaneous_confirmation(self):
        calibrator = MultilabelSplitConformal(
            ["alpha"],
            {"alpha": [.1] * 20},
            {"alpha": [.2]},
            .1,
        )
        decision = DecisionPolicy().family_multilabel_decision(
            ["alpha"], np.array([.9]), calibrator
        )
        self.assertEqual(decision["label"], "UNKNOWN")
        self.assertEqual(decision["candidate"], "alpha")
        self.assertEqual(decision["families"][0]["status"], "suspected")
        self.assertEqual(
            decision["families"][0]["reason"],
            "legacy_marginal_conformal_not_simultaneous",
        )
        self.assertEqual(
            decision["uncertainty_method"],
            "legacy-marginal-multilabel-split-conformal",
        )

    def test_multilabel_conformal_round_trip(self):
        original = MultilabelSplitConformal.fit(
            ["a"], np.asarray([[.2], [.8]]),
            np.asarray([[0], [1]]), np.ones((2, 1)), .1,
        )
        restored = MultilabelSplitConformal.from_dict(original.as_dict())
        self.assertEqual(restored.negative_scores, original.negative_scores)
        self.assertEqual(restored.positive_scores, original.positive_scores)
        self.assertEqual(restored.joint_scores, original.joint_scores)
        self.assertEqual(
            original.as_dict()["method"],
            "simultaneous-multilabel-split-conformal",
        )

    def test_legacy_mondrian_checkpoint_payload_remains_loadable(self):
        restored = MultilabelSplitConformal.from_dict({
            "method": "multilabel-mondrian-split-conformal",
            "labels": ["alpha"],
            "negative_scores": {"alpha": [.1]},
            "positive_scores": {"alpha": [.2]},
            "alpha": .1,
        })
        self.assertIsNone(restored.joint_scores)
        self.assertEqual(
            restored.as_dict()["method"],
            "legacy-marginal-multilabel-conformal",
        )

    def test_family_threshold_calibration_can_select_below_half_without_violating_fpr(self):
        probabilities = np.asarray([[.20], [.10], [.05], [.30], [.08], [.12]])
        targets = np.asarray([[1], [0], [0], [1], [0], [0]])
        result = fit_multilabel_thresholds(
            ["alpha"], probabilities, targets, np.ones_like(targets),
            target_fpr=.25, minimum_positives=2, minimum_negatives=3,
        )
        self.assertTrue(result["alpha"]["calibrated"])
        self.assertEqual(result["alpha"]["threshold"], .2)
        self.assertEqual(result["alpha"]["calibration_fpr"], 0.0)

    def test_calibrated_family_threshold_is_used_for_suspected_output(self):
        decision = DecisionPolicy().family_multilabel_decision(
            ["alpha", "beta"], np.asarray([.24, .19]),
            family_thresholds={"alpha": .2, "beta": .3},
        )
        self.assertEqual([item["label"] for item in decision["families"]], ["alpha"])
        self.assertEqual(decision["families"][0]["status"], "suspected")
        self.assertEqual(decision["families"][0]["reason"], "calibrated_threshold")


if __name__ == "__main__":
    unittest.main()
