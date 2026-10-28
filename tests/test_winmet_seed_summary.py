from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_winmet_seeds import summarize_reports


def _report(seed: int, attention_map: float, bag_map: float) -> dict:
    return {
        "seed": seed,
        "sample_seed": 42,
        "split_seed": 42,
        "source_sha256": "source-sha",
        "sample_cap": 1000,
        "split_strategy": "group-random-archive-block-65-10-10-15",
        "family_min_support": 5,
        "family_ranking_weight": 0.5,
        "modality_dropout": 0.15,
        "modality_dropout_scope": "summary_only_legacy",
        "d_model": 32,
        "batch_size": 4,
        "epochs": 3,
        "family_head_variant": "labelwise",
        "partitions": {
            name: {"sample_ids_sha256": f"{name}-sha"}
            for name in ("train", "validation", "calibration", "withheld_internal_test")
        },
        "attention_validation": {
            "family_map": attention_map,
            "family_calibrated_macro_f1": 0.2,
            "family_top1_accuracy": 0.5,
            "family_top3_accuracy": 0.7,
        },
        "bag_of_tokens_validation": {
            "family_map": bag_map,
            "family_calibrated_macro_f1": 0.1,
            "family_top1_accuracy": 0.4,
            "family_top3_accuracy": 0.6,
        },
        "test_metrics": None,
    }


class WinmetSeedSummaryTests(unittest.TestCase):
    def test_rejects_mixing_new_attention_training_and_reused_checkpoints(self):
        first, second = _report(42, .3, .2), _report(43, .5, .4)
        second["attention_checkpoint_reused"] = True
        with self.assertRaisesRegex(ValueError, "model settings"):
            summarize_reports([first, second])

    def test_rejects_mixing_bias_only_and_relation_value_models(self):
        first, second = _report(42, 0.3, 0.2), _report(43, 0.5, 0.4)
        second["model_config"] = {"temporal_relation_values": True}
        with self.assertRaisesRegex(ValueError, "differ in source, split"):
            summarize_reports([first, second])

    def test_rejects_mixing_temporal_window_policies(self):
        first, second = _report(42, 0.3, 0.2), _report(43, 0.5, 0.4)
        second["model_config"] = {"temporal_window_policy": "per_sample_v1"}
        with self.assertRaisesRegex(ValueError, "differ in source, split"):
            summarize_reports([first, second])

    def test_rejects_mixing_historical_and_matched_bag_objectives(self):
        first, second = _report(42, 0.3, 0.2), _report(43, 0.5, 0.4)
        second["bag_family_ranking_weight"] = .5
        with self.assertRaisesRegex(ValueError, "differ in source, split"):
            summarize_reports([first, second])

    def test_rejects_changed_signature_static_attention_architecture(self):
        first, second = _report(42, 0.3, 0.2), _report(43, 0.5, 0.4)
        second["model_config"] = {"signature_static_attention_enabled": True}
        with self.assertRaisesRegex(ValueError, "differ in source, split"):
            summarize_reports([first, second])

    def test_rejects_mixing_signature_input_policies(self):
        first, second = _report(42, 0.3, 0.2), _report(43, 0.5, 0.4)
        second["tensorizer_config"] = {"yara_input_policy": "reviewed_structural_yara_v1"}
        with self.assertRaisesRegex(ValueError, "differ in source, split"):
            summarize_reports([first, second])

    def test_rejects_mixing_feature_budget_policies(self):
        first, second = _report(42, 0.3, 0.2), _report(43, 0.5, 0.4)
        second["tensorizer_config"] = {"evidence_selection_policy": "kind_balanced_v1"}
        with self.assertRaisesRegex(ValueError, "differ in source, split"):
            summarize_reports([first, second])

    def test_aggregates_matched_metrics_and_paired_wins(self):
        result = summarize_reports([_report(42, 0.3, 0.2), _report(43, 0.5, 0.4)])
        self.assertAlmostEqual(result["metrics"]["attention"]["family_map"]["mean"], 0.4)
        self.assertAlmostEqual(
            result["metrics"]["paired_delta_attention_minus_bag"]["family_map"]["mean"],
            0.1,
        )
        self.assertEqual(
            result["metrics"]["paired_delta_attention_minus_bag"]["family_map"]["attention_wins"],
            2,
        )
        self.assertIsNone(result["test_metrics"])

    def test_rejects_changed_split_fingerprints(self):
        first, second = _report(42, 0.3, 0.2), _report(43, 0.5, 0.4)
        second["partitions"]["validation"]["sample_ids_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "differ in source, split"):
            summarize_reports([first, second])

    def test_rejects_any_test_evaluation(self):
        first, second = _report(42, 0.3, 0.2), _report(43, 0.5, 0.4)
        second["test_metrics"] = {"family_map": 0.9}
        with self.assertRaisesRegex(ValueError, "test partition"):
            summarize_reports([first, second])

    def test_rejects_different_modality_dropout_scope(self):
        first, second = _report(42, 0.3, 0.2), _report(43, 0.5, 0.4)
        second["modality_dropout_scope"] = "pre_cross_attention_all_branch_tokens_v1"
        with self.assertRaisesRegex(ValueError, "model settings"):
            summarize_reports([first, second])


if __name__ == "__main__":
    unittest.main()
