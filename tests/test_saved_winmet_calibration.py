from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

from malware_hybrid.dataset import TensorizerConfig
from malware_hybrid.model import HybridAttentionConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_saved_winmet import validate_saved_candidate


class SavedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(
            config=HybridAttentionConfig(token_vocab_size=8, edge_vocab_size=4,
                                        node_type_vocab_size=4, n_families=2, behavior_ids=["T1055"]),
            training_history=[{"epoch": 1, "family_map": .3}, {"epoch": 2, "family_map": .4}],
            training_selection_metric="family_map", selected_epoch=2,
        )
        self.tensorizer = SimpleNamespace(config=TensorizerConfig())
        self.reference = {"model_config": asdict(self.model.config),
                          "tensorizer_config": asdict(self.tensorizer.config)}

    def test_matching_saved_candidate_is_accepted(self):
        validate_saved_candidate(self.model, self.tensorizer, self.reference)

    def test_wrong_architecture_or_tensorizer_is_rejected(self):
        self.model.config.temporal_relation_values = True
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_saved_candidate(self.model, self.tensorizer, self.reference)
        self.model.config.temporal_relation_values = False
        self.tensorizer.config.max_events = 16
        with self.assertRaisesRegex(ValueError, "tensorizer"):
            validate_saved_candidate(self.model, self.tensorizer, self.reference)

    def test_missing_or_inconsistent_selection_provenance_is_rejected(self):
        for field, value in (("training_selection_metric", "loss"), ("training_history", []), ("selected_epoch", 1)):
            original = getattr(self.model, field)
            setattr(self.model, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_saved_candidate(self.model, self.tensorizer, self.reference)
            setattr(self.model, field, original)

    def test_nonfinite_or_missing_map_is_rejected(self):
        for value in (None, float("nan"), float("inf")):
            self.model.training_history[0]["family_map"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "nonfinite"):
                validate_saved_candidate(self.model, self.tensorizer, self.reference)


if __name__ == "__main__":
    unittest.main()
