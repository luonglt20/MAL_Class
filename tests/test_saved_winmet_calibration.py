from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stderr
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

from malware_hybrid.dataset import TensorizerConfig
from malware_hybrid.model import HybridAttentionConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_saved_winmet import validate_saved_candidate
from evaluate_winmet_flow_ablation import main as flow_main
from evaluate_winmet_flow_ablation import _paired_family_logit_change


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

    def test_flow_audit_rejects_changed_checkpoint_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data.jsonl"
            dataset.write_text("{}\n")
            Path(str(dataset) + ".manifest.json").write_text("{}")
            checkpoint = root / "candidate.pt"
            checkpoint.write_bytes(b"not the recorded checkpoint")
            reference = root / "experiment.json"
            reference.write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
            stderr = io.StringIO()
            argv = ["flow-audit", str(dataset), str(checkpoint), str(reference),
                    "--report", str(root / "result.json")]
            with patch.object(sys, "argv", argv), redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                flow_main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("checkpoint checksum differs", stderr.getvalue())
            self.assertFalse((root / "result.json").exists())

    def test_paired_scores_can_change_without_changing_top_family(self):
        class Model(torch.nn.Module):
            def forward(self, batch, return_attention=False):
                value = batch["resource_ids"].float().sum(1) * .01
                return {"family_logits": torch.stack([value + 10, value - 10], -1)}
        batch = {"resource_ids": torch.tensor([[1, 1, 0], [0, 0, 0]]),
                 "event_mask": torch.tensor([[True, True, False], [True, False, False]])}
        result = _paired_family_logit_change(Model(), [batch], torch.device("cpu"), "remove_resource_relations")
        self.assertGreater(result["maximum_absolute_logit_delta"], .01)
        self.assertEqual(result["argmax_changed_samples"], 0)
        self.assertEqual(result["events_with_known_resource_identity"], 2)
        self.assertEqual(result["valid_events"], 3)
        self.assertEqual(int(batch["resource_ids"].sum()), 2)


if __name__ == "__main__":
    unittest.main()
