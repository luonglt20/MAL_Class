import unittest
import copy
import tempfile
import pickle
from unittest.mock import patch
from pathlib import Path

import torch

from malware_hybrid.dataset import ReportTensorizer, collate_samples
from malware_hybrid.evaluation import ablate_report
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel, LabelwiseFamilyHead
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.training import load_checkpoint, save_checkpoint


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class _UnsafeCheckpointCallback:
    def __reduce__(self):
        return print, ("checkpoint callback must never execute",)


class ModelTests(unittest.TestCase):
    def test_checkpoint_loader_refuses_vulnerable_torch_versions_before_reading(self):
        with patch("torch.__version__", "2.5.1"), patch("torch.load") as loader:
            with self.assertRaisesRegex(ValueError, "PyTorch >= 2.6"):
                load_checkpoint("unused.pt")
            loader.assert_not_called()

    def test_checkpoint_loader_rejects_pickled_python_callbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "untrusted.pt"
            torch.save({"unexpected": _UnsafeCheckpointCallback()}, path)
            with patch("builtins.print") as callback:
                with self.assertRaises(pickle.UnpicklingError):
                    load_checkpoint(path)
                callback.assert_not_called()

    def test_auxiliary_ranking_excludes_benign_multilabel_and_unknown_rows(self):
        logits = torch.zeros(4, 3, requires_grad=True)
        output = {"family_logits": logits, "behavior_logits": torch.zeros(4, 1),
                  "malware_logits": torch.zeros(4)}
        batch = {
            "family_targets": torch.tensor([[1., 0., 0.], [0., 0., 0.], [1., 1., 0.], [0., 1., 0.]]),
            "family_target_mask": torch.tensor([[1., 1., 1.]] * 3 + [[0., 0., 0.]]),
            "behavior_targets": torch.zeros(4, 1), "behavior_target_mask": torch.zeros(4, 1),
            "malware_targets": torch.zeros(4), "malware_target_mask": torch.zeros(4),
        }
        default = HybridAttentionModel.compute_loss(output, batch)
        ranked = HybridAttentionModel.compute_loss(output, batch, family_ranking_weight=.5)
        torch.testing.assert_close(default["family_loss"], default["family_bce_loss"])
        gradient = torch.autograd.grad(ranked["family_ranking_loss"], logits)[0]
        self.assertGreater(float(gradient[0].abs().sum()), 0)
        self.assertEqual(float(gradient[1:].abs().sum()), 0)
        torch.testing.assert_close(ranked["family_loss"],
            ranked["family_bce_loss"] + .5 * ranked["family_ranking_loss"])
        with self.assertRaises(ValueError):
            HybridAttentionModel.compute_loss(output, batch, family_ranking_weight=-1)

    def test_family_head_checkpoint_layouts_roundtrip(self):
        report = CapeReportParser(max_events=64).parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        batch = collate_samples([tensorizer.encode(report)])
        with tempfile.TemporaryDirectory() as directory:
            for variant in ("legacy_shared", "labelwise"):
                with self.subTest(variant=variant):
                    config = HybridAttentionConfig(
                        token_vocab_size=len(tensorizer.token_vocab),
                        edge_vocab_size=len(tensorizer.edge_vocab),
                        node_type_vocab_size=len(tensorizer.node_type_vocab),
                        n_families=len(tensorizer.family_vocab),
                        behavior_ids=tensorizer.behavior_ids,
                        d_model=16, dropout=0., family_head_variant=variant,
                    )
                    model = HybridAttentionModel(config).eval()
                    model.family_training_priors = {label: .25 for label in tensorizer.family_vocab.token_to_id}
                    path = Path(directory) / f"{variant}.pt"
                    save_checkpoint(path, model, tensorizer)
                    if variant == "legacy_shared":
                        payload = torch.load(path, weights_only=False)
                        del payload["model_config"]["family_head_variant"]
                        torch.save(payload, path)
                    loaded, _ = load_checkpoint(path)
                    self.assertEqual(loaded.family_training_priors, model.family_training_priors)
                    self.assertFalse(loaded.malware_head_qualified)
                    loaded.eval()
                    with torch.no_grad():
                        torch.testing.assert_close(
                            model(batch)["family_logits"], loaded(batch)["family_logits"]
                        )

    def test_unqualified_malware_head_abstains_at_inference(self):
        report = CapeReportParser(max_events=64).parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        config = HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab),
            edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab),
            n_families=len(tensorizer.family_vocab),
            behavior_ids=tensorizer.behavior_ids,
            d_model=16,
            dropout=0.,
        )
        model = HybridAttentionModel(config)
        model.malware_head_trained = True
        model.malware_head_qualified = False
        model.malware_threshold = 0.0

        result = HybridInferencePipeline(model, tensorizer).predict(report)

        self.assertEqual(result["verdict"]["label"], "UNKNOWN")
        self.assertEqual(result["verdict"]["source"], "not_independently_qualified")
        self.assertEqual(result["verdict"]["reason"], "malware_head_did_not_pass_release_gate")

    def test_failed_checkpoint_write_preserves_previous_valid_file(self):
        report = CapeReportParser(max_events=64).parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        config = HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab),
            edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab),
            n_families=len(tensorizer.family_vocab),
            behavior_ids=tensorizer.behavior_ids,
            d_model=16,
        )
        model = HybridAttentionModel(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, model, tensorizer)
            original = path.read_bytes()
            with patch("malware_hybrid.training.torch.save", side_effect=OSError("interrupted")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    save_checkpoint(path, model, tensorizer)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])
            load_checkpoint(path)

    def test_labelwise_head_can_separate_identical_attention_contexts(self):
        head = LabelwiseFamilyHead(2, 4)
        context = torch.ones(1, 2, 4, requires_grad=True)
        with torch.no_grad():
            head.weight[0].fill_(1)
            head.weight[1].fill_(-1)
        logits = head(context).squeeze(-1)
        torch.testing.assert_close(logits, torch.tensor([[4., -4.]]))
        logits.sum().backward()
        self.assertGreater(float(context.grad.abs().sum()), 0)

    def test_modality_dropout_masks_raw_tokens_before_cross_attention(self):
        report = CapeReportParser(max_events=64).parse_file(FIXTURE)
        changed = copy.deepcopy(report)
        for event in changed.events:
            event.api = "UnrelatedSyntheticApi"
            event.timestamp += 12345.0
            event.target = "different-resource"
        tensorizer = ReportTensorizer.fit([report])
        original_batch = collate_samples([tensorizer.encode(report)])
        changed_batch = collate_samples([tensorizer.encode(changed)])
        config = HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab),
            edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab),
            n_families=len(tensorizer.family_vocab),
            behavior_ids=tensorizer.behavior_ids,
            d_model=16,
            n_heads=4,
            static_layers=1,
            temporal_layers=1,
            graph_layers=1,
            fusion_layers=1,
            window_size=64,
            window_overlap=32,
            dropout=0.0,
            modality_dropout=0.99,
            family_head_variant="labelwise",
        )
        model = HybridAttentionModel(config).train()
        # Force dropout on every available branch; the implementation must
        # preserve the first available (static) branch and mask raw dynamic
        # event/graph tokens, not only their pooled summaries.
        with patch("torch.rand_like", side_effect=lambda value: torch.zeros_like(value)):
            original_output = model(original_batch, return_attention=False)
            changed_output = model(changed_batch, return_attention=False)
        self.assertEqual(original_output["modality_mask"].tolist(), [[True, False, False]])
        for key in ("family_logits", "behavior_logits", "malware_logits"):
            torch.testing.assert_close(original_output[key], changed_output[key])

    def test_forward_exposes_every_attention_stage(self):
        report = CapeReportParser(max_events=64).parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        batch = collate_samples([tensorizer.encode(report)])
        self.assertEqual(batch["family_targets"].ndim, 2)
        self.assertEqual(float(batch["family_targets"].sum()), 1.0)
        config = HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab),
            edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab),
            n_families=len(tensorizer.family_vocab),
            behavior_ids=tensorizer.behavior_ids,
            d_model=32,
            n_heads=4,
            static_layers=1,
            temporal_layers=1,
            graph_layers=1,
            fusion_layers=1,
            window_size=3,
            window_overlap=1,
            dropout=0.0,
            modality_dropout=0.0,
            family_head_variant="labelwise",
        )
        model = HybridAttentionModel(config).eval()
        with torch.inference_mode():
            output = model(batch)
            logits_only = model(batch, return_attention=False)
        self.assertEqual(output["family_logits"].shape, (1, len(tensorizer.family_vocab)))
        self.assertEqual(output["behavior_logits"].shape, (1, len(tensorizer.behavior_ids)))
        self.assertEqual(output["attention"]["behavior_to_modality"].shape[-1], 3)
        self.assertEqual(
            output["attention"]["family_to_modality"].shape[-2],
            len(tensorizer.family_vocab),
        )
        self.assertEqual(output["attention"]["static_to_temporal"].shape[-1], len(report.events))
        self.assertEqual(output["attention"]["temporal"]["window_overlap"], 1)
        self.assertGreater(len(output["attention"]["temporal"]["window_starts"]), 1)
        self.assertIn("path_pool", output["attention"]["graph"])
        self.assertGreater(
            int(output["attention"]["graph"]["path_lengths"].max()),
            2,
        )
        self.assertTrue(torch.isfinite(output["family_logits"]).all())
        self.assertTrue(torch.isfinite(output["behavior_logits"]).all())
        self.assertNotIn("attention", logits_only)
        for key in ("family_logits", "behavior_logits", "malware_logits"):
            self.assertTrue(torch.allclose(output[key], logits_only[key], atol=1e-6, rtol=1e-5), key)

        # Backward must reach relation/time parameters and both modalities;
        # merely returning an attention tensor does not prove it is used.
        trained = model(batch)
        model.compute_loss(trained, batch)["loss"].backward()
        for parameter in (
            model.temporal_encoder.relation_weights,
            model.temporal_encoder.time_scale,
            model.static_encoder.value_projection[0].weight,
            model.graph_encoder.layers[0].edge_time.projection.weight,
            model.family_queries,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

        slower = copy.deepcopy(report)
        for event in slower.events:
            event.timestamp *= 1000
        slow_batch = collate_samples([tensorizer.encode(slower)])
        with torch.inference_mode():
            slow_output = model(slow_batch)
        self.assertFalse(torch.allclose(
            output["attention"]["temporal"]["pairwise_bias"],
            slow_output["attention"]["temporal"]["pairwise_bias"],
        ))

        prediction = HybridInferencePipeline(model, tensorizer).predict(report)
        alignment = prediction["attention_trace"]["static_dynamic_alignment"]
        self.assertTrue(alignment)
        self.assertIn("event_index", alignment[0])

        for mode in ("static_only", "dynamic_only"):
            partial = collate_samples([tensorizer.encode(ablate_report(report, mode))])
            with torch.inference_mode():
                partial_output = model(partial)
            self.assertTrue(torch.isfinite(partial_output["family_logits"]).all(), mode)
            self.assertTrue(torch.isfinite(partial_output["behavior_logits"]).all(), mode)


if __name__ == "__main__":
    unittest.main()
