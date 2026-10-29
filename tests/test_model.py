import unittest
import copy
import tempfile
import pickle
from unittest.mock import patch
from pathlib import Path

import torch

from malware_hybrid.dataset import ReportTensorizer, TensorizerConfig, collate_samples
from malware_hybrid.evaluation import ablate_report
from malware_hybrid.model import (
    EdgeGraphAttentionLayer,
    HybridAttentionConfig,
    HybridAttentionModel,
    LabelwiseFamilyHead,
    TemporalGraphEncoder,
)
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.schema import AtomicEvidence
from malware_hybrid.training import load_checkpoint, save_checkpoint


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class _UnsafeCheckpointCallback:
    def __reduce__(self):
        return print, ("checkpoint callback must never execute",)


class ModelTests(unittest.TestCase):
    def test_vectorized_edge_attention_matches_grouped_reference_and_backpropagates(self):
        torch.manual_seed(73)
        layer = EdgeGraphAttentionLayer(d_model=8, n_heads=2, edge_vocab_size=5, dropout=0.0).eval()
        nodes = torch.randn(2, 4, 8, requires_grad=True)
        edge_index = torch.tensor([
            [[0, 1, 2, 0, 0], [2, 2, 1, 1, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ])
        edge_types = torch.tensor([[1, 2, 3, 1, 0], [0, 0, 0, 0, 0]])
        edge_times = torch.tensor([[1.0, 2.0, 3.0, 1.5, 0.0], [0.0] * 5])
        edge_success = torch.tensor([[1, 1, 0, 1, 0], [0] * 5])
        edge_mask = torch.tensor([[True, True, True, True, False], [False] * 5])

        actual, actual_attention = layer(
            nodes, edge_index, edge_types, edge_times, edge_success,
            edge_mask, return_attention=True,
        )

        batch, node_count, d_model = nodes.shape
        q = layer.query(nodes).view(batch, node_count, layer.n_heads, layer.head_dim)
        k = layer.key(nodes).view(batch, node_count, layer.n_heads, layer.head_dim)
        v = layer.value(nodes).view(batch, node_count, layer.n_heads, layer.head_dim)
        edge_context = (
            layer.edge_embedding(edge_types)
            + layer.edge_time(edge_times)
            + layer.edge_success(edge_success)
        ).view(batch, -1, layer.n_heads, layer.head_dim)
        expected_aggregated = torch.zeros_like(q)
        expected_attention = torch.zeros_like(actual_attention)
        for batch_index in range(batch):
            valid = torch.nonzero(edge_mask[batch_index], as_tuple=False).flatten()
            if not len(valid):
                continue
            sources = edge_index[batch_index, 0, valid]
            targets = edge_index[batch_index, 1, valid]
            scores = (
                q[batch_index, targets]
                * (k[batch_index, sources] + edge_context[batch_index, valid])
            ).sum(dim=-1) / (layer.head_dim ** 0.5)
            values = v[batch_index, sources] + edge_context[batch_index, valid]
            for target in torch.unique(targets):
                group = torch.nonzero(targets == target, as_tuple=False).flatten()
                weights = torch.softmax(scores[group], dim=0)
                expected_aggregated[batch_index, target] = (
                    weights.unsqueeze(-1) * values[group]
                ).sum(dim=0)
                expected_attention[batch_index, valid[group]] = weights
        expected = layer.norm(
            nodes + layer.dropout(layer.output(expected_aggregated.reshape(batch, node_count, d_model)))
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual_attention, expected_attention, atol=1e-6, rtol=1e-5)
        (actual.square().sum() + actual_attention.square().sum()).backward()
        for parameter in (layer.query.weight, layer.key.weight, layer.value.weight, layer.edge_embedding.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_vectorized_causal_paths_preserve_chains_masks_and_gradients(self):
        encoder = TemporalGraphEncoder(
            token_vocab_size=16, node_type_vocab_size=4, edge_vocab_size=4,
            d_model=8, n_heads=2, layers=1, dropout=0.0,
        )
        nodes = torch.randn(1, 4, 8, requires_grad=True)
        edge_index = torch.tensor([[[0, 1, 2], [1, 2, 3]]])
        edge_times = torch.tensor([[1.0, 2.0, 3.0]])
        edge_mask = torch.tensor([[True, True, True]])
        edge_causal = torch.tensor([[True, True, True]])

        tokens, mask, lengths, path_nodes = encoder._causal_paths(
            nodes, edge_index, edge_times, edge_mask, edge_causal
        )

        self.assertEqual(mask.tolist(), [[True, True, True]])
        self.assertEqual(lengths.tolist(), [[2, 3, 4]])
        self.assertEqual(path_nodes.tolist(), [[
            [0, 1, -1, -1, -1],
            [0, 1, 2, -1, -1],
            [0, 1, 2, 3, -1],
        ]])
        tokens.square().sum().backward()
        self.assertTrue(torch.isfinite(nodes.grad).all())
        self.assertGreater(float(nodes.grad.abs().sum()), 0)
        self.assertIsNotNone(encoder.path_projection[0].weight.grad)

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
        sample = tensorizer.encode(report)
        batch = collate_samples([sample])
        signature_tokens = [tensorizer.token_vocab.decode(index) for index in sample.signature_ids]
        static_tokens = [tensorizer.token_vocab.decode(index) for index in sample.static_ids]
        self.assertTrue(any(token.startswith("peid:") for token in signature_tokens))
        self.assertTrue(any(token.startswith("imphash:") for token in signature_tokens))
        self.assertTrue(any(token.startswith("import_dll:") for token in static_tokens))
        self.assertTrue(any(token.startswith("section_entropy:") for token in static_tokens))
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

    def test_legacy_merged_static_checkpoint_loads_without_three_branch_fields(self):
        report = CapeReportParser(max_events=64).parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit(
            [report], config=TensorizerConfig(signature_branch_enabled=False)
        )
        config = HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab),
            edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab),
            n_families=len(tensorizer.family_vocab),
            behavior_ids=tensorizer.behavior_ids,
            d_model=16,
            dropout=0.0,
            signature_branch_enabled=False,
        )
        model = HybridAttentionModel(config).eval()
        batch = collate_samples([tensorizer.encode(report)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            save_checkpoint(path, model, tensorizer)
            payload = torch.load(path, weights_only=False)
            payload["model_config"].pop("signature_branch_enabled")
            payload["tensorizer_config"].pop("signature_branch_enabled")
            payload["tensorizer_config"].pop("max_signature_tokens")
            payload["tensorizer_config"].pop("evidence_selection_policy")
            torch.save(payload, path)
            loaded, loaded_tensorizer = load_checkpoint(path)
        self.assertFalse(loaded.config.signature_branch_enabled)
        self.assertFalse(loaded_tensorizer.config.signature_branch_enabled)
        self.assertEqual(loaded_tensorizer.config.evidence_selection_policy, "prefix_legacy")
        loaded.eval()
        with torch.inference_mode():
            torch.testing.assert_close(
                model(batch)["family_logits"], loaded(batch)["family_logits"]
            )

    def test_earlier_three_branch_checkpoint_loads_without_signature_static_modules(self):
        report = CapeReportParser(max_events=64).parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        config = HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab),
            edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab),
            n_families=len(tensorizer.family_vocab),
            behavior_ids=tensorizer.behavior_ids,
            d_model=16,
            dropout=0.0,
            signature_static_attention_enabled=False,
        )
        model = HybridAttentionModel(config).eval()
        batch = collate_samples([tensorizer.encode(report)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "three_branch_v1.pt"
            save_checkpoint(path, model, tensorizer)
            payload = torch.load(path, weights_only=False)
            payload["model_config"].pop("signature_static_attention_enabled")
            torch.save(payload, path)
            loaded, _ = load_checkpoint(path)
        self.assertTrue(loaded.config.signature_branch_enabled)
        self.assertFalse(loaded.config.signature_static_attention_enabled)
        loaded.eval()
        with torch.inference_mode():
            expected = model(batch, return_attention=False)
            actual = loaded(batch, return_attention=False)
        for key in ("family_logits", "behavior_logits", "malware_logits"):
            torch.testing.assert_close(expected[key], actual[key])

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
        self.assertFalse(result["attention_trace"]["behavior_head_trained"])
        self.assertFalse(result["attention_trace"]["behavior_to_modality"])
        self.assertFalse(result["attention_trace"]["per_label_evidence"])
        self.assertTrue(result["attention_trace"]["family_to_modality"])

    def test_incompatible_input_abstains_even_with_a_qualified_high_score_head(self):
        report = CapeReportParser(max_events=64).parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        model = HybridAttentionModel(HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab),
            edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab),
            n_families=len(tensorizer.family_vocab),
            behavior_ids=tensorizer.behavior_ids,
            d_model=16, dropout=0.0,
        ))
        model.malware_head_trained = model.malware_head_qualified = True
        model.behavior_head_trained = True
        with torch.no_grad():
            model.malware_head.weight.zero_()
            model.malware_head.bias.fill_(20)
            model.family_head.weight.zero_()
            model.family_head.bias.fill_(20)
            model.behavior_head.weight.zero_()
            model.behavior_head.bias.fill_(20)
        pipeline = HybridInferencePipeline(model, tensorizer)
        supported = pipeline.predict(report)
        self.assertTrue(supported["model_input_coverage"]["supported"])
        self.assertEqual(supported["verdict"]["label"], "MALICIOUS")
        supported_behavior = next(b for b in supported["behaviors"] if b["attack"]["technique_id"] == "T1055")
        self.assertEqual(supported_behavior["status"], "confirmed")
        self.assertEqual(supported_behavior["confidence_source"], "model_score")

        unknown = copy.deepcopy(report)
        unknown.events.clear()
        unknown.graph = None
        unknown.atomic_evidence = [
            AtomicEvidence("import_dll", f"unseen_dll_{index}") for index in range(12)
        ]
        unknown.metadata.pop("fuzzy_hashes", None)
        with patch.object(model, "forward", side_effect=AssertionError("unsupported input reached model")):
            rejected = pipeline.predict(unknown)
        self.assertFalse(rejected["model_input_coverage"]["supported"])
        self.assertEqual(rejected["family"]["label"], "UNKNOWN")
        self.assertEqual(rejected["family"]["families"], [])
        self.assertIsNone(rejected["family"]["candidate"])
        self.assertEqual(rejected["verdict"]["label"], "UNKNOWN")
        self.assertEqual(rejected["verdict"]["source"], "input_vocabulary_coverage_guard")
        self.assertNotIn("family_to_modality", rejected["attention_trace"])

        unknown_dynamic = copy.deepcopy(report)
        unknown_dynamic.atomic_evidence = unknown.atomic_evidence * 30
        unknown_dynamic.metadata.pop("fuzzy_hashes", None)
        with patch.object(model, "forward", side_effect=AssertionError("unsupported dynamic input reached model")):
            rejected_dynamic = pipeline.predict(unknown_dynamic)
        rejected_behavior = next(b for b in rejected_dynamic["behaviors"] if b["attack"]["technique_id"] == "T1055")
        self.assertEqual(rejected_behavior["status"], "suspected")
        self.assertEqual(rejected_behavior["evidence_status"], "confirmed")
        self.assertEqual(rejected_behavior["confidence_source"], "evidence_chain_coverage")

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
        # preserve the first available (signature) branch and mask raw dynamic
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
            no_cross = model(batch, disable_cross_branch_attention=True)
        self.assertEqual(output["family_logits"].shape, (1, len(tensorizer.family_vocab)))
        self.assertEqual(output["behavior_logits"].shape, (1, len(tensorizer.behavior_ids)))
        self.assertEqual(output["attention"]["behavior_to_modality"].shape[-1], 3)
        self.assertEqual(
            output["attention"]["modality_order"], ["signature", "static", "dynamic"]
        )
        self.assertEqual(
            output["attention"]["family_to_modality"].shape[-2],
            len(tensorizer.family_vocab),
        )
        dynamic_token_count = (
            batch["event_ids"].shape[1] + batch["graph_node_ids"].shape[1]
        )
        self.assertEqual(output["attention"]["signature_to_dynamic"].shape[-1], dynamic_token_count)
        self.assertEqual(output["attention"]["static_to_dynamic"].shape[-1], dynamic_token_count)
        self.assertEqual(
            output["attention"]["signature_to_static"].shape[-2:],
            (batch["signature_ids"].shape[1], batch["static_ids"].shape[1]),
        )
        self.assertEqual(
            output["attention"]["static_to_signature"].shape[-2:],
            (batch["static_ids"].shape[1], batch["signature_ids"].shape[1]),
        )
        self.assertEqual(
            output["attention"]["dynamic_to_signature_static"].shape[-1],
            batch["signature_ids"].shape[1] + batch["static_ids"].shape[1],
        )
        self.assertEqual(
            output["attention"]["dynamic_to_signature_static"].shape[-2],
            dynamic_token_count,
        )
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
        self.assertEqual(
            int(no_cross["attention"]["signature_to_dynamic"].count_nonzero()), 0
        )
        self.assertEqual(
            int(no_cross["attention"]["dynamic_to_signature_static"].count_nonzero()), 0
        )
        self.assertEqual(
            int(no_cross["attention"]["signature_to_static"].count_nonzero()), 0
        )
        self.assertEqual(
            int(no_cross["attention"]["static_to_signature"].count_nonzero()), 0
        )
        static_only_batch = collate_samples([
            tensorizer.encode(ablate_report(report, "static_only"))
        ])
        with torch.inference_mode():
            with (
                patch.object(model.temporal_encoder, "forward", side_effect=AssertionError("empty temporal encoder ran")),
                patch.object(model.graph_encoder, "forward", side_effect=AssertionError("empty graph encoder ran")),
            ):
                static_full = model(static_only_batch, return_attention=False)
            static_attention = model(static_only_batch)
            static_no_cross = model(
                static_only_batch,
                return_attention=False,
                disable_cross_branch_attention=True,
            )
        for key in ("family_logits", "behavior_logits", "malware_logits"):
            torch.testing.assert_close(static_full[key], static_attention[key], atol=1e-6, rtol=1e-5)
        self.assertGreater(
            float(static_attention["attention"]["signature_to_static"].abs().sum()), 0
        )
        self.assertGreater(
            float(static_attention["attention"]["static_to_signature"].abs().sum()), 0
        )
        self.assertTrue(any(
            not torch.allclose(static_full[key], static_no_cross[key])
            for key in ("family_logits", "behavior_logits", "malware_logits")
        ))
        for key in ("family_logits", "behavior_logits", "malware_logits"):
            self.assertTrue(torch.allclose(output[key], logits_only[key], atol=1e-6, rtol=1e-5), key)

        model.zero_grad(set_to_none=True)
        static_training = model.train()(static_only_batch, return_attention=False)
        model.compute_loss(static_training, static_only_batch)["loss"].backward()
        for parameter in (
            model.signature_to_static.in_proj_weight,
            model.static_to_signature.in_proj_weight,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertIsNone(model.temporal_encoder.event_embedding.weight.grad)
        self.assertIsNone(model.graph_encoder.layers[0].edge_time.projection.weight.grad)
        model.eval().zero_grad(set_to_none=True)

        # Backward must reach relation/time parameters and both modalities;
        # merely returning an attention tensor does not prove it is used.
        trained = model(batch)
        model.compute_loss(trained, batch)["loss"].backward()
        for parameter in (
            model.temporal_encoder.relation_weights,
            model.temporal_encoder.time_scale,
            model.static_encoder.value_projection[0].weight,
            model.signature_encoder.value_projection[0].weight,
            model.static_to_dynamic.in_proj_weight,
            model.signature_to_dynamic.in_proj_weight,
            model.signature_to_static.in_proj_weight,
            model.static_to_signature.in_proj_weight,
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
        signature_alignment = prediction["attention_trace"]["signature_dynamic_alignment"]
        self.assertTrue(signature_alignment)
        self.assertIn("mutual_attention", signature_alignment[0])
        self.assertTrue(prediction["attention_trace"]["signature_static_alignment"])
        signature_items, static_items = tensorizer.evidence_groups(report)
        for link in prediction["attention_trace"]["signature_static_alignment"]:
            self.assertEqual(link["signature"], signature_items[link["signature_index"]].token)
            self.assertEqual(link["static"], static_items[link["static_index"]].token)
        for link in signature_alignment + alignment:
            event = report.events[link["event_index"]]
            for field in ("timestamp", "process_id", "parent_process_id", "thread_id"):
                self.assertEqual(link[field], getattr(event, field))
        self.assertTrue(prediction["attention_trace"]["cross_branch_evidence_links"])
        self.assertTrue(prediction["attention_trace"]["graph_dynamic_alignment"])
        for link in prediction["attention_trace"]["graph_dynamic_alignment"]:
            self.assertTrue(set(link["adjacent_event_indices"]).issubset(
                {event.index for event in report.events}))
            self.assertLessEqual(len(link["adjacent_event_indices"]), 16)
            self.assertGreaterEqual(link["adjacent_event_count"], len(link["adjacent_event_indices"]))
            if link["node_type"] == "event":
                own_index = int(link["node"].split(":")[1])
                self.assertIn(own_index, link["adjacent_event_indices"])
        for link in prediction["attention_trace"]["cross_branch_evidence_links"]:
            if link["branch"] == "signature_static":
                continue
            branch_items = tensorizer.evidence_groups(report)[0 if link["branch"] == "signature" else 1]
            self.assertEqual(link["evidence"], branch_items[link["feature_index"]].token)

        for mode in ("static_only", "dynamic_only"):
            partial = collate_samples([tensorizer.encode(ablate_report(report, mode))])
            with torch.inference_mode():
                partial_output = model(partial)
            self.assertTrue(torch.isfinite(partial_output["family_logits"]).all(), mode)
            self.assertTrue(torch.isfinite(partial_output["behavior_logits"]).all(), mode)


if __name__ == "__main__":
    unittest.main()
