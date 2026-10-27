import unittest
from pathlib import Path

import numpy as np
import torch

from malware_hybrid.dataset import ReportTensorizer, collate_samples
from malware_hybrid.evaluation import (
    ablate_report,
    ablate_encoded_batch,
    ENCODED_ABLATIONS,
    attention_integrated_gradients_agreement,
    fold_window_attention,
    integrated_gradients_event_scores,
)
from malware_hybrid.evidence import EvidenceEngine
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel, RelationalAttentionBlock, TemporalHierarchicalEncoder
from malware_hybrid.normalization import CapeReportParser


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class EvaluationTests(unittest.TestCase):
    def test_order_control_retains_real_timestamp_gaps_and_event_multiset(self):
        report = CapeReportParser().parse_file(FIXTURE)
        times = [0., .0001, .037, 3.14]
        for event, timestamp in zip(report.events, times):
            event.timestamp = timestamp
        shuffled = ablate_report(report, "shuffle_events_keep_timestamps", seed=42)
        self.assertEqual([event.timestamp for event in shuffled.events], times)
        self.assertEqual([event.timestamp for event in report.events], times)
        self.assertCountEqual([event.token for event in report.events], [event.token for event in shuffled.events])
        self.assertNotEqual([event.api for event in report.events], [event.api for event in shuffled.events])
        self.assertEqual([event.index for event in shuffled.events], list(range(4)))

    def test_temporal_windows_do_not_change_when_batch_padding_increases(self):
        torch.manual_seed(44)
        encoder = TemporalHierarchicalEncoder(32, 16, 4, 1, 0., 4, 2, 16, 16).eval()
        ids = torch.tensor([[2, 3, 4, 5]])
        times = torch.arange(4).float().unsqueeze(0)
        ones = torch.ones_like(ids)
        zeros = torch.zeros_like(ids)
        mask = ones.bool()
        resources = torch.tensor([[2, 2, 3, 3]])
        args = [ids, times, ones, zeros.float(), ones, zeros, ones, mask, resources]
        padded = [torch.nn.functional.pad(value, (0, 8)) for value in args]
        with torch.inference_mode():
            original = encoder(*args, return_attention=False)
            longer_batch = encoder(*padded, return_attention=False)
        torch.testing.assert_close(original[0], longer_batch[0][:, :4], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(original[2], longer_batch[2], atol=1e-6, rtol=1e-6)

    def test_per_sample_tail_window_is_stable_in_a_mixed_length_batch(self):
        torch.manual_seed(45)
        encoder = TemporalHierarchicalEncoder(32, 16, 4, 1, 0., 4, 2, 16, 16).eval()
        ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
        ones, zeros = torch.ones_like(ids), torch.zeros_like(ids)
        args = [ids, zeros.float(), ones, zeros.float(), ones, zeros, ones, ones.bool(), ones]
        mixed = []
        for value in args:
            padded = torch.nn.functional.pad(value, (0, 4))
            other = torch.ones_like(padded)
            mixed.append(torch.cat([padded, other], dim=0))
        with torch.inference_mode():
            alone = encoder(*args, return_attention=False)
            batched = encoder(*mixed, return_attention=False)
        torch.testing.assert_close(alone[0], batched[0][:1, :7], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(alone[2], batched[2][:1], atol=1e-6, rtol=1e-6)

    def test_saved_configs_do_not_relabel_legacy_window_semantics(self):
        values = dict(token_vocab_size=32, edge_vocab_size=8, node_type_vocab_size=8,
                      n_families=2, behavior_ids=["T1055"])
        self.assertEqual(HybridAttentionConfig(**values).temporal_window_policy, "per_sample_v1")
        old = HybridAttentionConfig.from_saved(values)
        self.assertEqual(old.temporal_window_policy, "legacy_batch_tail")
        old.d_model, old.n_heads = 16, 4
        self.assertEqual(HybridAttentionModel(old).temporal_encoder.window_policy, "legacy_batch_tail")

    def test_uniform_relation_bias_is_not_absolute_relation_evidence(self):
        # Softmax cancels row-constant shifts. This scopes a limitation of
        # bias-only relation encoding, not a claim about the whole hybrid.
        torch.manual_seed(44)
        block = RelationalAttentionBlock(16, 4, 0.).eval()
        x = torch.randn(1, 4, 16)
        mask = torch.tensor([[True, True, True, False]])
        zeros = torch.zeros(1, 4, 4, 4)
        with torch.inference_mode():
            unknown, weights = block(x, mask, zeros)
            known_uniform, shifted_weights = block(x, mask, zeros + .7)
        torch.testing.assert_close(unknown, known_uniform, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(weights, shifted_weights, atol=1e-6, rtol=1e-6)

    def test_node_preserving_edge_control_still_runs_graph_encoder(self):
        from unittest.mock import patch
        report = CapeReportParser().parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        batch = collate_samples([tensorizer.encode(report)])
        config = HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab), edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab), n_families=len(tensorizer.family_vocab),
            behavior_ids=tensorizer.behavior_ids, d_model=16, n_heads=4, temporal_layers=1,
            graph_layers=1, static_layers=1, fusion_layers=1, window_size=4, window_overlap=2,
            dropout=0., modality_dropout=0.,
        )
        model = HybridAttentionModel(config).eval()
        changed = ablate_encoded_batch(batch, "mask_graph_edges_keep_nodes")
        with patch.object(model.graph_encoder, "forward", wraps=model.graph_encoder.forward) as forward:
            with torch.inference_mode():
                output = model(changed, return_attention=False)
        self.assertEqual(forward.call_count, 1)
        self.assertTrue(torch.isfinite(output["family_logits"]).all())

    def test_encoded_controls_preserve_tokens_shapes_labels_and_original_batch(self):
        report = CapeReportParser().parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        batch = collate_samples([tensorizer.encode(report)])
        original = {key: value.clone() for key, value in batch.items() if isinstance(value, torch.Tensor)}
        expected_changes = {
            "remove_resource_relations": {"resource_ids"},
            "remove_process_context": {"process_ids", "parent_process_ids", "thread_ids"},
            "mask_graph_edges_keep_nodes": {"edge_mask"},
        }
        for mode in ENCODED_ABLATIONS:
            with self.subTest(mode=mode):
                changed = ablate_encoded_batch(batch, mode)
                for key, before in original.items():
                    self.assertEqual(changed[key].shape, before.shape)
                    self.assertEqual(changed[key].dtype, before.dtype)
                    torch.testing.assert_close(batch[key], before)
                    if key in expected_changes[mode]:
                        self.assertEqual(int(changed[key].count_nonzero()), 0)
                    else:
                        torch.testing.assert_close(changed[key], before)
                self.assertGreater(int(changed["graph_node_mask"].sum()), 0)

    def test_unknown_encoded_control_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown encoded ablation"):
            ablate_encoded_batch({}, "invented")

    def test_reversing_flow_invalidates_process_injection_chain(self):
        report = CapeReportParser().parse_file(FIXTURE)
        reversed_report = ablate_report(report, "reverse_events")
        match = EvidenceEngine().evaluate(reversed_report.events)["T1055"]
        self.assertNotEqual(match.status, "confirmed")

    def test_shuffling_flow_reassigns_time_and_breaks_original_event_order(self):
        report = CapeReportParser().parse_file(FIXTURE)
        original_apis = [event.api for event in report.events]
        shuffled = ablate_report(report, "shuffle_events", seed=42)
        shuffled_apis = [event.api for event in shuffled.events]

        self.assertNotEqual(shuffled_apis, original_apis)
        self.assertEqual(
            [event.timestamp for event in shuffled.events],
            [float(index) for index in range(len(shuffled.events))],
        )
        match = EvidenceEngine().evaluate(shuffled.events)["T1055"]
        self.assertNotEqual(match.status, "confirmed")

    def test_breaking_targets_invalidates_process_injection_chain(self):
        report = CapeReportParser().parse_file(FIXTURE)
        broken = ablate_report(report, "break_targets")
        match = EvidenceEngine().evaluate(broken.events)["T1055"]
        self.assertNotEqual(match.status, "confirmed")

    def test_overlap_attention_folds_to_event_axis(self):
        folded = fold_window_attention(
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            [0, 2],
            5,
        )
        np.testing.assert_allclose(folded, [1.0, 2.0, 3.5, 5.0, 6.0])

    def test_layer_integrated_gradients_and_rank_agreement(self):
        report = CapeReportParser().parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        batch = collate_samples([tensorizer.encode(report)])
        config = HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab),
            edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab),
            n_families=len(tensorizer.family_vocab),
            behavior_ids=tensorizer.behavior_ids,
            d_model=16, n_heads=4, static_layers=1, temporal_layers=1,
            graph_layers=1, fusion_layers=1, window_size=3,
            window_overlap=1, dropout=0.0, modality_dropout=0.0,
        )
        model = HybridAttentionModel(config).eval()
        scores = integrated_gradients_event_scores(
            model, batch, "family", 0, steps=4
        )
        self.assertEqual(scores.shape, tuple(batch["event_ids"].shape))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(scores.sum()), 0.0)
        self.assertAlmostEqual(
            attention_integrated_gradients_agreement(
                np.asarray([0.1, 0.4, 0.2]), np.asarray([1.0, 3.0, 2.0])
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
