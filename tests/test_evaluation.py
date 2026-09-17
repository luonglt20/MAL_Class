import unittest
from pathlib import Path

import numpy as np

from malware_hybrid.dataset import ReportTensorizer, collate_samples
from malware_hybrid.evaluation import (
    ablate_report,
    attention_integrated_gradients_agreement,
    fold_window_attention,
    integrated_gradients_event_scores,
)
from malware_hybrid.evidence import EvidenceEngine
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
from malware_hybrid.normalization import CapeReportParser


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class EvaluationTests(unittest.TestCase):
    def test_reversing_flow_invalidates_process_injection_chain(self):
        report = CapeReportParser().parse_file(FIXTURE)
        reversed_report = ablate_report(report, "reverse_events")
        match = EvidenceEngine().evaluate(reversed_report.events)["T1055"]
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
