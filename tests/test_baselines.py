import unittest
from pathlib import Path

import torch

from malware_hybrid.baselines import BASELINE_NAMES, IndicatorRuleBaseline, build_baseline
from malware_hybrid.dataset import ReportTensorizer, collate_samples
from malware_hybrid.model import HybridAttentionConfig
from malware_hybrid.normalization import CapeReportParser


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.report = CapeReportParser(max_events=64).parse_file(FIXTURE)
        self.tensorizer = ReportTensorizer.fit([self.report])
        self.batch = collate_samples([self.tensorizer.encode(self.report)])
        self.config = HybridAttentionConfig(
            token_vocab_size=len(self.tensorizer.token_vocab),
            edge_vocab_size=len(self.tensorizer.edge_vocab),
            node_type_vocab_size=len(self.tensorizer.node_type_vocab),
            n_families=len(self.tensorizer.family_vocab),
            behavior_ids=self.tensorizer.behavior_ids,
            d_model=16, n_heads=4, static_layers=1, temporal_layers=1,
            graph_layers=1, fusion_layers=1, window_size=3,
            window_overlap=1, dropout=0.0, modality_dropout=0.0,
        )

    def test_all_seven_protocol_baselines_are_registered(self):
        self.assertEqual(len(BASELINE_NAMES), 7)
        self.assertEqual(len(set(BASELINE_NAMES)), 7)

    def test_rule_baseline_refuses_family_inference(self):
        output = IndicatorRuleBaseline().predict(self.report)
        self.assertIsNone(output.family_scores)
        self.assertEqual(output.evidence_status["T1055"], "confirmed")

    def test_neural_baselines_share_multilabel_output_contract(self):
        for name in BASELINE_NAMES[1:]:
            with self.subTest(name=name):
                model = build_baseline(name, self.config).eval()
                with torch.inference_mode():
                    output = model(self.batch, return_attention=False)
                self.assertEqual(output["family_logits"].shape, (1, 1))
                self.assertEqual(
                    output["behavior_logits"].shape,
                    (1, len(self.tensorizer.behavior_ids)),
                )
                self.assertTrue(torch.isfinite(output["family_logits"]).all())
                self.assertTrue(torch.isfinite(output["behavior_logits"]).all())


if __name__ == "__main__":
    unittest.main()
