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
                self.assertEqual(output["malware_logits"].shape, (1,))
                self.assertTrue(torch.isfinite(output["family_logits"]).all())
                self.assertTrue(torch.isfinite(output["behavior_logits"]).all())

    def test_bag_baseline_honors_labelwise_family_head_configuration(self):
        torch.manual_seed(123)
        config = HybridAttentionConfig(
            token_vocab_size=len(self.tensorizer.token_vocab),
            edge_vocab_size=len(self.tensorizer.edge_vocab),
            node_type_vocab_size=len(self.tensorizer.node_type_vocab),
            n_families=3,
            behavior_ids=self.tensorizer.behavior_ids,
            d_model=16, n_heads=4, static_layers=1, temporal_layers=1,
            graph_layers=1, fusion_layers=1, window_size=3,
            window_overlap=1, dropout=0.0, modality_dropout=0.0,
            family_head_variant="labelwise",
        )
        model = build_baseline("bag_of_tokens", config).eval()

        with torch.inference_mode():
            output = model(self.batch, return_attention=False)
            changed_values = dict(self.batch)
            changed_values["static_values"] = self.batch["static_values"] + 1.0
            changed_output = model(changed_values, return_attention=False)

        self.assertEqual(output["family_logits"].shape, (1, 3))
        self.assertEqual(model.family_head.weight.shape, (3, 16))
        self.assertFalse(torch.allclose(output["family_logits"], changed_output["family_logits"]))

    def test_bilstm_predictions_do_not_depend_on_batch_padding(self):
        torch.manual_seed(44)
        model = build_baseline("bilstm", self.config).eval()
        padded = dict(self.batch)
        padded["event_ids"] = torch.nn.functional.pad(self.batch["event_ids"], (0, 9))
        padded["event_mask"] = torch.nn.functional.pad(self.batch["event_mask"], (0, 9), value=False)
        with torch.inference_mode():
            short = model(self.batch)
            long = model(padded)
        for key in ("family_logits", "behavior_logits", "malware_logits"):
            torch.testing.assert_close(short[key], long[key], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
