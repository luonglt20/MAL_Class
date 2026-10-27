import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from malware_hybrid.dataset import ReportTensorizer, collate_samples
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel, LabelwiseFamilyHead
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.readout import family_attention_readout


class AnalyticReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(signature_branch_enabled=True, family_head_variant="labelwise")
        self.family_attention = torch.nn.MultiheadAttention(2, 1, batch_first=True)
        self.family_evidence_attention = torch.nn.MultiheadAttention(2, 1, batch_first=True)
        self.family_head = LabelwiseFamilyHead(1, 2)
        with torch.no_grad():
            for module in (self.family_attention, self.family_evidence_attention):
                module.in_proj_weight.zero_()
                module.in_proj_weight[4:].copy_(torch.eye(2))
                module.in_proj_bias.zero_()
                module.out_proj.weight.copy_(torch.eye(2))
                module.out_proj.bias.fill_(.2)
            self.family_head.weight.fill_(1.)
            self.family_head.bias.fill_(.3)

    def forward(self, batch, return_attention=True):
        query = torch.zeros(1, 1, 2)
        memory = torch.tensor([[[2., 2.], [-1., -1.]]])
        first, _ = self.family_attention(query, memory, memory, average_attn_weights=False)
        second, weights = self.family_evidence_attention(query, memory, memory, average_attn_weights=False)
        return {
            "family_logits": self.family_head(first + second).squeeze(-1),
            "attention": {"evidence_layout": {"modality": 0, "signature": 2, "static": 0, "temporal_events": 0, "graph": 0},
                          "family_to_evidence": weights},
        }


class ReadoutTests(unittest.TestCase):
    def test_equal_attention_has_opposite_signed_value_contributions(self):
        model = AnalyticReadout().train()
        result = family_attention_readout(model, {"event_ids": torch.ones(1, 1, dtype=torch.long)})
        torch.testing.assert_close(result["evidence_contributions"], torch.tensor([2., -1.]))
        self.assertAlmostEqual(result["bias"], 1.1, places=6)
        self.assertAlmostEqual(result["reconstructed_logit"], 3.1, places=6)
        self.assertLess(result["reconstruction_error"], 1e-6)
        self.assertTrue(model.training)

    def test_actual_hybrid_reconstructs_both_family_heads_and_relation_variants(self):
        report = CapeReportParser().parse_file(Path(__file__).parent / "fixtures" / "cape_process_injection.json")
        tensorizer = ReportTensorizer.fit([report])
        batch = collate_samples([tensorizer.encode(report)])
        for variant in ("labelwise", "legacy_shared"):
            for values in (False, True):
                with self.subTest(head=variant, relation_values=values):
                    config = HybridAttentionConfig(
                        token_vocab_size=len(tensorizer.token_vocab), edge_vocab_size=len(tensorizer.edge_vocab),
                        node_type_vocab_size=len(tensorizer.node_type_vocab), n_families=len(tensorizer.family_vocab),
                        behavior_ids=tensorizer.behavior_ids, d_model=16, n_heads=4,
                        temporal_layers=1, graph_layers=1, static_layers=1, fusion_layers=1,
                        window_size=4, window_overlap=2, dropout=0., modality_dropout=0.,
                        family_head_variant=variant, temporal_relation_values=values,
                    )
                    model = HybridAttentionModel(config).eval()
                    before = copy.deepcopy(model.state_dict())
                    original_batch = copy.deepcopy(batch)
                    result = family_attention_readout(model, batch)
                    self.assertLess(result["reconstruction_error"], 2e-6)
                    self.assertEqual(len(result["evidence_contributions"]), sum(result["evidence_layout"].values()))
                    self.assertFalse(model.training)
                    self.assertEqual(len(model.family_attention._forward_hooks), 0)
                    self.assertEqual(len(model.family_evidence_attention._forward_hooks), 0)
                    for key, parameter in before.items():
                        torch.testing.assert_close(parameter, model.state_dict()[key])
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(value, original_batch[key])

    def test_invalid_target_cleans_hooks_and_restores_training(self):
        model = AnalyticReadout().train()
        with self.assertRaises(ValueError):
            family_attention_readout(model, {"event_ids": torch.ones(1, 1)}, 10)
        self.assertTrue(model.training)
        self.assertEqual(len(model.family_attention._forward_hooks), 0)
        self.assertEqual(len(model.family_evidence_attention._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
