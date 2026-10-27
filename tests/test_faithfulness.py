import copy
import unittest
from pathlib import Path

import numpy as np
import torch

from malware_hybrid.faithfulness import family_token_faithfulness


class KnownImportanceModel(torch.nn.Module):
    """Only the first signature token contributes to the family logit."""

    def __init__(self, misleading=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.2))
        self.misleading = misleading
        self.seen = []

    def forward(self, batch, return_attention=True):
        self.seen.append({key: value.clone() for key, value in batch.items()})
        value = batch["signature_ids"][:, 0].float() * self.scale
        result = {"family_logits": torch.stack([value, -value], dim=1)}
        if return_attention:
            weights = torch.ones(1, 1, 2, 12) * .01
            weights[:, :, :, 3 + (2 if self.misleading else 0)] = .9
            result["attention"] = {
                "evidence_layout": {"modality": 3, "signature": 3, "static": 2, "temporal_events": 2, "graph": 2},
                "family_to_evidence": weights,
            }
        return result


def example_batch():
    result = {}
    for name, values in [("signature", [5, 3, 2]), ("static", [4, 0]), ("event", [6, 7]), ("graph_node", [8, 9, 10])]:
        result[name + "_ids"] = torch.tensor([values])
        result[name + "_mask"] = result[name + "_ids"] != 0
    result["static_values"] = torch.tensor([[.9, 0.]])
    result["process_ids"] = torch.tensor([[2, 2]])
    result["edge_mask"] = torch.tensor([[True]])
    return result


class FaithfulnessTests(unittest.TestCase):
    def test_real_hybrid_contract_and_frozen_parameters(self):
        from malware_hybrid.dataset import ReportTensorizer, collate_samples
        from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
        from malware_hybrid.normalization import CapeReportParser

        report = CapeReportParser().parse_file(Path(__file__).parent / "fixtures" / "cape_process_injection.json")
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
        before = copy.deepcopy(model.state_dict())
        measured = family_token_faithfulness(model, batch, fractions=(.2,), random_repeats=1)
        self.assertEqual(set(measured["branches"]), {"signature", "static", "temporal_events", "graph"})
        self.assertTrue(np.isfinite(measured["baseline_probability"]))
        for branch in measured["branches"].values():
            if branch["evaluated"]:
                self.assertTrue(np.isfinite(branch["top_minus_random_aopc"]))
        for key, value in before.items():
            torch.testing.assert_close(model.state_dict()[key], value)
        signed = family_token_faithfulness(model, batch, fractions=(.2,), random_repeats=1, ranking="signed_readout")
        self.assertLess(signed["readout_reconstruction_error"], 2e-6)
        self.assertEqual(signed["target_index"], measured["target_index"])
        self.assertAlmostEqual(signed["baseline_probability"], measured["baseline_probability"])
        for branch in signed["branches"].values():
            if branch["evaluated"]:
                self.assertLessEqual(branch["positive_readout_tokens"], branch["valid_tokens"])

    def test_known_important_identity_beats_random_and_insertion_recovers(self):
        model = KnownImportanceModel().train()
        batch = example_batch()
        saved = copy.deepcopy(batch)
        parameters = copy.deepcopy(model.state_dict())
        report = family_token_faithfulness(model, batch, fractions=(.2, 1.), random_repeats=12)
        branch = report["branches"]["signature"]
        self.assertGreater(branch["top_minus_random_aopc"], 0)
        self.assertAlmostEqual(branch["top_insertion_probability"][0], report["baseline_probability"])
        self.assertAlmostEqual(branch["all_ranked_identities_zero_probability"], .5)
        self.assertAlmostEqual(branch["top_confidence_drop"][1], branch["random_mean_confidence_drop"][1])
        self.assertEqual(report["target_selection"], "original_top_logit")
        self.assertTrue(model.training)
        for key in batch:
            torch.testing.assert_close(batch[key], saved[key])
        for key in parameters:
            torch.testing.assert_close(model.state_dict()[key], parameters[key])
        self.assertEqual(branch["masked_counts"], [1, 3])
        self.assertEqual(report["branches"]["graph"]["valid_tokens"], 2)
        self.assertEqual(report["branches"]["graph"]["total_encoded_tokens"], 3)
        for seen in model.seen:
            for key in ("edge_mask", "process_ids", "static_values", "event_mask"):
                torch.testing.assert_close(seen[key], batch[key])

    def test_misleading_attention_is_not_forced_to_pass(self):
        report = family_token_faithfulness(KnownImportanceModel(True), example_batch(), fractions=(.2,), random_repeats=16)
        self.assertLess(report["branches"]["signature"]["top_minus_random_aopc"], 0)

    def test_seed_repeats_and_explicit_target(self):
        model = KnownImportanceModel().eval()
        first = family_token_faithfulness(model, example_batch(), target_index=0, random_repeats=2)
        second = family_token_faithfulness(model, example_batch(), target_index=0, random_repeats=2)
        self.assertEqual(first, second)
        self.assertEqual(first["target_selection"], "explicit")
        self.assertFalse(model.training)

    def test_empty_branch_and_invalid_arguments(self):
        batch = example_batch()
        batch["static_mask"].zero_()
        model = KnownImportanceModel()
        self.assertFalse(family_token_faithfulness(model, batch, random_repeats=1)["branches"]["static"]["evaluated"])
        for kwargs in ({"fractions": ()}, {"fractions": (np.nan,)}, {"fractions": (0.,)}, {"random_repeats": 0}, {"target_index": 2}, {"ranking": "invented"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                family_token_faithfulness(model, batch, **kwargs)
        self.assertTrue(model.training)
        batch["event_ids"] = batch["event_ids"].repeat(2, 1)
        with self.assertRaises(ValueError):
            family_token_faithfulness(model, batch)


if __name__ == "__main__":
    unittest.main()
