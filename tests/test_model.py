import unittest
from pathlib import Path

import torch

from malware_hybrid.dataset import ReportTensorizer, collate_samples
from malware_hybrid.evaluation import ablate_report
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
from malware_hybrid.normalization import CapeReportParser


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class ModelTests(unittest.TestCase):
    def test_forward_exposes_every_attention_stage(self):
        report = CapeReportParser(max_events=64).parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        batch = collate_samples([tensorizer.encode(report)])
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
            window_size=4,
            dropout=0.0,
            modality_dropout=0.0,
        )
        model = HybridAttentionModel(config).eval()
        with torch.inference_mode():
            output = model(batch)
        self.assertEqual(output["family_logits"].shape, (1, len(tensorizer.family_vocab)))
        self.assertEqual(output["behavior_logits"].shape, (1, len(tensorizer.behavior_ids)))
        self.assertEqual(output["attention"]["behavior_to_modality"].shape[-1], 3)
        self.assertTrue(torch.isfinite(output["family_logits"]).all())
        self.assertTrue(torch.isfinite(output["behavior_logits"]).all())

        for mode in ("static_only", "dynamic_only"):
            partial = collate_samples([tensorizer.encode(ablate_report(report, mode))])
            with torch.inference_mode():
                partial_output = model(partial)
            self.assertTrue(torch.isfinite(partial_output["family_logits"]).all(), mode)
            self.assertTrue(torch.isfinite(partial_output["behavior_logits"]).all(), mode)


if __name__ == "__main__":
    unittest.main()
