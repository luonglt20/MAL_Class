import unittest
import copy
from pathlib import Path

import torch

from malware_hybrid.dataset import ReportTensorizer, collate_samples
from malware_hybrid.evaluation import ablate_report
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.pipeline import HybridInferencePipeline


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class ModelTests(unittest.TestCase):
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
        )
        model = HybridAttentionModel(config).eval()
        with torch.inference_mode():
            output = model(batch)
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
