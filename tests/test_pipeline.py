import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch

from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.dataset import ReportTensorizer, collate_samples
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.readout import family_attention_readout


FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


class PipelineTests(unittest.TestCase):
    def test_signed_readout_matches_candidate_and_complete_logit_without_extra_forward(self):
        report = CapeReportParser().parse_file(FIXTURE)
        tensorizer = ReportTensorizer.fit([report])
        model = HybridAttentionModel(HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab), edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab), n_families=len(tensorizer.family_vocab),
            behavior_ids=tensorizer.behavior_ids, d_model=16, n_heads=4, static_layers=1,
            temporal_layers=1, graph_layers=1, fusion_layers=1, window_size=8,
            window_overlap=4, dropout=0., modality_dropout=0., family_head_variant="labelwise",
        )).eval()
        pipeline = HybridInferencePipeline(model, tensorizer)
        with torch.inference_mode():
            expected_logits = model(collate_samples([tensorizer.encode(report)]))["family_logits"][0]
        with patch.object(model, "forward", wraps=model.forward) as forward:
            result = pipeline.predict(report)
        self.assertEqual(forward.call_count, 1)
        readout = result["attention_trace"]["family_readout"]
        target = int(expected_logits.argmax())
        self.assertEqual(readout["target_family"], tensorizer.family_vocab.decode(target))
        self.assertAlmostEqual(readout["family_logit"], float(expected_logits[target]), places=5)
        total = readout["bias"] + sum(readout["modality_contributions"].values()) + sum(
            readout["evidence_group_contributions"].values()
        )
        self.assertAlmostEqual(total, readout["family_logit"], places=5)
        self.assertLess(readout["reconstruction_error"], 1e-5)
        self.assertTrue(all(row["logit_contribution"] > 0 for row in readout["positive_evidence"]))
        self.assertTrue(all(row["logit_contribution"] < 0 for row in readout["negative_evidence"]))
        self.assertTrue(all("<PAD>" not in row["evidence"] for row in
                            readout["positive_evidence"] + readout["negative_evidence"]))
        self.assertEqual(result["family"]["label"], "UNKNOWN")
        json.dumps(result, allow_nan=False)
        self.assertEqual(len(model.family_attention._forward_hooks), 0)
        # An invalid explanation must neither be displayed nor alter decisions.
        def invalid_readout(*args, **kwargs):
            value = family_attention_readout(*args, **kwargs)
            value["reconstruction_error"] = .1
            return value

        with patch("malware_hybrid.pipeline.family_attention_readout", side_effect=invalid_readout):
            invalid = pipeline.predict(report)
        self.assertIsNone(invalid["attention_trace"]["family_readout"])
        self.assertEqual(invalid["family"], result["family"])
        self.assertEqual(invalid["attention_trace"]["family_readout_notice"],
                         "unavailable_logit_reconstruction_failed")
        # Schema rejection must happen before either inference or explanation.
        model.normalization_versions = ["another-schema"]
        with patch.object(model, "forward", side_effect=AssertionError("must not infer")):
            rejected = pipeline.predict(report)
        self.assertFalse(rejected["model_input_coverage"]["supported"])
        self.assertNotIn("family_readout", rejected["attention_trace"])

    def test_evidence_only_pipeline_maps_frameworks(self):
        result = HybridInferencePipeline().predict_file(FIXTURE)
        behavior = next(item for item in result["behaviors"] if item["attack"]["technique_id"] == "T1055")
        self.assertEqual(behavior["confidence_source"], "evidence_chain_coverage")
        self.assertEqual(behavior["status"], "suspected")
        self.assertEqual(behavior["evidence_status"], "confirmed")
        self.assertIn("behavior_model_score_unavailable", behavior["reasons"])
        self.assertIn("SI-4", behavior["nist"]["sp_800_53_r5"])
        self.assertEqual(len(behavior["evidence_chain"]), 4)
        self.assertEqual(result["family"]["label"], "UNKNOWN")
        self.assertEqual(result["family"]["source"], "no_model_checkpoint")
        layer = result["attack_navigator_layer"]
        self.assertEqual(layer["versions"]["layer"], "4.5")
        self.assertIn("T1055", [item["techniqueID"] for item in layer["techniques"]])


if __name__ == "__main__":
    unittest.main()
