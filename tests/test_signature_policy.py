from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from malware_hybrid.dataset import ReportTensorizer, TensorizerConfig
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.schema import AtomicEvidence, ParsedReport
from malware_hybrid.signature_policy import (
    LEGACY_YARA_INPUT_POLICY, YARA_INPUT_POLICY, structural_yara_evidence,
)
from malware_hybrid.training import evaluate, load_checkpoint, save_checkpoint


class SignaturePolicyTests(unittest.TestCase):
    def test_excludes_whole_label_rule_without_ground_truth_or_match_count(self):
        for rule in (
            {"name": "redline_rule", "tags": ["packer"]},
            {"name": "unknown_family", "tags": ["structural"]},
            {"name": "UPX", "tags": ["packer", "agenttesla"]},
            {"name": "UPX", "category": "T1055"},
            {"name": "UPX", "meta": {"family": "redline"}},
            {"name": "UPX", "meta": {"description": "malware verdict"}},
            "T1055_process_injection",
        ):
            with self.subTest(rule=rule):
                report = CapeReportParser().parse({"target": {"file": {"yara": [rule]}}})
                self.assertFalse(any(item.kind.startswith("yara") for item in report.atomic_evidence))

    def test_retains_reviewed_structural_rule_in_signature_branch(self):
        evidence = structural_yara_evidence({"name": "UPX", "tags": ["packer", "pe"]})
        self.assertEqual([item.token for item in evidence], ["yara_category:packer:upx"])
        self.assertEqual(evidence[0].metadata["input_policy"], YARA_INPUT_POLICY)
        report = ParsedReport("structural", family="test", events=[], atomic_evidence=evidence)
        tensorizer = ReportTensorizer.fit([report])
        signatures, static = tensorizer.evidence_groups(report)
        self.assertEqual([item.token for item in signatures], ["yara_category:packer:upx"])
        self.assertFalse(static)
        self.assertTrue(tensorizer.encode(report).signature_ids)

    def test_cached_unverified_yara_cannot_enter_vocabulary_or_encoding(self):
        report = ParsedReport("cached", family="test", events=[], atomic_evidence=[
            AtomicEvidence("yara", "matched"),
            AtomicEvidence("YARA_CATEGORY", "packer"),
            AtomicEvidence("yara_category", "redline"),
            AtomicEvidence("section_entropy", ".text", 7.9),
        ])
        tensorizer = ReportTensorizer.fit([report])
        self.assertFalse(any(token.startswith("yara") for token in tensorizer.token_vocab.token_to_id))
        self.assertEqual(tensorizer.encode(report).signature_ids, [])
        self.assertEqual(len(tensorizer.encode(report).static_ids), 1)

    def test_legacy_yara_checkpoint_abstains_and_cannot_be_evaluated_as_clean(self):
        report = ParsedReport("legacy", family="test", events=[], atomic_evidence=[AtomicEvidence("file_size", "b2")])
        tensorizer = ReportTensorizer.fit([report])
        tensorizer.token_vocab.add("yara:matched")
        model = HybridAttentionModel(HybridAttentionConfig(
            token_vocab_size=len(tensorizer.token_vocab),
            edge_vocab_size=len(tensorizer.edge_vocab),
            node_type_vocab_size=len(tensorizer.node_type_vocab),
            n_families=len(tensorizer.family_vocab), behavior_ids=tensorizer.behavior_ids,
            d_model=16, dropout=0.0,
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            save_checkpoint(path, model, tensorizer)
            payload = torch.load(path, weights_only=True)
            payload["tensorizer_config"].pop("yara_input_policy")
            torch.save(payload, path)
            loaded, loaded_tensorizer = load_checkpoint(path)
        self.assertFalse(loaded_tensorizer.signature_policy_compatible)
        with patch.object(loaded, "forward", side_effect=AssertionError("legacy model was invoked")):
            result = HybridInferencePipeline(loaded, loaded_tensorizer).predict(report)
        self.assertEqual(result["family"]["reason"], "legacy_yara_checkpoint_requires_retraining")
        self.assertFalse(result["model_input_coverage"]["supported"])
        with self.assertRaisesRegex(ValueError, "requires retraining"):
            evaluate(loaded, [], torch.device("cpu"))

    def test_legacy_static_checkpoint_without_yara_remains_compatible(self):
        report = ParsedReport("static", family=None, events=[], atomic_evidence=[AtomicEvidence("file_size", "b2")])
        tensorizer = ReportTensorizer.fit([report], config=TensorizerConfig(
            yara_input_policy=LEGACY_YARA_INPUT_POLICY,
        ))
        self.assertTrue(tensorizer.signature_policy_compatible)
