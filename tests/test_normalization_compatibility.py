from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
from malware_hybrid.normalization import CAPE_FEATURE_SCHEMA_VERSION, CapeReportParser
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.training import _normalization_versions, load_checkpoint, save_checkpoint


def candidate():
    report = CapeReportParser().parse_file(Path(__file__).parent / "fixtures/cape_process_injection.json")
    tensorizer = ReportTensorizer.fit([report])
    model = HybridAttentionModel(HybridAttentionConfig(
        token_vocab_size=len(tensorizer.token_vocab), edge_vocab_size=len(tensorizer.edge_vocab),
        node_type_vocab_size=len(tensorizer.node_type_vocab), n_families=len(tensorizer.family_vocab),
        behavior_ids=tensorizer.behavior_ids, d_model=8, n_heads=2, temporal_layers=1,
        graph_layers=1, static_layers=1, fusion_layers=1, window_size=4, window_overlap=2,
        dropout=0., modality_dropout=0.,
    ))
    return report, tensorizer, model


def test_normalization_provenance_distinguishes_static_from_legacy_unknown():
    report, _, _ = candidate()
    assert _normalization_versions([report]) == [CAPE_FEATURE_SCHEMA_VERSION]
    static = deepcopy(report)
    static.events = []
    assert _normalization_versions([static]) == []
    unknown = deepcopy(report)
    unknown.metadata.pop("normalization_version")
    assert _normalization_versions([report, unknown]) is None


@pytest.mark.parametrize("versions", [[CAPE_FEATURE_SCHEMA_VERSION], [], None])
def test_checkpoint_preserves_exact_normalization_provenance(tmp_path, versions):
    _, tensorizer, model = candidate()
    model.normalization_versions = versions
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, tensorizer)
    loaded, _ = load_checkpoint(checkpoint)
    assert loaded.normalization_versions == versions


@pytest.mark.parametrize("versions", [["cape-pe-locations-v2"], []])
def test_manual_checkpoint_mismatch_abstains_without_neural_forward(versions):
    report, tensorizer, model = candidate()
    model.normalization_versions = versions
    pipeline = HybridInferencePipeline(model, tensorizer)
    with patch.object(model, "forward", side_effect=AssertionError("must not score incompatible trace")):
        result = pipeline.predict(report)
    assert result["model_input_coverage"]["known_fraction"] == 1.
    assert result["model_input_coverage"]["normalization_compatible"] is False
    assert result["family"]["reason"] == "normalization_version_mismatch"
    assert result["attention_trace"]["mode"] == "evidence_only_normalization_rejected"
    assert all(item["status"] != "confirmed" for item in result["behaviors"])


def test_matching_schema_and_legacy_unknown_are_not_silently_relabelled():
    report, tensorizer, model = candidate()
    for versions, provenance in (([CAPE_FEATURE_SCHEMA_VERSION], "recorded"), (None, "unknown_legacy")):
        model.normalization_versions = versions
        result = HybridInferencePipeline(model, tensorizer).predict(report)
        assert result["model_input_coverage"]["normalization_compatible"] is True
        assert result["model_input_coverage"]["normalization_provenance"] == provenance
