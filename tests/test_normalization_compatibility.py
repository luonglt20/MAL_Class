from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
from malware_hybrid.normalization import CAPE_FEATURE_SCHEMA_VERSION, FILE_HANDLE_SCHEMA_VERSION, CapeReportParser
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


def test_file_resource_v4_is_rejected_by_a_v3_checkpoint_before_forward():
    report, tensorizer, model = candidate()
    raw = json.loads((Path(__file__).parent / "fixtures/cape_process_injection.json").read_text())
    report = CapeReportParser(file_handle_resolution=True).parse(raw)
    model.normalization_versions = [CAPE_FEATURE_SCHEMA_VERSION]
    with patch.object(model, "forward", side_effect=AssertionError("must not infer v4 with v3 weights")):
        result = HybridInferencePipeline(model, tensorizer).predict(report)
    assert result["family"]["reason"] == "normalization_version_mismatch"


def test_v4_checkpoint_parses_uploaded_cape_with_compatible_schema():
    from malware_hybrid.batch_analysis import analyze_uploaded_sample
    _, tensorizer, model = candidate()
    model.normalization_versions = [FILE_HANDLE_SCHEMA_VERSION]
    pipeline = HybridInferencePipeline(model, tensorizer)
    raw = (Path(__file__).parent / "fixtures/cape_process_injection.json").read_bytes()
    report, result = analyze_uploaded_sample("trace.json", raw, pipeline)
    assert report.metadata["normalization_version"] == FILE_HANDLE_SCHEMA_VERSION
    assert result["model_input_coverage"]["normalization_compatible"] is True
    assert result["model_input_coverage"]["supported"] is True
    # An explicitly supplied parser remains authoritative, and the normal
    # compatibility guard prevents accidental inference with a v3 trace.
    explicit = HybridInferencePipeline(model, tensorizer, parser=CapeReportParser())
    with patch.object(model, "forward", side_effect=AssertionError("must not infer v3 with v4 model")):
        _, rejected = analyze_uploaded_sample("trace.json", raw, explicit)
    assert rejected["family"]["reason"] == "normalization_version_mismatch"


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
