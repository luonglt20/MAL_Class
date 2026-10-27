from pathlib import Path
from dataclasses import replace
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_ember_attention_baseline import _train_neural_baseline
from compare_winmet_protocol_baselines import validate_partitions
from compare_winmet_attention_baseline import _fingerprint, validate_reference_split
from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
from malware_hybrid.normalization import CapeReportParser


def test_partition_guard_rejects_changed_members_and_size():
    report = CapeReportParser().parse_file(Path(__file__).parent / "fixtures/cape_process_injection.json")
    partitions = {name: [replace(report, sample_hash=name)]
                  for name in ("train", "validation", "calibration", "withheld_internal_test")}
    experiment = {"split_strategy": "test", "partitions": {
        name: {"rows": 1, "sample_ids_sha256": _fingerprint(rows)} for name, rows in partitions.items()
    }}
    validate_partitions(experiment, partitions, "test")
    with pytest.raises(ValueError, match="strategy"):
        validate_partitions(experiment, partitions, "different")
    experiment["partitions"]["calibration"]["sample_ids_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="calibration"):
        validate_partitions(experiment, partitions, "test")


def test_partition_guard_rejects_overlap_even_if_manifest_fingerprints_match():
    report = CapeReportParser().parse_file(Path(__file__).parent / "fixtures/cape_process_injection.json")
    partitions = {"train": [report], "validation": [report]}
    experiment = {"split_strategy": "test", "partitions": {
        name: {"rows": 1, "sample_ids_sha256": _fingerprint(rows)} for name, rows in partitions.items()
    }}
    with pytest.raises(ValueError, match="overlapping"):
        validate_partitions(experiment, partitions, "test")


def test_schema_comparison_refuses_partition_drift_before_training():
    from copy import deepcopy
    reference = {"sample_cap": 1000, "sample_seed": 44, "split_seed": 44, "split_strategy": "test",
                 "partitions": {name: {"rows": 10, "sample_ids_sha256": name}
                                for name in ("train", "validation", "calibration", "withheld_internal_test")}}
    candidate = deepcopy(reference)
    candidate["source_sha256"] = "new features may have a different checksum"
    validate_reference_split(candidate, reference)
    candidate["partitions"]["withheld_internal_test"]["sample_ids_sha256"] = "changed"
    with pytest.raises(ValueError, match="withheld_internal_test"):
        validate_reference_split(candidate, reference)


@pytest.mark.parametrize("name", ["bag_of_tokens", "bilstm", "temporal_transformer", "graph_attention", "sequence_graph"])
def test_protocol_training_uses_matched_objective_and_calibration(name):
    report = CapeReportParser(max_events=8).parse_file(
        Path(__file__).parent / "fixtures/cape_process_injection.json"
    )
    tensorizer = ReportTensorizer.fit([report])
    config = HybridAttentionConfig(
        token_vocab_size=len(tensorizer.token_vocab), edge_vocab_size=len(tensorizer.edge_vocab),
        node_type_vocab_size=len(tensorizer.node_type_vocab), n_families=len(tensorizer.family_vocab),
        behavior_ids=tensorizer.behavior_ids, d_model=8, n_heads=2, temporal_layers=1,
        graph_layers=1, window_size=4, window_overlap=2, dropout=0., modality_dropout=0.,
    )
    original_loss = HybridAttentionModel.compute_loss
    observed = []

    def loss(*args, **kwargs):
        observed.append(kwargs.get("family_ranking_weight"))
        return original_loss(*args, **kwargs)

    with patch("malware_hybrid.baselines._MultiLabelBaseline.compute_loss", staticmethod(loss)):
        model, weights, thresholds, calibration, epoch, history = _train_neural_baseline(
            name, config, tensorizer, [report], [report], [report], epochs=1,
            batch_size=1, seed=44, device=torch.device("cpu"), family_ranking_weight=.5,
        )
    assert observed[0] == .5
    assert epoch == 1
    assert len(history) == 1
    assert model.behavior_head_trained is False
    assert thresholds == model.family_thresholds
    assert calibration
