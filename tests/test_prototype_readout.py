from pathlib import Path

import numpy as np
import pytest
import torch

from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.prototype_readout import (
    FrozenReadout, load_readout, prototype_scores, save_readout,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cape_process_injection.json"


def test_prototype_vote_prefers_nearest_family():
    base = np.array([[1., 0.], [0., 1.], [0.9, 0.1]])
    scores = prototype_scores(base, [[0], [1], [0]], np.array([[1., 0.05]]), 2, k=2)
    assert scores[0].argmax() == 0 and scores[0, 1] == 0


def test_sidecar_roundtrip_and_checksum_binding(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    config = {"readout": "prototype_knn2", "threshold": .5, "label_scope": "2 families"}
    save_readout(checkpoint, config, embeddings=np.eye(2), labels=[[0], [1, 0]])
    loaded = load_readout(checkpoint)
    assert loaded.name == "prototype_knn2" and loaded.labels == [[0], [1, 0]]
    with pytest.raises(FileExistsError):
        save_readout(checkpoint, config, embeddings=np.eye(2), labels=[[0], [1]])
    checkpoint.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        load_readout(checkpoint)
    assert load_readout(tmp_path / "missing.pt") is None


def test_pipeline_reports_frozen_top1_readout():
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
    assert "top1_readout" not in pipeline.predict(report)["family"]
    pipeline.top1_readout = FrozenReadout({"readout": "model_sigmoid", "threshold": 2.0})
    head = pipeline.predict(report)["family"]["top1_readout"]
    assert head["readout"] == "model_sigmoid" and head["accepted"] is False
    width = 3 * 16
    pipeline.top1_readout = FrozenReadout({"readout": "prototype_knn1", "threshold": 0.0},
                                          np.ones((1, width), dtype=np.float32), [[0]])
    knn = pipeline.predict(report)["family"]["top1_readout"]
    assert knn["label"] == tensorizer.family_vocab.decode(0) and knn["accepted"] is True
    assert len(model.malware_attention._forward_pre_hooks) == 0
