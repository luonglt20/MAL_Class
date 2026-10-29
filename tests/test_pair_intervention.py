import copy
import math
from pathlib import Path

import pytest
import torch

from malware_hybrid.dataset import ReportTensorizer
from malware_hybrid.graph import ExecutionGraphBuilder
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.pair_intervention import compare_attention_pairs, pair_intervention

FIXTURE = Path(__file__).parent / 'fixtures/cape_process_injection.json'


class ProductModel(torch.nn.Module):
    def __init__(self, event_token):
        super().__init__()
        self.event_token = event_token
        self.seen = []

    def forward(self, batch, return_attention=False):
        self.seen.append({key: value.clone() for key, value in batch.items() if isinstance(value, torch.Tensor)})
        feature = batch['signature_mask'][:, 0].float()
        event = ((batch['event_ids'] == self.event_token) & batch['event_mask']).any(1).float()
        logit = 2 * feature + 3 * event + 4 * feature * event
        return {'family_logits': torch.stack([logit, -logit], dim=1)}


def test_pair_removal_detects_known_interaction_and_preserves_report():
    report = CapeReportParser().parse_file(FIXTURE)
    tensorizer = ReportTensorizer.fit([report])
    token = tensorizer.encode(report).event_ids[0]
    model = ProductModel(token).train()
    before = copy.deepcopy(report)
    result = pair_intervention(model, tensorizer, report, branch='signature', feature_index=0, event_index=0)
    assert result['scores'] == {'full': 9., 'without_feature': 3., 'without_event': 2., 'without_both': 0.}
    assert result['interaction_logit'] == pytest.approx(4.)
    assert report == before and model.training
    assert model.seen[0]['event_mask'].sum() == 4
    assert model.seen[2]['event_mask'].sum() == 3
    assert model.seen[2]['edge_mask'].sum() < model.seen[0]['edge_mask'].sum()
    assert model.seen[1]['signature_ids'][0, 0] == 0
    assert model.seen[1]['signature_values'][0, 0] == 0
    assert not model.seen[1]['signature_mask'][0, 0]


def test_pair_removal_rejects_incompatible_graph_and_indices():
    report = CapeReportParser().parse_file(FIXTURE)
    tensorizer = ReportTensorizer.fit([report])
    model = ProductModel(tensorizer.encode(report).event_ids[0])
    for kwargs in ({'branch': 'graph', 'feature_index': 0, 'event_index': 0},
                   {'branch': 'signature', 'feature_index': -1, 'event_index': 0},
                   {'branch': 'signature', 'feature_index': 0, 'event_index': len(report.events)}):
        with pytest.raises(ValueError):
            pair_intervention(model, tensorizer, report, **kwargs)
    report.graph = ExecutionGraphBuilder().build(report.events)
    report.graph.edges.pop()
    with pytest.raises(ValueError, match='not reproducible'):
        pair_intervention(model, tensorizer, report, branch='signature', feature_index=0, event_index=0)


def test_pair_intervention_runs_on_hybrid_attention_model():
    from malware_hybrid.model import HybridAttentionConfig, HybridAttentionModel
    from malware_hybrid.pipeline import HybridInferencePipeline

    report = CapeReportParser().parse_file(FIXTURE)
    tensorizer = ReportTensorizer.fit([report])
    config = HybridAttentionConfig(
        token_vocab_size=len(tensorizer.token_vocab), edge_vocab_size=len(tensorizer.edge_vocab),
        node_type_vocab_size=len(tensorizer.node_type_vocab), n_families=len(tensorizer.family_vocab),
        behavior_ids=tensorizer.behavior_ids, d_model=16, n_heads=4, temporal_layers=1,
        graph_layers=1, static_layers=1, fusion_layers=1, window_size=4, window_overlap=2,
        dropout=0., modality_dropout=0.,
    )
    model = HybridAttentionModel(config).eval()
    result = pair_intervention(model, tensorizer, report, branch='static', feature_index=0, event_index=1)
    assert math.isfinite(result['interaction_logit'])
    assert all(math.isfinite(value) for value in result['scores'].values())
    assert not model.training
    pipeline = HybridInferencePipeline(model, tensorizer)
    first = compare_attention_pairs(pipeline, report, seed=42, random_controls=2)
    second = compare_attention_pairs(pipeline, report, seed=42, random_controls=2)
    assert first == second
    for branch in ('signature', 'static'):
        assert first['branches'][branch]['evaluated']
        assert math.isfinite(first['branches'][branch]['top_minus_random_absolute_interaction'])
