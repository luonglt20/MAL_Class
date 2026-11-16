import copy
import tempfile
from pathlib import Path

import pytest

from malware_hybrid.dataset import ReportTensorizer, TensorizerConfig
from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.training import load_checkpoint, train_model

FIXTURE = Path(__file__).parent / 'fixtures' / 'cape_process_injection.json'


def _reports(n, prefix):
    base = CapeReportParser(max_events=64).parse_file(FIXTURE)
    rows = []
    for index in range(n):
        report = copy.deepcopy(base)
        report.sample_hash = f'{prefix}{index}'
        report.metadata['cape_signatures'] = ['injection_rwx'] + (['persistence_autorun'] if index % 2 else [])
        rows.append(report)
    unlabeled = copy.deepcopy(base)
    unlabeled.sample_hash = f'{prefix}none'
    rows.append(unlabeled)
    return rows


def test_signature_policy_labels_present_and_absent_names_and_masks_unlabeled_reports():
    config = TensorizerConfig(max_events=64, behavior_target_policy='cape_signatures_v1')
    reports = _reports(4, 't')
    tensorizer = ReportTensorizer.fit(reports, config, behavior_min_support=2)
    assert tensorizer.behavior_ids == ['injection_rwx', 'persistence_autorun']
    first, second, unlabeled = (tensorizer.encode(r) for r in (reports[0], reports[1], reports[-1]))
    assert first.behavior_targets == [1.0, 0.0] and first.behavior_mask == [1.0, 1.0]
    assert second.behavior_targets == [1.0, 1.0]
    assert unlabeled.behavior_mask == [0.0, 0.0]


def test_unknown_behavior_policy_is_rejected():
    with pytest.raises(ValueError):
        ReportTensorizer.fit(_reports(2, 'x'), TensorizerConfig(behavior_target_policy='other'))


def test_training_with_signature_targets_trains_and_scores_the_behavior_head():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'behavior.pt'
        result = train_model(
            _reports(6, 't'), _reports(4, 'v'), checkpoint, epochs=1, batch_size=2, d_model=16,
            window_size=8, window_overlap=4, family_min_support=1, behavior_min_support=2,
            train_behavior_head=True, behavior_target_policy='cape_signatures_v1')
        model, tensorizer = load_checkpoint(checkpoint)
    assert result['behavior_head_trained'] is True
    assert tensorizer.config.behavior_target_policy == 'cape_signatures_v1'
    assert result['history'][-1]['behavior_map'] is not None


def test_behavior_map_selection_needs_a_trained_behavior_head():
    with tempfile.TemporaryDirectory() as directory:
        result = train_model(
            _reports(6, 't'), _reports(4, 'v'), Path(directory) / 'b.pt', epochs=1, batch_size=2, d_model=16,
            window_size=8, window_overlap=4, family_min_support=1, behavior_min_support=2,
            train_behavior_head=True, behavior_target_policy='cape_signatures_v1', selection_metric='behavior_map')
        assert result['selected_epoch'] == 1
        with pytest.raises(ValueError, match='behavior_map selection'):
            train_model(_reports(6, 't'), _reports(4, 'v'), Path(directory) / 'c.pt', epochs=1, batch_size=2,
                        d_model=16, window_size=8, window_overlap=4, family_min_support=1, behavior_min_support=2,
                        train_behavior_head=False, behavior_target_policy='cape_signatures_v1',
                        selection_metric='behavior_map')


def test_encoding_cache_reproduces_training_exactly_and_is_reused():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        kwargs = dict(epochs=2, batch_size=2, d_model=16, window_size=8, window_overlap=4,
                      family_min_support=1, behavior_min_support=2, train_behavior_head=True,
                      behavior_target_policy='cape_signatures_v1', pre_encode=True, seed=7)
        plain = train_model(_reports(6, 't'), _reports(4, 'v'), root / 'a.pt', **kwargs)
        events = []
        first = train_model(_reports(6, 't'), _reports(4, 'v'), root / 'b.pt',
                            encoding_cache=root / 'cache', progress=events.append, **kwargs)
        second_events = []
        second = train_model(_reports(6, 't'), _reports(4, 'v'), root / 'c.pt',
                             encoding_cache=root / 'cache', progress=second_events.append, **kwargs)
        assert any(e.get('phase') == 'encoding_cache_saved' for e in events)
        assert any(e.get('phase') == 'encoding_cache_hit' for e in second_events)
        for run in (first, second):
            assert [h['loss'] for h in run['history']] == [h['loss'] for h in plain['history']]
        assert len(list((root / 'cache').glob('encoded_*.pkl'))) == 1
        with pytest.raises(ValueError, match='requires pre_encode'):
            train_model(_reports(6, 't'), _reports(4, 'v'), root / 'd.pt', **{**kwargs, 'pre_encode': False},
                        encoding_cache=root / 'cache')
