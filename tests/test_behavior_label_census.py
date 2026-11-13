import importlib.util
from pathlib import Path

from malware_hybrid.schema import ExecutionEvent, ParsedReport

SPEC = importlib.util.spec_from_file_location(
    'behavior_label_census', Path(__file__).parents[1] / 'scripts' / 'behavior_label_census.py')
census_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(census_module)


def _event(i, api, target='pid:4242'):
    return ExecutionEvent(i, float(i), '100', '1', '1', api, 'process', True, '0', 'pid:100', target, 'process')


def test_census_counts_rule_status_and_weak_agreement():
    injected = ParsedReport('a', family='x', atomic_evidence=[], weak_labels=['T1055', 'T1999'], events=[
        _event(0, 'OpenProcess'), _event(1, 'VirtualAllocEx'),
        _event(2, 'WriteProcessMemory'), _event(3, 'CreateRemoteThread')])
    partial = ParsedReport('b', family='x', atomic_evidence=[], events=[_event(0, 'OpenProcess'), _event(1, 'VirtualAllocEx')])
    single = ParsedReport('d', family='x', atomic_evidence=[], events=[_event(0, 'OpenProcess')])
    static_only = ParsedReport('c', family='x', atomic_evidence=[], events=[], weak_labels=['T1055.001'])
    result = census_module.census({'train': [injected, partial, static_only, single]})
    row = result['per_technique']['train']['T1055']
    assert result['totals']['train'] == {'reports': 4, 'with_events': 3, 'with_cape_signatures': 0}
    assert row['weak_positive'] == 2
    assert row['confirmed'] == 1 and row['suspected'] == 1
    assert row['weak_positive_and_confirmed'] == 1
    assert row['not_detected'] == 1  # one isolated API never becomes a behavior
    assert ['T1999', 1] in [list(item) for item in result['top_weak_ttps_without_rule']]
