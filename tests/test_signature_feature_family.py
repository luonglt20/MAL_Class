import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('sklearn')

from malware_hybrid.schema import ExecutionEvent, ParsedReport

SPEC = importlib.util.spec_from_file_location(
    'signature_feature_family', Path(__file__).parents[1] / 'scripts' / 'signature_feature_family.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _report(name, family, signatures):
    events = [ExecutionEvent(0, 0.0, '1', '0', '1', 'CreateFileW', 'file', True, '0', 'p', 't', 'file')]
    return ParsedReport(name, family=family, events=events, atomic_evidence=[],
                        metadata={'family_supervised': True, 'cape_signatures': signatures})


def test_signature_tokens_are_prefixed_and_optional():
    report = _report('a', 'x', ['injection_rwx'])
    assert module.tokens(report, 'signatures') == ['g:injection_rwx']
    assert 'g:injection_rwx' in module.tokens(report, 'base+signatures')
    assert 'g:injection_rwx' not in module.tokens(report, 'base')


def test_signatures_separate_families_that_events_cannot():
    def corpus(prefix, n):
        rows = []
        for i in range(n):
            rows.append(_report(f'{prefix}a{i}', 'alpha', ['infostealer_browser']))
            rows.append(_report(f'{prefix}b{i}', 'beta', ['ransomware_file_modifications']))
        return rows
    result = module.run(corpus('t', 6), corpus('v', 3), c=4.0, log=lambda _: None)
    assert result['features']['base']['family_top1_accuracy'] == 0.5
    assert result['features']['base+signatures']['family_top1_accuracy'] == 1.0
    assert result['base+signatures_vs_base']['gained'] == 3
