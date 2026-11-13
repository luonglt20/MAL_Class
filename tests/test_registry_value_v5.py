import pytest

from malware_hybrid.evidence import EvidenceEngine
from malware_hybrid.normalization import CapeReportParser, behavioral_signature_names

RUN = 'HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run'


def _call(api, args, ts):
    return {'api': api, 'category': 'registry', 'status': True, 'return': '0x00000000', 'timestamp': ts,
            'arguments': [{'name': k, 'value': v} for k, v in args.items()]}


def _report(set_fullname):
    return {'target': {'file': {'sha256': 'a' * 64}},
            'signatures': [{'name': 'persistence_autorun'}, {'name': 'cape_detected_threat'},
                           {'name': 'procmem_yara'}],
            'behavior': {'processes': [{'process_id': 3020, 'parent_id': 1, 'calls': [
                _call('RegOpenKeyExW', {'Registry': '0xffffffff80000001',
                                         'SubKey': 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run',
                                         'Handle': '0x00000450', 'FullName': RUN}, '2026-01-01 00:00:01,000'),
                _call('RegSetValueExW', {'Handle': '0x00000450', 'ValueName': 'ajjf', 'Type': '1',
                                          'Buffer': 'C:\\x.exe', 'FullName': set_fullname},
                      '2026-01-01 00:00:02,000')]}]}}


def _status(raw, v5):
    parser = CapeReportParser(file_handle_resolution=True, registry_value_resolution=v5)
    return EvidenceEngine().evaluate(parser.parse(raw).events)['T1547.001'].status


def test_v5_links_run_value_write_to_opened_key_and_v4_is_unchanged():
    raw = _report(RUN + '\\ajjf')
    assert _status(raw, False) != 'confirmed'
    assert _status(raw, True) == 'confirmed'


def test_v5_rejects_a_write_whose_path_contradicts_the_handle():
    # v4 trusted the handle here and confirmed; v5 sees the real paths disagree.
    raw = _report('HKEY_CURRENT_USER\\SOFTWARE\\Other\\Key\\ajjf')
    assert _status(raw, False) == 'confirmed'
    assert _status(raw, True) != 'confirmed'


def test_v5_records_only_behavioral_signatures():
    parsed = CapeReportParser(file_handle_resolution=True, registry_value_resolution=True).parse(_report(RUN))
    assert parsed.metadata['cape_signatures'] == ['persistence_autorun']
    assert parsed.metadata['normalization_version'] == 'cape-resource-relations-v5'
    assert 'cape_signatures' not in CapeReportParser(file_handle_resolution=True).parse(_report(RUN)).metadata
    assert behavioral_signature_names({'signatures': [{'name': '../bad'}, {'name': 'virus'}]}) == []


def test_v5_requires_file_handle_resolution():
    with pytest.raises(ValueError):
        CapeReportParser(registry_value_resolution=True)
