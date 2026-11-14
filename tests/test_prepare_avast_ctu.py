import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'prepare_avast_ctu', Path(__file__).parents[1] / 'scripts' / 'prepare_avast_ctu.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

FIXTURE = Path(__file__).parent / 'fixtures' / 'cape_process_injection.json'


def test_family_names_match_whole_words_only():
    pattern = module.family_name_pattern(['emotet', 'zeus'])
    assert not pattern.search('functionname=createremotethread')
    assert pattern.search('mutex=emotet_lock') and pattern.search('yara:zeus')


def test_normalize_removes_leaky_fields_and_uses_csv_labels():
    import json
    from malware_hybrid.normalization import CapeReportParser
    raw = json.loads(FIXTURE.read_text())
    raw.update({'malfamily_tag': 'Emotet', 'detections': [{'family': 'Emotet'}],
                'ttps': [{'signature': 'x', 'ttps': ['T1055', 'bad']}], 'avclass_detection': 'zeus'})
    label = {'family': 'qakbot', 'type': 'banker', 'date': '2019-03-20'}
    parser = CapeReportParser(max_events=64, file_handle_resolution=True, registry_value_resolution=True)
    report, _, leaky = module.normalize(raw, 'a' * 64, label, ['emotet', 'qakbot', 'zeus'], parser)
    assert report.family == 'qakbot' and report.families == ['qakbot']
    assert report.metadata['cape_ttps'] == ['T1055']
    assert report.metadata['archive_group_id'] == '2019-03'
    assert 'malfamily_tag' not in raw and 'detections' not in raw
    assert leaky == 0


def test_resume_partial_keeps_complete_lines_and_drops_a_torn_tail(tmp_path):
    import json
    partial = tmp_path / 'out.jsonl.partial'
    rows = [{'sample_hash': 'a' * 64, 'family': 'zeus'}, {'sample_hash': 'b' * 64, 'family': 'emotet'}]
    partial.write_text(''.join(json.dumps(r) + '\n' for r in rows) + '{"sample_hash": "c')
    done = module.resume_partial(partial)
    assert done == {'a' * 64: 'zeus', 'b' * 64: 'emotet'}
    assert partial.read_text().count('\n') == 2 and partial.read_text().endswith('\n')
