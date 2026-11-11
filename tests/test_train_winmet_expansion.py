import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.serialization import report_to_dict

spec = importlib.util.spec_from_file_location('expanded_training', Path(__file__).parents[1] / 'scripts/train_winmet_expansion.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_expansion_loader_retains_only_training_validation_calibration(tmp_path):
    fixture = Path(__file__).parent / 'fixtures/cape_process_injection.json'
    parts = {}
    lines = []
    for index, name in enumerate(('train', 'validation', 'calibration', 'withheld_internal_test')):
        report = CapeReportParser().parse_file(fixture)
        report.sample_hash = str(index)
        report.metadata['archive_group_id'] = name
        payload = report_to_dict(report)
        if name == 'withheld_internal_test':
            # The loader must not construct withheld events/graphs.
            payload['events'] = [{'invalid': 'must not deserialize'}]
        lines.append(json.dumps(payload) + '\n')
        parts[name] = {'archive_groups': [name], 'rows': 1,
                       'sample_ids_sha256': hashlib.sha256(str(index).encode()).hexdigest()}
    source = tmp_path / 'data.jsonl'
    source.write_text(''.join(lines))
    manifest = {'partitions': parts, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    loaded = module.load_expansion(source, manifest)
    assert set(loaded) == {'train', 'validation', 'calibration'}
    assert all(len(rows) == 1 and rows[0].graph is None for rows in loaded.values())
    manifest['source_sha256'] = 'invalid'
    with pytest.raises(ValueError, match='checksum'):
        module.load_expansion(source, manifest)
    manifest['partitions']['validation']['archive_groups'].append('train')
    with pytest.raises(ValueError, match='overlaps'):
        module.load_expansion(source, manifest)


def test_multi_source_manifest_and_named_withheld_partition(tmp_path):
    fixture = Path(__file__).parent / 'fixtures/cape_process_injection.json'
    names = ('train', 'validation', 'calibration', 'withheld_internal_test', 'withheld_test_v2')
    parts, sources = {}, []
    for index, name in enumerate(names):
        report = CapeReportParser().parse_file(fixture)
        report.sample_hash = str(index)
        report.metadata['archive_group_id'] = name
        path = tmp_path / f'{index % 2}.jsonl'
        with path.open('a') as handle:
            handle.write(json.dumps(report_to_dict(report)) + '\n')
        parts[name] = {'archive_groups': [name], 'rows': 1,
                       'sample_ids_sha256': hashlib.sha256(str(index).encode()).hexdigest()}
    for path in sorted(tmp_path.glob('*.jsonl')):
        sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {'partitions': parts, 'sources': sources}
    assert set(module.load_expansion(None, manifest)) == {'train', 'validation', 'calibration'}
    loaded = module.load_expansion(None, manifest, include_test='withheld_test_v2')
    assert set(loaded) == {'train', 'validation', 'calibration', 'withheld_test_v2'}
    with pytest.raises(ValueError, match='not in the manifest'):
        module.load_expansion(None, manifest, include_test='withheld_missing')
    sources[0]['sha256'] = 'bad'
    with pytest.raises(ValueError, match='checksum'):
        module.load_expansion(None, manifest)
