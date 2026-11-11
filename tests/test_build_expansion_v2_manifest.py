import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

builder_path = Path(__file__).parents[1] / 'scripts/build_expansion_v2_manifest.py'
spec = importlib.util.spec_from_file_location('expanded_training', Path(__file__).parents[1] / 'scripts/train_winmet_expansion.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
from malware_hybrid.normalization import CapeReportParser  # noqa: E402
from malware_hybrid.serialization import report_to_dict  # noqa: E402

FIXTURE = Path(__file__).parent / 'fixtures/cape_process_injection.json'


def _write(path, rows):
    with path.open('w') as handle:
        for sample_hash, group in rows:
            report = CapeReportParser(file_handle_resolution=True).parse_file(FIXTURE)
            report.sample_hash = sample_hash
            report.family_labels = ['fam']
            report.metadata.update(archive_group_id=group, fuzzy_hashes={'imphash': 'abc'})
            handle.write(json.dumps(report_to_dict(report)) + '\n')


def test_v2_manifest_keeps_v1_partitions_and_dedupes(tmp_path):
    old = tmp_path / 'old.jsonl'
    names = ('train', 'validation', 'calibration', 'withheld_internal_test')
    _write(old, [(f'h{i}', f'g{i}') for i in range(4)])
    v1 = {'source': str(old), 'source_sha256': hashlib.sha256(old.read_bytes()).hexdigest(),
          'partitions': {name: {'rows': 1, 'archive_groups': [f'g{i}'],
                                'sample_ids_sha256': hashlib.sha256(f'h{i}'.encode()).hexdigest()}
                         for i, name in enumerate(names)}}
    v1_path = tmp_path / 'v1.json'
    v1_path.write_text(json.dumps(v1))
    extra = tmp_path / 'extra.jsonl'
    _write(extra, [('h3', 'n0'), ('x1', 'n0'), ('x2', 'n1')])  # h3 duplicates a v1 test row
    second = tmp_path / 'second.jsonl'
    _write(second, [(f'y{i}', f's{i}') for i in range(10)])
    output = tmp_path / 'v2.json'
    subprocess.run([sys.executable, str(builder_path), str(v1_path), '--train-source', str(extra),
                    '--second-test-source', str(second), '--output', str(output)], check=True,
                   capture_output=True)
    manifest = json.loads(output.read_text())
    parts = manifest['partitions']
    assert manifest['v1_test_unchanged'] and parts['withheld_internal_test']['archive_groups'] == ['g3']
    assert [row['sample_hash'] for row in manifest['excluded_rows']] == ['h3']
    assert parts['withheld_test_v2']['rows'] == 2 and parts['validation']['rows'] == 2
    assert parts['train']['rows'] == 1 + 2 + 7
    assert sum(part['rows'] for part in parts.values()) == 4 + 2 + 10
    loaded = loader.load_expansion(None, manifest)
    assert {name: len(rows) for name, rows in loaded.items()} == {'train': 10, 'validation': 2, 'calibration': 1}
    assert manifest['imphash_overlap_new_train_vs_withheld']['withheld_internal_test'][
        'new_train_rows_with_imphash_seen_in_partition'] == 9
