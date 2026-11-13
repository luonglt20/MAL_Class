"""Point a frozen expansion manifest at re-normalized copies of its sources.

The partitions (archive groups, row counts and sample-ID fingerprints) are kept
unchanged; only source paths and checksums are replaced. The new manifest is then
loaded with the normal loader, which re-checks every partition fingerprint, so a
re-normalization that dropped or added a sample is refused.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_winmet_expansion import load_expansion  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rebase(manifest: dict, mapping: dict[str, str], normalization_version: str) -> dict:
    sources = manifest.get('sources')
    if not sources or {s['path'] for s in sources} != set(mapping):
        raise ValueError('mapping must cover exactly the manifest sources')
    result = json.loads(json.dumps(manifest))
    result['sources'] = []
    for source in sources:
        new_path = mapping[source['path']]
        prepared = json.loads(Path(new_path + '.manifest.json').read_text())
        if prepared.get('normalization_version') != normalization_version:
            raise ValueError(f'{new_path} is not {normalization_version}')
        result['sources'].append({'path': new_path, 'sha256': sha256(new_path)})
    result['excluded_rows'] = [dict(row, source=mapping[row['source']]) for row in manifest.get('excluded_rows', [])]
    result['rebased_from'] = {'normalization_version': normalization_version,
                              'previous_sources': sources}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--map', nargs=2, action='append', metavar=('OLD', 'NEW'), required=True)
    parser.add_argument('--normalization-version', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    manifest = json.loads(args.manifest.read_text())
    result = rebase(manifest, dict(args.map), args.normalization_version)
    reports = load_expansion(result.get('source'), result, pack=True)  # re-checks every partition fingerprint
    counts = {name: len(rows) for name, rows in reports.items()}
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'output': str(args.output), 'verified_partition_rows': counts}))


if __name__ == '__main__':
    main()
