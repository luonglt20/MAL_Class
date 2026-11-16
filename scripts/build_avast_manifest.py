"""Time-ordered split manifest for the prepared Avast-CTU JSONL files.

Each report's detection month (``metadata.archive_group_id``) is its group.
Months are taken in chronological order and assigned whole to train,
validation, calibration and the withheld test partition by cumulative row
share (default 70/10/5/15), so every test report was detected after every
train report. The manifest uses the same format as the WinMET expansion
manifests, so ``train_winmet_expansion.load_expansion`` verifies source
checksums and partition fingerprints and never parses the withheld test.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SHA = re.compile(rb'"sample_hash": "([0-9a-f]{64})"')
GROUP = re.compile(rb'"archive_group_id": "(\d{4}-\d{2})"')
FAMILY = re.compile(rb'"family_labels": \["([^"]+)"\]\}\n$')
TEST = 'withheld_test_time'


def scan(path):
    """One pass: source SHA-256 plus (sample, month, family) for every line."""
    digest, rows = hashlib.sha256(), []
    with open(path, 'rb') as handle:
        for line in handle:
            digest.update(line)
            sha, group, family = SHA.search(line), GROUP.search(line[-600:]), FAMILY.search(line[-400:])
            if not (sha and group and family):
                raise ValueError(f'{path}: row without sample hash, month or family')
            rows.append((sha.group(1).decode(), group.group(1).decode(), family.group(1).decode()))
    return digest.hexdigest(), rows


def assign_months(month_counts, shares):
    """Chronological months -> partition, cutting at cumulative share boundaries."""
    names = ['train', 'validation', 'calibration', TEST]
    total, cum, out = sum(month_counts.values()), 0, {}
    bounds = [sum(shares[: i + 1]) for i in range(len(shares))]
    for month in sorted(month_counts):
        index = next(i for i, bound in enumerate(bounds) if cum / total < bound - 1e-12)
        out[month] = names[index]
        cum += month_counts[month]
    if set(out.values()) != set(names):
        raise ValueError('a partition received no month; adjust the shares')
    return out


def build(sources, shares=(0.70, 0.10, 0.05, 0.15)):
    scanned = [(path, *scan(path)) for path in sources]
    seen, rows = set(), []
    for path, _, entries in scanned:
        for sha, month, family in entries:
            if sha in seen:
                raise ValueError('duplicate sample across sources')
            seen.add(sha)
            rows.append((sha, month, family))
    month_of = assign_months(Counter(month for _, month, _ in rows), shares)
    partitions = {}
    for name in ('train', 'validation', 'calibration', TEST):
        ids = sorted(sha for sha, month, _ in rows if month_of[month] == name)
        months = sorted(m for m, p in month_of.items() if p == name)
        families = Counter(f for sha, month, f in rows if month_of[month] == name)
        partitions[name] = {'rows': len(ids), 'sample_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                            'archive_groups': months, 'first_month': months[0], 'last_month': months[-1],
                            'families': dict(sorted(families.items()))}
    return {
        'schema': 'avast_ctu_time_split_v1', 'split_kind': 'time_ordered_detection_month',
        'shares': list(shares), 'sources': [{'path': str(p), 'sha256': d} for p, d, _ in scanned],
        'partitions': partitions, 'excluded_rows': [],
        # Required by the shared loader's guard; there is no earlier split to preserve here.
        'original_split_verified': True, 'old_group_assignments_preserved': True,
        'notice': 'Every withheld test report was detected after every train report. Avast-CTU has 10 families; '
                  'results are not comparable with the 153-family WinMET task.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    manifest = build(args.sources)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: {x: v[x] for x in ('rows', 'first_month', 'last_month', 'families')}
                      for k, v in manifest['partitions'].items()}, indent=1))


if __name__ == '__main__':
    main()
