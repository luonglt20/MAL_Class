"""Where are the unused WinMET reports, how big are they, and which families are they?"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import PurePosixPath
import py7zr
manifest = json.load(open(sys.argv[1])); labels = json.load(open(sys.argv[2]))
group_part = {g: name for name, p in manifest['partitions'].items() for g in p['archive_groups']}
used = set()
for src in manifest['sources']:
    with open(src['path'], 'rb') as f:
        for line in f:
            m = re.search(rb'"sample_hash": "([0-9a-f]{64})"', line)
            used.add(m.group(1).decode())
train_support = Counter()
out = Counter(); fam = Counter(); sizes = []
for path in sys.argv[3:]:
    with py7zr.SevenZipFile(path, password='infected') as a:
        blocks = {}
        for m in a.files:
            if m.is_file and not m.is_symlink:
                key = id(m.folder) if m.folder is not None else ('file', m.filename)
                blocks.setdefault(key, []).append(m)
        for members in blocks.values():
            bid = hashlib.sha256('\n'.join(sorted(x.filename for x in members)).encode()).hexdigest()[:16]
            part = group_part.get(bid, 'unassigned_block')
            for m in members:
                name = PurePosixPath(m.filename)
                if not re.fullmatch(r'[0-9a-f]{64}\.json', name.name):
                    continue
                h = name.stem; family = (labels.get(h) or {}).get('avclass')
                if h in used:
                    if part == 'train' and family: train_support[str(family).lower()] += 1
                    continue
                out[part] += 1; sizes.append(m.uncompressed)
                if part in ('train', 'unassigned_block') and family: fam[str(family).lower()] += 1
sizes.sort()
rare = {f: n for f, n in fam.items() if 5 <= train_support[f] < 200 or (train_support[f] < 5 and train_support[f] + n >= 5)}
print(json.dumps({'unused_by_partition': out, 'unused_size_mib_median': sizes[len(sizes)//2] / 2**20 if sizes else None,
                  'unused_over_64mib': sum(s > 64 * 2**20 for s in sizes),
                  'unused_train_or_unassigned_reports_in_rare_or_newly_supported_families': sum(rare.values()),
                  'top_unused_families': fam.most_common(15)}, indent=1))
