"""Census of behavior labels available for training a multi-label behavior head.

For each ATT&CK technique with an evidence-chain rule, counts reports where the
ordered chain is ``confirmed`` or only ``suspected``, and compares them with the
weak CAPE signature TTP labels (``report.weak_labels``). Also lists the most
frequent weak TTPs that have no rule yet. Withheld test partitions are never parsed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_winmet_expansion import load_expansion  # noqa: E402
from malware_hybrid.evidence import EvidenceEngine  # noqa: E402


def _base(ttp: str) -> str:
    return ttp.split('.')[0]


def census(partitions, engine=None, top_unruled=25):
    engine = engine or EvidenceEngine()
    techniques = [spec.technique_id for spec in engine.specifications]
    rows = {name: {t: Counter() for t in techniques} for name in partitions}
    weak_all = {name: Counter() for name in partitions}
    signatures = {name: Counter() for name in partitions}
    reports_with_signatures = Counter()
    totals = {}
    for name, reports in partitions.items():
        n = with_events = 0
        for report in reports:
            n += 1
            with_events += bool(report.events)
            weak = set(report.weak_labels)
            weak_base = {_base(label) for label in weak}
            weak_all[name].update(weak)
            names = report.metadata.get('cape_signatures') or []
            signatures[name].update(names)
            reports_with_signatures[name] += bool(names)
            matches = engine.evaluate(report.events) if report.events else {}
            for technique in techniques:
                status = matches[technique].status if technique in matches else 'absent'
                in_weak = technique in weak or _base(technique) in weak_base
                rows[name][technique][status] += 1
                if in_weak:
                    rows[name][technique]['weak_positive'] += 1
                    rows[name][technique][f'weak_positive_and_{status}'] += 1
        totals[name] = {'reports': n, 'with_events': with_events,
                        'with_cape_signatures': reports_with_signatures[name]}
    ruled = set(techniques) | {_base(t) for t in techniques}
    unruled = Counter()
    for counter in weak_all.values():
        for label, count in counter.items():
            if label.startswith('T') and label not in ruled and _base(label) not in ruled:
                unruled[label] += count
    return {
        'totals': totals,
        'per_technique': {name: {t: dict(c) for t, c in per.items()} for name, per in rows.items()},
        'weak_label_kinds': {name: len(c) for name, c in weak_all.items()},
        'top_weak_ttps_without_rule': unruled.most_common(top_unruled),
        'cape_signature_kinds': {name: len(c) for name, c in signatures.items()},
        'cape_signature_counts': {name: c.most_common() for name, c in signatures.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite output')
    manifest = json.loads(args.manifest.read_text())
    reports = load_expansion(manifest.get('source'), manifest, pack=True)  # no withheld partitions
    result = census({name: reports[name] for name in ('train', 'validation', 'calibration')})
    result['notice'] = ('Rule statuses come from evidence.py chains; weak labels are CAPE signature TTP/MBC '
                        'annotations. Neither is ground truth. Test partitions were not parsed.')
    args.output.write_text(json.dumps(result, indent=2))
    train = result['per_technique']['train']
    print(json.dumps({t: {k: train[t].get(k, 0) for k in ('confirmed', 'suspected', 'weak_positive',
                                                             'weak_positive_and_confirmed')}
                      for t in train}, indent=1))
    print(json.dumps({'totals': result['totals'], 'top_unruled': result['top_weak_ttps_without_rule'][:10],
                      'signature_kinds': result['cape_signature_kinds'],
                      'top_train_signatures': result['cape_signature_counts']['train'][:30]}))


if __name__ == '__main__':
    main()
