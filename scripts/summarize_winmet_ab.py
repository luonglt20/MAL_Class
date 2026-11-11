"""Summarize finished A/B training reports (validation only) into JSON and Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = ('family_top1_accuracy', 'family_top3_accuracy', 'family_macro_f1', 'family_map', 'loss')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, required=True, help='output stem; writes .json and .md')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'artifacts' / 'winmet'
    rows = []
    for path in sorted(root.glob(f'winmet_ab_{args.tag}_*.json')):
        result = json.loads(path.read_text())
        history = result['result']['history']
        selected = result['result']['selected_epoch']
        best = next(item for item in history if item['epoch'] == selected)
        rows.append({'variant': path.stem.removeprefix(f'winmet_ab_{args.tag}_'), 'config': result['config'],
                     'selected_epoch': selected, 'epochs_completed': len(history),
                     'evaluated_samples': best.get('family_evaluated_samples'),
                     **{key: best.get(key) for key in KEYS}})
    rows.sort(key=lambda row: -(row['family_top1_accuracy'] or 0))
    args.output.with_suffix('.json').write_text(json.dumps(rows, indent=2))
    lines = ['| Variant | Epoch | Top-1 | Top-3 | Macro-F1 | mAP | Val loss | n |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    pct = lambda value: '–' if value is None else f'{100 * value:.2f}%'
    for row in rows:
        lines.append(f"| {row['variant']} | {row['selected_epoch']}/{row['epochs_completed']} | {pct(row['family_top1_accuracy'])} | "
                     f"{pct(row['family_top3_accuracy'])} | {pct(row['family_macro_f1'])} | {pct(row['family_map'])} | "
                     f"{row['loss']:.4f} | {row['evaluated_samples']} |")
    note = ('\nValidation partition of the v2 manifest only; withheld test partitions were not loaded. '
            'Checkpoints are selected by family mAP on validation.\n')
    args.output.with_suffix('.md').write_text('\n'.join(lines) + '\n' + note)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
