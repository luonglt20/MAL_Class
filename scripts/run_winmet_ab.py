"""Run WinMET expansion A/B training variants one at a time on one manifest.

Variants change one factor against the baseline so validation differences can
be attributed. Runs are sequential because a 16 GB machine cannot hold two
training processes. Existing checkpoints/reports are skipped, never
overwritten, so the runner can be restarted after an interruption. The
withheld test partitions are never loaded here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MEMORY = ['--pre-encode', '--mps-empty-cache-every', '50']
VARIANTS = {
    'baseline': [],
    'baseline_seed45': ['--seed', '45'],
    'baseline_seed46': ['--seed', '46'],
    'd64': ['--d-model', '64'],
    'b16_cosine': ['--batch-size', '16', '--lr', '6e-4', '--lr-schedule', 'warmup_cosine'],
    'balanced': ['--balanced-sampling'],
    'cosine_balanced': ['--lr-schedule', 'warmup_cosine', '--balanced-sampling'],
    'b16_cosine_balanced': ['--batch-size', '16', '--lr', '6e-4', '--lr-schedule', 'warmup_cosine', '--balanced-sampling'],
}


def plan(manifest: Path, tag: str, epochs: int, device: str, names: list[str], patience: int | None = None):
    for name in names:
        stem = f'winmet_ab_{tag}_{name}'
        yield name, {
            'checkpoint': ROOT / 'models' / f'{stem}.pt',
            'report': ROOT / 'artifacts' / 'winmet' / f'{stem}.json',
            'log': ROOT / 'artifacts' / 'winmet' / f'{stem}.log',
            'command': [sys.executable, str(ROOT / 'scripts' / 'train_winmet_expansion.py'), str(manifest),
                        '--checkpoint', str(ROOT / 'models' / f'{stem}.pt'),
                        '--report', str(ROOT / 'artifacts' / 'winmet' / f'{stem}.json'),
                        '--epochs', str(epochs), '--device', device, *MEMORY, *VARIANTS[name],
                        *(['--patience', str(patience)] if patience else [])],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--tag', required=True, help='short dataset tag used in output names, e.g. v2')
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--only', nargs='+', choices=sorted(VARIANTS), default=list(VARIANTS),
                        help='variants to run, in the given order')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    env = {'PYTHONPATH': f"{ROOT / 'src'}:{ROOT / 'scripts'}"}
    for name, run in plan(args.manifest.resolve(), args.tag, args.epochs, args.device, args.only, args.patience):
        if run['checkpoint'].exists() or run['report'].exists():
            print(json.dumps({'variant': name, 'status': 'skipped_existing'}), flush=True)
            continue
        print(json.dumps({'variant': name, 'status': 'start', 'command': run['command'][1:]}), flush=True)
        if args.dry_run:
            continue
        with run['log'].open('x') as log:
            code = subprocess.run(run['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, **env}).returncode
        print(json.dumps({'variant': name, 'status': 'done' if code == 0 else 'failed', 'returncode': code}), flush=True)


if __name__ == '__main__':
    main()
