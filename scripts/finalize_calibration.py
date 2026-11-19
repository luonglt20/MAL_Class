"""Finish a training run whose post-training calibration step crashed.

``train_model`` saves the selected weights at every improving epoch, then fits
family conformal sets, per-family thresholds and the malware threshold on the
calibration partition. If that last step fails (e.g. MPS out of memory), this
script loads the saved checkpoint in a fresh process, runs the same three
calibration functions on the same calibration partition, saves the calibrated
checkpoint and writes the run report. No training and no withheld partition.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

import torch

from malware_hybrid.training import (
    fit_family_conformal, fit_family_thresholds, fit_malware_threshold, load_checkpoint, save_checkpoint,
)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_winmet_expansion import load_expansion

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--config', type=json.loads, default={}, help='training config to record in the report')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('refusing to overwrite report')
    backup = args.checkpoint.with_suffix('.uncalibrated.pt')
    if backup.exists():
        parser.error('uncalibrated backup already exists; this checkpoint may already be finalized')
    manifest = json.loads(args.manifest.read_text())
    reports = load_expansion(manifest.get('source'), manifest, only=('calibration',))
    calibration = reports['calibration']
    device = torch.device(args.device)
    model, tensorizer = load_checkpoint(args.checkpoint, device=args.device)
    history = model.training_history
    print(json.dumps({'stage': 'finalize', 'calibration_reports': len(calibration),
                      'selected_epoch': model.selected_epoch}), flush=True)
    conformal = fit_family_conformal(model, tensorizer, calibration, device, batch_size=args.batch_size)
    thresholds = fit_family_thresholds(model, tensorizer, calibration, device, batch_size=args.batch_size)
    model.family_thresholds = {label: float(item['threshold']) for label, item in thresholds.items()}
    malware = fit_malware_threshold(model, tensorizer, calibration, device, batch_size=args.batch_size)
    model.malware_threshold = float(malware['threshold'])
    shutil.copy2(args.checkpoint, backup)
    save_checkpoint(args.checkpoint, model, tensorizer, history, conformal,
                    malware_calibration=malware, family_thresholds=thresholds)
    result = {
        'checkpoint': str(args.checkpoint), 'history': history, 'epochs_completed': len(history),
        'selection_metric': model.training_selection_metric, 'selected_epoch': model.selected_epoch,
        'conformal': conformal.as_dict() if conformal else None, 'family_thresholds': thresholds,
        'behavior_head_trained': bool(model.behavior_head_trained), 'malware_calibration': malware,
        'normalization_versions': model.normalization_versions,
    }
    with args.report.open('x') as output:
        json.dump({'manifest': manifest, 'config': args.config, 'result': result, 'test_metrics': None,
                   'notice': 'Calibration finalized in a separate process after the in-run calibration step '
                             'ran out of MPS memory; weights are the selected epoch saved during training. '
                             'Withheld partitions were not loaded.'}, output, indent=2)
    print(json.dumps({'stage': 'done', 'report': str(args.report)}), flush=True)


if __name__ == '__main__':
    main()
