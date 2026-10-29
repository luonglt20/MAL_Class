"""Evaluate a completed expanded candidate on preserved validation groups only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch

from evaluate_winmet_flow_ablation import _family_metrics, _sha256_file
from train_winmet_expansion import load_expansion
from malware_hybrid.evaluation import ENCODED_ABLATIONS
from malware_hybrid.training import _family_training_priors, _training_loss_weights, load_checkpoint


def validate_inputs(checkpoint, checksum, manifest, training_report):
    if _sha256_file(checkpoint) != checksum:
        raise ValueError('checkpoint checksum mismatch')
    if training_report.get('manifest') != manifest:
        raise ValueError('training manifest mismatch')
    result = training_report.get('result') or {}
    if (not result.get('epochs_completed') or result.get('epochs_completed') != result.get('epochs_requested')
            or not result.get('conformal')):
        raise ValueError('completed training and calibration report required')
    if Path(result.get('checkpoint', '')).resolve() != checkpoint.resolve():
        raise ValueError('checkpoint path differs from training report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('training_report', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--modes', nargs='*', choices=('disable_cross_branch_attention', *ENCODED_ABLATIONS),
                        default=['disable_cross_branch_attention', 'mask_graph_edges_keep_nodes'])
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error('output must be new and batch size positive')
    manifest = json.loads(args.manifest.read_text())
    training = json.loads(args.training_report.read_text())
    validate_inputs(args.checkpoint, args.checkpoint_sha256, manifest, training)
    torch.set_num_threads(2)
    reports = load_expansion(manifest['source'], manifest)
    model, tensorizer = load_checkpoint(args.checkpoint, args.device)
    if _family_training_priors(tensorizer, reports['train']) != model.family_training_priors:
        raise ValueError('model priors differ from preserved training partition')
    if model.selected_epoch != training['result']['selected_epoch']:
        raise ValueError('selected epoch differs from training report')
    device = torch.device(args.device)
    weights = _training_loss_weights(tensorizer, reports['train'], device)
    baseline = _family_metrics(model, reports['validation'], tensorizer, device, args.batch_size, weights)
    controls = {}
    for mode in args.modes:
        metric = _family_metrics(model, reports['validation'], tensorizer, device, args.batch_size, weights,
                                disable_cross_branch_attention=mode == 'disable_cross_branch_attention',
                                encoded_ablation=mode if mode in ENCODED_ABLATIONS else None)
        controls[mode] = {'metrics': metric, 'delta': {
            key: metric[key] - baseline[key] for key in ('family_map', 'family_calibrated_macro_f1', 'family_top1_accuracy')
            if metric.get(key) is not None and baseline.get(key) is not None}}
    result = {'checkpoint_sha256': args.checkpoint_sha256,
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'training_report_sha256': hashlib.sha256(args.training_report.read_bytes()).hexdigest(),
              'source_sha256': manifest['source_sha256'], 'partitions': manifest['partitions'],
              'evaluated_partition': 'validation', 'baseline': baseline, 'controls': controls,
              'test_metrics': None,
              'notice': 'Weak-label validation with preserved groups; expanded label vocabulary differs '
                        'from the old model. Counterfactual score changes do not establish causal explanations. '
                        'No test predictions, model updates, or UI promotion.'}
    with args.output.open('x') as target:
        json.dump(result, target, indent=2)


if __name__ == '__main__':
    main()
