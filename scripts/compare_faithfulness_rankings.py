#!/usr/bin/env python3
"""Compare matched, frozen faithfulness reports without exposing sample IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _extract(report: dict) -> dict:
    result = report.get('static_feature_faithfulness') or report.get('token_identity_faithfulness')
    if not isinstance(result, dict):
        raise ValueError('missing faithfulness result')
    return result


def compare(left: dict, right: dict) -> dict:
    for key in ('source_sha256', 'checkpoint_sha256', 'evaluated_partition'):
        if not left.get(key) or left.get(key) != right.get(key):
            raise ValueError(f'reports differ in {key}')
    a, b = _extract(left), _extract(right)
    for key in ('intervention', 'sample_ids_sha256'):
        # Older identity-only reports predate the explicit intervention field.
        if key == 'intervention':
            x, y = a.get(key, 'identity'), b.get(key, 'identity')
        else:
            x, y = a.get(key), b.get(key)
        if not x or x != y:
            raise ValueError(f'faithfulness reports differ in {key}')
    aa, bb = a['samples'], b['samples']
    if len(aa) != len(bb) or not aa:
        raise ValueError('sample counts differ or are empty')
    result = {}
    for name in ('signature', 'static', 'temporal_events', 'graph'):
        differences = []
        for x, y in zip(aa, bb):
            if any(x.get(key) != y.get(key) for key in
                   ('sample_hash', 'target_index', 'seed', 'fractions', 'random_repeats')):
                raise ValueError('paired sample, target or control settings differ')
            if not math.isclose(x['baseline_probability'], y['baseline_probability'], abs_tol=1e-6):
                raise ValueError('paired baseline scores differ')
            left_branch, right_branch = x['branches'][name], y['branches'][name]
            if left_branch['evaluated'] != right_branch['evaluated']:
                raise ValueError('branch availability differs')
            if not left_branch['evaluated']:
                continue
            if left_branch['masked_counts'] != right_branch['masked_counts']:
                raise ValueError('masking budgets differ')
            if left_branch['valid_tokens'] != right_branch['valid_tokens']:
                raise ValueError('valid token counts differ')
            if not math.isclose(left_branch['random_aopc'], right_branch['random_aopc'], abs_tol=1e-6):
                raise ValueError('random controls differ')
            l_random = left_branch['random_mean_confidence_drop']
            r_random = right_branch['random_mean_confidence_drop']
            if len(l_random) != len(r_random) or any(
                not math.isclose(lv, rv, abs_tol=1e-6) for lv, rv in zip(l_random, r_random)
            ):
                raise ValueError('random controls differ')
            differences.append(right_branch['top_minus_random_aopc'] - left_branch['top_minus_random_aopc'])
        result[name] = {
            'paired_samples': len(differences),
            'mean_right_minus_left_aopc': sum(differences) / len(differences) if differences else None,
            'right_better': sum(value > 1e-9 for value in differences),
            'left_better': sum(value < -1e-9 for value in differences),
            'ties': sum(abs(value) <= 1e-9 for value in differences),
        }
    return {'left_ranking': a['ranking'], 'right_ranking': b['ranking'],
            'intervention': a.get('intervention', 'identity'),
            'sample_ids_sha256': a['sample_ids_sha256'], 'branches': result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    args = parser.parse_args()
    left_bytes, right_bytes = args.left.read_bytes(), args.right.read_bytes()
    output = compare(json.loads(left_bytes), json.loads(right_bytes))
    output['left_report_sha256'] = hashlib.sha256(left_bytes).hexdigest()
    output['right_report_sha256'] = hashlib.sha256(right_bytes).hexdigest()
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
