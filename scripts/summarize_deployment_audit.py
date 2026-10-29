#!/usr/bin/env python3
"""Summarize saved guarded-decision reports without publishing sample IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from malware_hybrid.deployment_audit import explore_candidate_thresholds, summarize_candidate_errors, summarize_labeled_decisions


def summarize(document: dict, source_sha256: str) -> dict:
    audit = document.get('deployment_family_decisions')
    if not isinstance(audit, dict) or not isinstance(audit.get('rows'), list):
        raise ValueError('report contains no saved guarded-decision rows')
    rows = audit['rows']
    if audit.get('samples') != len(rows):
        raise ValueError('saved sample count does not match decision rows')
    for row in rows:
        if (not isinstance(row.get('reference_families'), list)
                or not isinstance(row.get('reference_known_family'), bool)
                or not isinstance(row.get('accepted'), bool)):
            raise ValueError('invalid saved decision row')
        if row.get('accepted') and row.get('primary_matches_reference') not in (True, False, None):
            raise ValueError('invalid reference-match value')
    return {
        'source_report_sha256': source_sha256,
        'checkpoint_sha256': document.get('checkpoint_sha256'),
        'evaluated_partition': document.get('evaluated_partition'),
        'samples': len(rows), 'errors': audit.get('errors'),
        'summary': summarize_labeled_decisions(rows),
        'candidate_threshold_exploration': explore_candidate_thresholds(rows),
        'candidate_error_analysis': summarize_candidate_errors(rows),
        'notice': ('Guarded primary-family agreement with available reference labels; '
                   'unknown references are excluded, abstentions and errors are unresolved. '
                   'Not multi-label exact-match accuracy or independent ground truth.'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    print(json.dumps(summarize(json.loads(raw), hashlib.sha256(raw).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
