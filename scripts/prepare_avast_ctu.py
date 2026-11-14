"""Prepare the Avast-CTU CAPEv2 dataset as normalized v5 JSONL, streaming from the zip.

Reports are read one at a time from ``public_full_reports.zip`` (564 GiB
uncompressed, never extracted). Family labels, malware type and detection date
come only from ``public_labels.csv`` (Avast analyst labels). Fields that name a
family or restate a detection (``malfamily_tag``, ``detections``, ``CAPE``
configs, Suricata, process memory, dropped files, strings) are removed before
parsing, and any remaining atomic evidence token that contains one of the
dataset's family names is dropped for every report alike, so label-dependent
filtering never differs between train and inference. CAPE ``ttps`` are kept as
behavior metadata, never as model input. ``archive_group_id`` is the detection
month, which supports time-ordered splits. Nothing is executed.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path, PurePosixPath
import re
import zipfile

from malware_hybrid.normalization import CapeReportParser
from malware_hybrid.serialization import report_to_dict

LEAKY_KEYS = ('malfamily_tag', 'detections', 'detections2pid', 'CAPE', 'suricata', 'procmemory', 'procdump',
              'dropped', 'strings', 'curtain', 'debug', 'deduplicated_shots', 'malscore', 'ttps')

_STATE = {}


def load_labels(path):
    labels = {}
    with open(path, newline='') as handle:
        for row in csv.DictReader(handle):
            sha = row['sha256'].strip().lower()
            if not re.fullmatch(r'[0-9a-f]{64}', sha):
                raise ValueError('label row without a SHA-256')
            labels[sha] = {'family': row['classification_family'].strip().lower(),
                           'type': row['classification_type'].strip().lower(), 'date': row['date'].strip()}
    return labels


def cape_ttps(raw):
    out = set()
    for entry in raw.get('ttps') or []:
        values = entry.get('ttps') if isinstance(entry, dict) else entry
        for value in values if isinstance(values, list) else [values]:
            value = value.get('ttp') if isinstance(value, dict) else value
            if isinstance(value, str) and re.fullmatch(r'T\d{4}(?:\.\d{3})?', value):
                out.add(value)
    return sorted(out)


def family_name_pattern(family_names):
    """Whole-word match, so e.g. 'createremotethread' does not match 'emotet'."""
    return re.compile(r'(?<![a-z0-9])(?:' + '|'.join(map(re.escape, family_names)) + r')(?![a-z0-9])')


def normalize(raw, sha, label, family_names, parser):
    pattern = family_name_pattern(family_names)
    ttps = cape_ttps(raw)
    for key in LEAKY_KEYS:
        raw.pop(key, None)
    raw.pop('avclass_detection', None)
    report = parser.parse(raw)
    report.sample_hash = sha
    before = len(report.atomic_evidence)
    report.atomic_evidence = [item for item in report.atomic_evidence
                              if not pattern.search(item.token.lower())]
    report.family = label['family']
    report.family_labels = [label['family']]
    report.metadata.update({
        'family_supervised': True, 'family_label_source': 'Avast-CTU analyst label',
        'malware_type': label['type'], 'detection_date': label['date'],
        'archive_group_id': label['date'][:7], 'cape_ttps': ttps,
    })
    leaky_events = sum(bool(pattern.search(event.token.lower())) for event in report.events)
    return report, before - len(report.atomic_evidence), leaky_events


def _init(zip_path, family_names):
    _STATE['zip'] = zipfile.ZipFile(zip_path)
    _STATE['names'] = family_names
    _STATE['parser'] = CapeReportParser(max_events=2048, file_handle_resolution=True, registry_value_resolution=True)


def _work(task):
    member, sha, label = task
    try:
        raw = json.loads(_STATE['zip'].read(member))
        report, dropped, leaky_events = normalize(raw, sha, label, _STATE['names'], _STATE['parser'])
        return sha, json.dumps(report_to_dict(report)), dropped, leaky_events, None
    except Exception as error:  # a malformed report is recorded, never fatal
        return sha, None, 0, 0, type(error).__name__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports_zip', type=Path)
    parser.add_argument('labels_csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-report-mib', type=int, default=512)
    parser.add_argument('--min-report-mib', type=int, default=0,
                        help='only reports above this size (second pass for large reports)')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--limit', type=int, default=None, help='first N eligible reports (smoke runs)')
    args = parser.parse_args()
    manifest_path = Path(str(args.output) + '.manifest.json')
    if args.output.exists() or manifest_path.exists():
        parser.error('refusing to overwrite output')
    labels = load_labels(args.labels_csv)
    family_names = sorted({label['family'] for label in labels.values()})
    tasks, skipped_size, unlabeled = [], Counter(), 0
    with zipfile.ZipFile(args.reports_zip) as archive:
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if info.is_dir() or '..' in name.parts or not re.fullmatch(r'[0-9a-f]{64}\.json', name.name):
                continue
            sha = name.stem
            if sha not in labels:
                unlabeled += 1
                continue
            if info.file_size > args.max_report_mib * 2 ** 20:
                skipped_size[labels[sha]['family']] += 1
                continue
            if info.file_size <= args.min_report_mib * 2 ** 20:
                continue
            tasks.append((info.filename, sha, labels[sha]))
    tasks = tasks[: args.limit] if args.limit else tasks
    written, errors, dropped_tokens, leaky_event_reports = Counter(), Counter(), 0, 0
    temporary = args.output.with_name(args.output.name + '.partial')
    with temporary.open('w') as sink, Pool(args.workers, _init, (str(args.reports_zip), family_names)) as pool:
        for index, (sha, line, dropped, leaky_events, error) in enumerate(pool.imap(_work, tasks, chunksize=4), 1):
            if error:
                errors[error] += 1
                continue
            sink.write(line + '\n')
            written[labels[sha]['family']] += 1
            dropped_tokens += dropped
            leaky_event_reports += bool(leaky_events)
            if index % 500 == 0:
                print(json.dumps({'progress': index, 'of': len(tasks)}), flush=True)
    temporary.rename(args.output)
    digest = hashlib.sha256()
    with args.output.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            digest.update(chunk)
    manifest = {
        'source': str(args.reports_zip), 'labels': str(args.labels_csv),
        'normalization_version': CapeReportParser(file_handle_resolution=True,
                                                  registry_value_resolution=True).normalization_version,
        'max_report_mib': args.max_report_mib, 'min_report_mib': args.min_report_mib, 'eligible': len(tasks), 'written': sum(written.values()),
        'written_by_family': dict(written), 'skipped_over_size_by_family': dict(skipped_size),
        'unlabeled_members': unlabeled, 'errors': dict(errors),
        'family_name_tokens_dropped': dropped_tokens, 'reports_with_family_name_in_events': leaky_event_reports,
        'removed_fields': list(LEAKY_KEYS), 'output_sha256': digest.hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == '__main__':
    main()
