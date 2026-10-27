"""Frozen evaluation on disjoint WinMET blocks or a different archive volume."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

from malware_hybrid.dataset import HybridDataset, collate_samples
from malware_hybrid.training import (
    _family_training_priors,
    _training_loss_weights,
    evaluate,
    load_cape_reports,
    load_checkpoint,
    research_split,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(reports) -> str:
    value = "\n".join(sorted(report.sample_hash for report in reports))
    return hashlib.sha256(value.encode()).hexdigest()


def _archive_sha(manifest: dict, filename: str) -> str | None:
    for item in manifest.get("verification", []):
        if Path(item.get("path", "")).name == filename:
            return item.get("sha256")
    return None


def validate_block_provenance(source: dict, external: dict) -> bool:
    """Require explicit exclusions within one archive, or verified cross-volume provenance."""
    source_blocks = set(source.get("selected_block_ids") or [])
    external_blocks = set(external.get("selected_block_ids") or [])
    excluded = set(external.get("excluded_block_ids") or [])
    if not source_blocks or not external_blocks:
        raise ValueError("source and external manifests must record selected solid-block IDs")
    if source_blocks & external_blocks or excluded & external_blocks:
        raise ValueError("external block IDs overlap source/excluded blocks")
    if external.get("excluded_block_ids") != sorted(excluded):
        raise ValueError("external exclusion list is not canonical")

    archives = []
    for manifest in (source, external):
        filename = Path(manifest.get("source", "")).name
        entry = next((item for item in manifest.get("verification", [])
                      if Path(item.get("path", "")).name == filename), {})
        if not entry.get("sha256") or not entry.get("record"):
            raise ValueError("source and external manifests must verify their source archives")
        archives.append(entry)
    if archives[0]["record"] != archives[1]["record"]:
        raise ValueError("source and external archives do not belong to the same pinned WinMET record")
    source_labels = _archive_sha(source, "reports_consensus_label.json")
    external_labels = _archive_sha(external, "reports_consensus_label.json")
    if not source_labels or source_labels != external_labels:
        raise ValueError("source and external cohorts must verify the same AVClass label file")
    same_archive = archives[0]["sha256"] == archives[1]["sha256"]
    if same_archive and not source_blocks.issubset(excluded):
        raise ValueError("same-archive external sampler did not exclude every source-experiment block")
    return same_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, help="Newly prepared block-disjoint WinMET JSONL")
    parser.add_argument("checkpoint", type=Path, help="Frozen three-branch checkpoint")
    parser.add_argument("source_experiment", type=Path, help="Matched training report for the checkpoint")
    parser.add_argument("--report", type=Path, required=True, help="New result JSON; never overwritten")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    external_manifest_path = Path(str(args.reports) + ".manifest.json")
    for path in (args.reports, external_manifest_path, args.checkpoint, args.source_experiment):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    source_experiment = json.loads(args.source_experiment.read_text(encoding="utf-8"))
    if args.report.exists():
        parser.error(f"refusing to overwrite existing result: {args.report}")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.report.resolve() in {
        args.reports.resolve(), args.checkpoint.resolve(), args.source_experiment.resolve(),
        external_manifest_path.resolve(),
    }:
        parser.error("result path must be distinct from every input")

    external_manifest = json.loads(external_manifest_path.read_text(encoding="utf-8"))
    source_path = Path(source_experiment.get("source", ""))
    source_manifest_path = Path(str(source_path) + ".manifest.json")
    if not source_path.is_file() or not source_manifest_path.is_file():
        parser.error("original training JSONL and its sidecar manifest are required")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_experiment.get("source_sha256") != _sha256_file(source_path):
        parser.error("training JSONL checksum differs from the matched experiment")
    if Path(source_experiment.get("attention_checkpoint", "")).resolve() != args.checkpoint.resolve():
        parser.error("checkpoint path differs from the matched experiment")
    if external_manifest.get("sampling") != "random-solid-blocks":
        parser.error("external cohort must use random-solid-blocks sampling")
    if external_manifest.get("written") != external_manifest.get("selected"):
        parser.error("external cohort contains invalid or missing reports")

    source_blocks = set(source_manifest.get("selected_block_ids") or [])
    external_blocks = set(external_manifest.get("selected_block_ids") or [])
    excluded_blocks = set(external_manifest.get("excluded_block_ids") or [])
    try:
        same_archive = validate_block_provenance(source_manifest, external_manifest)
    except ValueError as error:
        parser.error(str(error))

    external_archive_filename = Path(external_manifest.get("source", "")).name
    source_archive_filename = Path(source_manifest.get("source", "")).name
    source_archive_sha = _archive_sha(source_manifest, source_archive_filename)
    external_archive_sha = _archive_sha(external_manifest, external_archive_filename)
    expected_source_manifest = source_experiment.get("source_manifest", {})
    if source_manifest.get("written") != expected_source_manifest.get("written"):
        parser.error("training JSONL sidecar does not match the source experiment")
    if source_manifest.get("selected_block_ids") != expected_source_manifest.get("selected_block_ids"):
        parser.error("training sidecar block provenance differs from the source experiment")
    if source_manifest.get("verification") != expected_source_manifest.get("verification"):
        parser.error("training sidecar checksum evidence differs from the source experiment")

    torch.set_num_threads(2)

    def loading_progress(name):
        def report_progress(state):
            print(json.dumps({"loading": name, **state}), file=sys.stderr, flush=True)
        return report_progress

    source_reports = load_cape_reports(
        source_path,
        max_reports=int(source_experiment["sample_cap"]),
        seed=int(source_experiment.get("sample_seed", source_experiment["seed"])),
        progress=loading_progress("training_reference"),
    )
    train, validation, calibration, withheld, split_strategy = research_split(
        source_reports,
        int(source_experiment.get("split_seed", source_experiment["seed"])),
    )
    partitions = {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "withheld_internal_test": withheld,
    }
    for name, rows in partitions.items():
        expected = source_experiment["partitions"][name]["sample_ids_sha256"]
        if _fingerprint(rows) != expected:
            parser.error(f"reconstructed source partition differs: {name}")
    if split_strategy != source_experiment.get("split_strategy"):
        parser.error("reconstructed source split strategy differs")

    external_reports = load_cape_reports(
        args.reports, progress=loading_progress("external_cohort"),
    )
    if len(external_reports) != int(external_manifest["written"]):
        parser.error("external JSONL row count differs from its manifest")
    training_hashes = {report.sample_hash for report in source_reports}
    external_hashes = [report.sample_hash for report in external_reports]
    if len(set(external_hashes)) != len(external_hashes):
        parser.error("external cohort contains duplicate sample hashes")
    if training_hashes.intersection(external_hashes):
        parser.error("external cohort contains a sample hash present in the training experiment")

    model, tensorizer = load_checkpoint(args.checkpoint, args.device)
    if not model.config.signature_branch_enabled or not tensorizer.config.signature_branch_enabled:
        parser.error("checkpoint does not use the three-branch architecture")
    if model.config.family_head_variant != source_experiment.get("family_head_variant"):
        parser.error("checkpoint family-head variant differs from the source experiment")
    if _family_training_priors(tensorizer, train) != model.family_training_priors:
        parser.error("checkpoint training priors differ from the reconstructed source train split")

    model.to(args.device).eval()
    family_weights, behavior_weights, malware_weights = _training_loss_weights(
        tensorizer, train, torch.device(args.device),
    )
    family_labels = [
        tensorizer.family_vocab.decode(index)
        for index in range(model.config.n_families)
    ]
    known_labels = set(tensorizer.family_vocab.token_to_id)
    external_family_counts = Counter(
        label for sample in external_reports if sample.family_supervised
        for label in sample.families
    )
    family_supervised_rows = sum(sample.family_supervised for sample in external_reports)
    known_label_rows = sum(
        sample.family_supervised and any(label in known_labels for label in sample.families)
        for sample in external_reports
    )
    unknown_family_counts = {
        label: count for label, count in sorted(external_family_counts.items())
        if label not in known_labels
    }
    loader = DataLoader(
        HybridDataset(external_reports, tensorizer),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )
    metrics = evaluate(
        model,
        loader,
        torch.device(args.device),
        family_labels,
        family_weights,
        behavior_weights,
        malware_weights,
        behavior_head_trained=bool(getattr(model, "behavior_head_trained", False)),
        family_thresholds=getattr(model, "family_thresholds", None),
    )
    attention_disabled_metrics = evaluate(
        model,
        loader,
        torch.device(args.device),
        family_labels,
        family_weights,
        behavior_weights,
        malware_weights,
        behavior_head_trained=bool(getattr(model, "behavior_head_trained", False)),
        family_thresholds=getattr(model, "family_thresholds", None),
        disable_cross_branch_attention=True,
    )
    attention_metric_deltas = {
        key: metrics.get(key) - attention_disabled_metrics.get(key)
        for key in (
            "family_map", "family_calibrated_macro_f1", "family_top1_accuracy",
            "family_top3_accuracy",
        )
        if metrics.get(key) is not None and attention_disabled_metrics.get(key) is not None
    }
    result = {
        "notice": (
            "One-time frozen evaluation on WinMET reports with disjoint sample hashes and solid-block IDs. "
            "Different archive volumes provide cross-volume validation within WinMET, not an independent "
            "dataset or temporal benchmark; same-volume evaluation requires explicit block exclusions. "
            "AVClass consensus family labels are weak labels. No weights or thresholds were fitted or changed."
        ),
        "source_experiment": str(args.source_experiment),
        "source_experiment_sha256": _sha256_file(args.source_experiment),
        "training_jsonl": str(source_path),
        "training_jsonl_sha256": _sha256_file(source_path),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "external_jsonl": str(args.reports),
        "external_jsonl_sha256": _sha256_file(args.reports),
        "external_manifest": str(external_manifest_path),
        "source_archive_filename": source_archive_filename,
        "source_archive_sha256": source_archive_sha,
        "external_archive_filename": external_archive_filename,
        "external_archive_sha256": external_archive_sha,
        "same_archive": same_archive,
        "cross_volume_evaluation": not same_archive,
        "label_file_sha256": _archive_sha(external_manifest, "reports_consensus_label.json"),
        "source_block_count": len(source_blocks),
        "external_block_count": len(external_blocks),
        "excluded_block_count": len(excluded_blocks),
        "external_block_ids": sorted(external_blocks),
        "external_sample_ids_sha256": _fingerprint(external_reports),
        "external_rows": len(external_reports),
        "family_supervised_rows": family_supervised_rows,
        "known_family_label_rows": known_label_rows,
        "known_family_label_row_coverage": known_label_rows / max(1, family_supervised_rows),
        "known_family_labels_present": sorted(set(external_family_counts) & known_labels),
        "unseen_family_label_counts": unknown_family_counts,
        "metrics": metrics,
        "metrics_cross_branch_attention_disabled": attention_disabled_metrics,
        "full_minus_attention_disabled_metric_delta": attention_metric_deltas,
        "attention_ablation_notice": (
            "Frozen-weight counterfactual on the same external cohort and fixed calibration thresholds. "
            "No retraining or threshold fitting; reports the total contribution of cross-branch attention, "
            "not a causal or human-validated explanation."
        ),
        "weights_changed": False,
        "thresholds_changed": False,
        "behavior_head_trained": bool(getattr(model, "behavior_head_trained", False)),
        "sample_hash_disjoint": True,
        "solid_block_ids_disjoint": True,
        "same_archive_block_disjoint": True if same_archive else None,
        "external_dataset_test": False,
        "official_test_opened": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({
        "complete": True,
        "external_rows": len(external_reports),
        "family_supervised_rows": family_supervised_rows,
        "known_family_label_rows": known_label_rows,
        "family_map": metrics.get("family_map"),
        "family_calibrated_macro_f1": metrics.get("family_calibrated_macro_f1"),
        "family_top1_accuracy": metrics.get("family_top1_accuracy"),
        "family_top3_accuracy": metrics.get("family_top3_accuracy"),
        "attention_disabled_family_map": attention_disabled_metrics.get("family_map"),
        "attention_disabled_calibrated_macro_f1": attention_disabled_metrics.get(
            "family_calibrated_macro_f1"
        ),
        "full_minus_attention_disabled": attention_metric_deltas,
        "report": str(args.report),
    }), flush=True)


if __name__ == "__main__":
    main()
