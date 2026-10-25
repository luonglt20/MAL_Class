"""Validation-only family comparison on a bounded CAPE/WinMET cohort.

The prepared source is a bounded diagnostic sample, not a full-volume WinMET
benchmark. This runner reservoir-samples it deterministically and keeps the
internal test partition completely unevaluated.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from compare_ember_attention_baseline import (
    _fingerprint,
    _metrics,
    _train_bag_baseline,
)
from malware_hybrid.training import (
    _family_training_priors,
    _training_loss_weights,
    load_cape_reports,
    load_checkpoint,
    research_split,
    set_seed,
    train_model,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split_summary(reports):
    supervised = [report for report in reports if report.family_supervised]
    families = Counter(
        family for report in supervised for family in report.families
    )
    return {
        "rows": len(reports),
        "sample_ids_sha256": _fingerprint(reports),
        "family_supervised_rows": len(supervised),
        "families": dict(sorted(families.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, help="Prepared CAPE/WinMET JSONL with sidecar manifest")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-reports", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--family-min-support", type=int, default=5)
    parser.add_argument("--family-ranking-weight", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest_path = Path(str(args.reports) + ".manifest.json")
    if not args.reports.is_file() or not manifest_path.is_file():
        parser.error("prepared JSONL and its sidecar manifest are required")
    for path in (args.checkpoint, args.report):
        if path.exists():
            parser.error(f"refusing to overwrite existing output: {path}")
    if (args.max_reports < 100 or args.epochs < 1 or args.family_min_support < 1
            or not math.isfinite(args.family_ranking_weight) or args.family_ranking_weight < 0):
        parser.error("invalid sample, epoch, support or ranking-weight setting")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") not in {None, "malware-hybrid-jsonl-v1"}:
        parser.error("unsupported prepared-report manifest format")
    if not manifest.get("verification") or manifest.get("written", 0) < args.max_reports:
        parser.error("manifest does not verify a sufficiently large prepared dataset")
    torch.set_num_threads(2)
    report_sha256 = _file_sha256(args.reports)

    def loading_progress(state):
        if state["processed"] % 200 == 0 or state["processed"] == manifest["written"]:
            print(json.dumps({"loading": state}), flush=True)

    reports = load_cape_reports(
        args.reports, max_reports=args.max_reports, seed=args.seed,
        progress=loading_progress,
    )
    train, validation, calibration, withheld, split_strategy = research_split(
        reports, args.seed,
    )
    if min(map(len, (train, validation, calibration, withheld))) == 0:
        parser.error("sample is too small for disjoint train/validation/calibration/test partitions")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "notice": (
            "Validation-only diagnostic on a bounded WinMET sample. "
            "Archive-prefix sampling is order-biased; random-solid-block sampling is clustered. "
            "Neither the internal test partition nor any official external test set was evaluated. "
            "This is not a representative benchmark."
        ),
        "source": str(args.reports),
        "source_sha256": report_sha256,
        "source_manifest": manifest,
        "seed": args.seed,
        "prepared_sampling_method": manifest.get("sampling", "unspecified"),
        "sample_method": "deterministic reservoir from the verified prepared WinMET JSONL",
        "sample_cap": args.max_reports,
        "split_strategy": split_strategy,
        "family_min_support": args.family_min_support,
        "family_ranking_weight": args.family_ranking_weight,
        "d_model": args.d_model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "family_head_variant": "labelwise",
        "partitions": {
            "train": _split_summary(train),
            "validation": _split_summary(validation),
            "calibration": _split_summary(calibration),
            "withheld_internal_test": {
                "rows": len(withheld),
                "sample_ids_sha256": _fingerprint(withheld),
            },
        },
    }
    print(json.dumps({
        "phase": "training_attention",
        "partitions": {key: value["rows"] for key, value in result["partitions"].items()},
        "labeled": {key: value.get("family_supervised_rows") for key, value in result["partitions"].items()},
        "source_sha256": report_sha256,
    }), flush=True)

    def training_progress(state):
        print(json.dumps({"attention_training": state}), file=sys.stderr, flush=True)

    attention_run = train_model(
        train, validation, args.checkpoint,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        device=args.device, d_model=args.d_model,
        family_min_support=args.family_min_support, behavior_min_support=2,
        calibration_reports=calibration, test_reports=None,
        train_behavior_head=False, family_head_variant="labelwise",
        family_ranking_weight=args.family_ranking_weight,
        selection_metric="family_map", progress=training_progress,
    )
    attention, tensorizer = load_checkpoint(args.checkpoint, args.device)
    if _family_training_priors(tensorizer, train) != attention.family_training_priors:
        raise ValueError("attention checkpoint family priors do not match this reservoir/split")
    weights = _training_loss_weights(tensorizer, train, torch.device(args.device))
    family_ids = [tensorizer.family_vocab.decode(i) for i in range(len(tensorizer.family_vocab))]
    attention_validation = _metrics(
        attention, validation, tensorizer, torch.device(args.device), args.batch_size,
        weights, attention.family_thresholds,
    )

    print(json.dumps({"phase": "training_bag_baseline", "family_labels": len(family_ids)}), flush=True)
    set_seed(args.seed)
    bag, bag_weights, bag_thresholds, bag_threshold_report, bag_epoch, bag_history = _train_bag_baseline(
        attention.config, tensorizer, train, validation, calibration,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        device=torch.device(args.device),
    )
    bag_validation = _metrics(
        bag, validation, tensorizer, torch.device(args.device), args.batch_size,
        bag_weights, bag_thresholds,
    )
    result.update({
        "attention_checkpoint": str(args.checkpoint),
        "attention_selected_epoch": attention_run["selected_epoch"],
        "attention_validation_history": attention_run["history"],
        "attention_validation": attention_validation,
        "bag_selected_epoch": bag_epoch,
        "bag_validation_history": bag_history,
        "bag_family_thresholds": bag_threshold_report,
        "bag_of_tokens_validation": bag_validation,
        "test_metrics": None,
        "behavior_head_trained": attention_run["behavior_head_trained"],
    })
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({
        "complete": True,
        "attention": {
            "family_map": attention_validation["family_map"],
            "macro_f1": attention_validation["family_calibrated_macro_f1"],
            "top1": attention_validation["family_top1_accuracy"],
            "top3": attention_validation["family_top3_accuracy"],
        },
        "bag_of_tokens": {
            "family_map": bag_validation["family_map"],
            "macro_f1": bag_validation["family_calibrated_macro_f1"],
            "top1": bag_validation["family_top1_accuracy"],
            "top3": bag_validation["family_top3_accuracy"],
        },
        "report": str(args.report),
    }), flush=True)


if __name__ == "__main__":
    main()
