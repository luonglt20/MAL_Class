"""Train an EMBER family candidate and report validation metrics only.

The official later-time test archive is never opened by this script. The
checksum-locked EMBER training archive is split into train/validation/
calibration/internal-test partitions; the internal test is withheld as well.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch

from malware_hybrid.dataset_sources import verify_dataset_file
from malware_hybrid.training import load_ember_win32_archive, research_split, train_model


def _split_summary(reports):
    supervised = [report for report in reports if report.family_supervised]
    weak_reference = [
        report for report in reports if report.families and not report.family_supervised
    ]
    counts = Counter(
        family for report in supervised for family in report.families
    )
    return {
        "rows": len(reports),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(report.sample_hash for report in reports)).encode()
        ).hexdigest(),
        "family_supervised_rows": len(supervised),
        "retained_low_confidence_rows": len(weak_reference),
        "family_positive_counts": dict(sorted(counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Checksum-locked EMBER2024 Win32_train.zip")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=7000)
    parser.add_argument("--rows-per-week", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family-min-support", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    for path in (args.checkpoint, args.report):
        if path.exists():
            parser.error(f"refusing to overwrite existing output: {path}")
    if args.archive.name != "Win32_train.zip":
        parser.error("only the pinned Win32_train.zip is accepted; the test archive is forbidden")
    if args.max_samples < 100 or args.rows_per_week < 1 or args.rows_per_week > args.max_samples:
        parser.error("require max-samples >= 100 and 1 <= rows-per-week <= max-samples")

    torch.set_num_threads(2)
    verification = verify_dataset_file(args.archive, dataset_name="ember2024_win32_train")
    with zipfile.ZipFile(args.archive) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".jsonl")]
        if not members or any(not name.lower().endswith("_train.jsonl") for name in members):
            parser.error("archive contains a non-training JSONL member")

    reports = load_ember_win32_archive(
        args.archive, rows_per_week=args.rows_per_week,
        max_samples=args.max_samples, seed=args.seed,
    )
    train, validation, calibration, withheld, strategy = research_split(reports, args.seed)
    if not train or not validation or not calibration or not withheld:
        parser.error("sample is too small for four disjoint partitions")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "notice": "Validation-only research run. Official and internal test partitions were not evaluated.",
        "dataset_verification": verification,
        "seed": args.seed,
        "epochs": args.epochs,
        "device": args.device,
        "family_min_support": args.family_min_support,
        "family_head_variant": "labelwise",
        "selection_metric": "family_map",
        "split_strategy": strategy,
        "partitions": {
            "train": _split_summary(train),
            "validation": _split_summary(validation),
            "calibration": _split_summary(calibration),
            "withheld_internal_test": {
                "rows": len(withheld),
                "sample_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(report.sample_hash for report in withheld)).encode()
                ).hexdigest(),
            },
        },
    }
    partition_summary = {
        name: {
            "rows": partition["rows"],
            "family_supervised_rows": partition["family_supervised_rows"],
            "retained_low_confidence_rows": partition["retained_low_confidence_rows"],
            "families": len(partition["family_positive_counts"]),
        }
        for name, partition in result["partitions"].items()
        if name != "withheld_internal_test"
    }
    partition_summary["withheld_internal_test"] = result["partitions"]["withheld_internal_test"]
    print(json.dumps({"phase": "training", "partitions": partition_summary}), flush=True)

    def progress(state):
        print(json.dumps({"training": state}), file=sys.stderr, flush=True)

    run = train_model(
        train, validation, args.checkpoint,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        device=args.device, d_model=args.d_model,
        family_min_support=args.family_min_support, behavior_min_support=2,
        calibration_reports=calibration, test_reports=None,
        train_behavior_head=False, family_head_variant="labelwise",
        selection_metric="family_map", progress=progress,
    )
    selected = next(
        (row for row in run["history"] if row.get("epoch") == run["selected_epoch"]),
        None,
    )
    result.update({
        "checkpoint": str(args.checkpoint),
        "selected_epoch": run["selected_epoch"],
        "best_validation": selected,
        "validation_history": run["history"],
        "family_thresholds": run["family_thresholds"],
        "test_metrics": None,
    })
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({
        "complete": True,
        "selected_epoch": run["selected_epoch"],
        "validation_family_map": selected.get("family_map") if selected else None,
        "validation_family_macro_f1": selected.get("family_macro_f1") if selected else None,
        "checkpoint": str(args.checkpoint),
        "report": str(args.report),
    }), flush=True)


if __name__ == "__main__":
    main()
