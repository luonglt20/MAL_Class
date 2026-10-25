"""Recalibrate a trained EMBER checkpoint with simultaneous family conformal sets.

This only reads the checksum-locked Win32 training archive. The internal test
and official later-time test partitions are intentionally not evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch

from malware_hybrid.dataset_sources import verify_dataset_file
from malware_hybrid.training import (
    _family_training_priors,
    fit_family_conformal,
    load_checkpoint,
    load_ember_win32_archive,
    research_split,
    save_checkpoint,
)


def _fingerprint(reports) -> str:
    value = "\n".join(sorted(report.sample_hash for report in reports))
    return hashlib.sha256(value.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Checksum-locked EMBER2024 Win32_train.zip")
    parser.add_argument("checkpoint", type=Path, help="Previously trained candidate checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New calibrated checkpoint; never overwritten")
    parser.add_argument("--report", type=Path, required=True, help="Calibration provenance JSON; never overwritten")
    parser.add_argument("--max-samples", type=int, default=40_000)
    parser.add_argument("--rows-per-week", type=int, default=650)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.archive.name != "Win32_train.zip":
        parser.error("only the pinned Win32_train.zip is accepted; test archives are forbidden")
    if args.max_samples < 100 or not 1 <= args.rows_per_week <= args.max_samples:
        parser.error("require max-samples >= 100 and 1 <= rows-per-week <= max-samples")
    if args.output.exists() or args.report.exists():
        parser.error("refusing to overwrite an existing checkpoint or report")
    if args.output.resolve() == args.checkpoint.resolve():
        parser.error("output must be a new path, not the source checkpoint")

    torch.set_num_threads(2)
    verification = verify_dataset_file(args.archive, dataset_name="ember2024_win32_train")
    with zipfile.ZipFile(args.archive) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".jsonl")]
        if not members or any(not name.lower().endswith("_train.jsonl") for name in members):
            parser.error("archive contains a non-training JSONL member")

    model, tensorizer = load_checkpoint(args.checkpoint, args.device)
    raw_payload = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    reports = load_ember_win32_archive(
        args.archive,
        rows_per_week=args.rows_per_week,
        max_samples=args.max_samples,
        seed=args.seed,
        progress=lambda state: print(
            json.dumps({"loading": state}), file=sys.stderr, flush=True
        ),
    )
    train, validation, calibration, withheld, strategy = research_split(reports, args.seed)
    if not train or not validation or not calibration or not withheld:
        parser.error("sample is too small for four disjoint partitions")
    if _family_training_priors(tensorizer, train) != model.family_training_priors:
        parser.error("reconstructed training sample/split does not match checkpoint priors")

    conformal = fit_family_conformal(
        model, tensorizer, calibration, torch.device(args.device),
        batch_size=args.batch_size,
    )
    if conformal is None or conformal.joint_threshold() is None:
        parser.error("calibration partition did not produce simultaneous family scores")

    save_checkpoint(
        args.output,
        model,
        tensorizer,
        history=raw_payload.get("history", []),
        family_conformal=conformal,
        malware_calibration=raw_payload.get("malware_calibration"),
        family_thresholds=raw_payload.get("family_thresholds", {}),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "notice": (
            "Calibration-only update using the original training archive sample. "
            "No model weights were trained or changed; validation, internal test, and official test were not evaluated."
        ),
        "dataset_verification": verification,
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(args.output),
        "seed": args.seed,
        "sample_cap": args.max_samples,
        "rows_per_week": args.rows_per_week,
        "split_strategy": strategy,
        "partitions": {
            "train": {"rows": len(train), "fingerprint": _fingerprint(train)},
            "validation": {"rows": len(validation), "fingerprint": _fingerprint(validation)},
            "calibration": {"rows": len(calibration), "fingerprint": _fingerprint(calibration)},
            "withheld_internal_test": {"rows": len(withheld), "fingerprint": _fingerprint(withheld)},
        },
        "conformal": {
            "method": "simultaneous_multilabel_split_conformal",
            "alpha": conformal.alpha,
            "calibration_rows": len(conformal.joint_scores or []),
            "joint_nonconformity_threshold": conformal.joint_threshold(),
        },
        "weights_changed": False,
        "test_metrics": None,
    }
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps({
        "complete": True,
        "output_checkpoint": str(args.output),
        "report": str(args.report),
        "calibration_rows": len(conformal.joint_scores or []),
        "joint_threshold": conformal.joint_threshold(),
        "test_metrics": None,
    }), flush=True)


if __name__ == "__main__":
    main()
