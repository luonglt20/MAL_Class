"""One-time evaluation of a frozen family candidate on its reserved EMBER split.

The script accepts only the checksum-locked Win32 training archive, reconstructs
the exact sample and temporal split recorded by the joint-calibration manifest,
and evaluates only the previously withheld internal-test partition. It never
fits thresholds, modifies a checkpoint, or opens the official test archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch
from torch.utils.data import DataLoader

from malware_hybrid.dataset import HybridDataset, collate_samples
from malware_hybrid.dataset_sources import verify_dataset_file
from malware_hybrid.training import (
    _family_training_priors,
    _training_loss_weights,
    evaluate,
    load_checkpoint,
    load_ember_win32_archive,
    research_split,
)


def _fingerprint(reports) -> str:
    value = "\n".join(sorted(report.sample_hash for report in reports))
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Checksum-locked EMBER2024 Win32_train.zip")
    parser.add_argument("checkpoint", type=Path, help="Frozen candidate checkpoint to evaluate")
    parser.add_argument(
        "calibration_report", type=Path,
        help="Manifest that records the exact sample/split and calibrated checkpoint",
    )
    parser.add_argument("--report", type=Path, required=True, help="New result JSON; never overwritten")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.archive.name != "Win32_train.zip":
        parser.error("only the checksum-locked Win32_train.zip is accepted; test archives are forbidden")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.report.exists():
        parser.error(f"refusing to overwrite existing result: {args.report}")
    if args.report.resolve() in {args.archive.resolve(), args.checkpoint.resolve(), args.calibration_report.resolve()}:
        parser.error("result path must be distinct from every input")

    with args.calibration_report.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    if manifest.get("weights_changed") is not False or manifest.get("test_metrics") is not None:
        parser.error("calibration manifest is not an untouched calibration-only record")
    if Path(manifest.get("output_checkpoint", "")).resolve() != args.checkpoint.resolve():
        parser.error("checkpoint path does not match the calibration manifest")

    torch.set_num_threads(2)
    verification = verify_dataset_file(args.archive, dataset_name="ember2024_win32_train")
    expected_verification = manifest.get("dataset_verification", {})
    if verification.get("sha256") != expected_verification.get("sha256"):
        parser.error("training archive checksum differs from the calibration manifest")
    with zipfile.ZipFile(args.archive) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".jsonl")]
        if not members or any(not name.lower().endswith("_train.jsonl") for name in members):
            parser.error("archive contains a non-training JSONL member")

    rows_per_week = int(manifest["rows_per_week"])
    sample_cap = int(manifest["sample_cap"])
    seed = int(manifest["seed"])
    reports = load_ember_win32_archive(
        args.archive,
        rows_per_week=rows_per_week,
        max_samples=sample_cap,
        seed=seed,
        progress=lambda state: print(
            json.dumps({"loading": state}), file=sys.stderr, flush=True
        ),
    )
    train, validation, calibration, withheld, strategy = research_split(reports, seed)
    partitions = {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "withheld_internal_test": withheld,
    }
    if strategy != manifest.get("split_strategy"):
        parser.error("reconstructed split strategy differs from the calibration manifest")
    reconstructed = {
        name: {"rows": len(rows), "fingerprint": _fingerprint(rows)}
        for name, rows in partitions.items()
    }
    if reconstructed != manifest.get("partitions"):
        parser.error("reconstructed partition sizes/fingerprints differ; refusing test evaluation")
    if not train or not validation or not calibration or not withheld:
        parser.error("one or more reconstructed partitions are empty")

    model, tensorizer = load_checkpoint(args.checkpoint, args.device)
    if model.config.family_head_variant != "labelwise":
        parser.error("checkpoint does not use the family head recorded for this candidate")
    if _family_training_priors(tensorizer, train) != model.family_training_priors:
        parser.error("checkpoint training priors do not match the reconstructed training partition")

    device = torch.device(args.device)
    model.to(device).eval()
    family_weights, behavior_weights, malware_weights = _training_loss_weights(
        tensorizer, train, device,
    )
    loader = DataLoader(
        HybridDataset(withheld, tensorizer),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )
    family_labels = [
        tensorizer.family_vocab.decode(index)
        for index in range(model.config.n_families)
    ]
    metrics = evaluate(
        model,
        loader,
        device,
        family_labels,
        family_weights,
        behavior_weights,
        malware_weights,
        behavior_head_trained=bool(getattr(model, "behavior_head_trained", False)),
        family_thresholds=getattr(model, "family_thresholds", None),
    )
    metrics.pop("loss", None)

    result = {
        "notice": (
            "One-time frozen-checkpoint evaluation on the pre-reserved internal-test split from the "
            "EMBER2024 training archive. No thresholds or weights were fit or changed. The official "
            "later-time test archive was not opened. Family labels are dataset labels, not independent "
            "manual annotations; this result alone does not qualify a malware verdict for deployment."
        ),
        "dataset_verification": verification,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "calibration_manifest": str(args.calibration_report),
        "seed": seed,
        "sample_cap": sample_cap,
        "rows_per_week": rows_per_week,
        "split_strategy": strategy,
        "partitions": {
            name: {"rows": len(rows), "fingerprint": _fingerprint(rows)}
            for name, rows in partitions.items()
        },
        "evaluated_partition": "withheld_internal_test",
        "metrics": metrics,
        "test_metrics_before_this_run": None,
        "official_test_opened": False,
        "weights_changed": False,
        "thresholds_changed": False,
        "malware_head_qualified": bool(getattr(model, "malware_head_qualified", False)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({
        "complete": True,
        "evaluated_rows": len(withheld),
        "family_evaluated_samples": metrics.get("family_evaluated_samples"),
        "family_map": metrics.get("family_map"),
        "family_calibrated_macro_f1": metrics.get("family_calibrated_macro_f1"),
        "family_top1_accuracy": metrics.get("family_top1_accuracy"),
        "family_top3_accuracy": metrics.get("family_top3_accuracy"),
        "malware_false_positive_rate": metrics.get("malware_false_positive_rate"),
        "malware_fpr_wilson_95": metrics.get("malware_fpr_wilson_95"),
        "report": str(args.report),
    }), flush=True)


if __name__ == "__main__":
    main()
