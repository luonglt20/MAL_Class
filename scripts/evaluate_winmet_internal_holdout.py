"""One-time frozen family evaluation on a prepared WinMET block holdout.

The evaluator reconstructs the exact sample and group-isolated split recorded
by a matched comparison run. It evaluates only the reserved internal holdout;
it does not fit model weights or thresholds and does not open malware binaries.
WinMET family labels are AVClass consensus weak labels.
"""

from __future__ import annotations

import argparse
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


def _fingerprint(reports) -> str:
    value = "\n".join(sorted(report.sample_hash for report in reports))
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, help="Prepared WinMET JSONL used by the experiment")
    parser.add_argument("checkpoint", type=Path, help="Frozen three-branch checkpoint")
    parser.add_argument("source_report", type=Path, help="Matched comparison report with split fingerprints")
    parser.add_argument("--report", type=Path, required=True, help="New result JSON; never overwritten")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest_path = Path(str(args.reports) + ".manifest.json")
    if not args.reports.is_file() or not manifest_path.is_file():
        parser.error("prepared JSONL and its sidecar manifest are required")
    for path in (args.checkpoint, args.source_report):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    if args.report.exists():
        parser.error(f"refusing to overwrite existing result: {args.report}")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.report.resolve() in {
        args.reports.resolve(), args.checkpoint.resolve(), args.source_report.resolve(),
        manifest_path.resolve(),
    }:
        parser.error("result path must be distinct from every input")

    experiment = json.loads(args.source_report.read_text(encoding="utf-8"))
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    if experiment.get("checkpoint_sha256") and experiment["checkpoint_sha256"] != checkpoint_sha256:
        parser.error("checkpoint checksum differs from source experiment")
    prepared_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(experiment.get("source", "")).resolve() != args.reports.resolve():
        parser.error("prepared JSONL path differs from the source experiment")
    if experiment.get("source_sha256") != _sha256_file(args.reports):
        parser.error("prepared JSONL checksum differs from the source experiment")
    if experiment.get("source_manifest", {}).get("written") != prepared_manifest.get("written"):
        parser.error("prepared JSONL sidecar does not match the source experiment")
    if Path(experiment.get("attention_checkpoint", "")).resolve() != args.checkpoint.resolve():
        parser.error("checkpoint path differs from the source experiment")
    sample_cap = int(experiment["sample_cap"])
    if sample_cap < 1 or sample_cap > int(prepared_manifest.get("written", 0)):
        parser.error("source experiment sample cap is outside the prepared dataset")

    torch.set_num_threads(2)
    sample_seed = int(experiment.get("sample_seed", experiment["seed"]))
    split_seed = int(experiment.get("split_seed", experiment["seed"]))

    def loading_progress(state):
        if state["processed"] % 200 == 0 or state["processed"] == sample_cap:
            print(json.dumps({"loading": state}), file=sys.stderr, flush=True)

    reports = load_cape_reports(
        args.reports, max_reports=sample_cap, seed=sample_seed,
        progress=loading_progress,
    )
    if len(reports) != sample_cap:
        parser.error("reconstructed WinMET sample size differs from the source experiment")
    train, validation, calibration, withheld, split_strategy = research_split(reports, split_seed)
    partitions = {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "withheld_internal_test": withheld,
    }
    if split_strategy != experiment.get("split_strategy"):
        parser.error("reconstructed split strategy differs from the source experiment")
    actual_fingerprints = {
        name: _fingerprint(rows) for name, rows in partitions.items()
    }
    expected_fingerprints = {
        name: experiment["partitions"][name]["sample_ids_sha256"]
        for name in partitions
    }
    if actual_fingerprints != expected_fingerprints:
        parser.error("reconstructed partition fingerprints differ; refusing evaluation")
    if not train or not validation or not calibration or not withheld:
        parser.error("one or more reconstructed partitions are empty")

    model, tensorizer = load_checkpoint(args.checkpoint, args.device)
    if not model.config.signature_branch_enabled or not tensorizer.config.signature_branch_enabled:
        parser.error("checkpoint does not use the three-branch architecture")
    if model.config.family_head_variant != experiment.get("family_head_variant"):
        parser.error("checkpoint family-head variant differs from the source experiment")
    if _family_training_priors(tensorizer, train) != model.family_training_priors:
        parser.error("checkpoint training priors differ from the reconstructed train partition")

    device = torch.device(args.device)
    model.to(device).eval()
    family_weights, behavior_weights, malware_weights = _training_loss_weights(
        tensorizer, train, device,
    )
    family_labels = [
        tensorizer.family_vocab.decode(index)
        for index in range(model.config.n_families)
    ]
    loader = DataLoader(
        HybridDataset(withheld, tensorizer),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )
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

    result = {
        "notice": (
            "One-time frozen evaluation on the pre-reserved group-isolated internal holdout from a "
            "bounded WinMET sample. Family targets are AVClass consensus weak labels. "
            "No weights or thresholds were fit or changed; the official later-time EMBER test was "
            "not opened. This clustered cohort is not a representative full-volume benchmark."
        ),
        "source": str(args.reports),
        "source_sha256": _sha256_file(args.reports),
        "source_manifest": str(manifest_path),
        "source_experiment": str(args.source_report),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_hash_verified_against_source_report": bool(experiment.get("checkpoint_sha256")),
        "seed": int(experiment["seed"]),
        "sample_seed": sample_seed,
        "split_seed": split_seed,
        "split_strategy": split_strategy,
        "prepared_sampling_method": prepared_manifest.get("sampling"),
        "prepared_block_count": prepared_manifest.get("selected_block_count"),
        "partitions": {
            name: {
                "rows": len(rows),
                "sample_ids_sha256": actual_fingerprints[name],
                "family_supervised_rows": sum(report.family_supervised for report in rows),
            }
            for name, rows in partitions.items()
        },
        "evaluated_partition": "withheld_internal_test",
        "metrics": metrics,
        "weights_changed": False,
        "thresholds_changed": False,
        "behavior_head_trained": bool(getattr(model, "behavior_head_trained", False)),
        "official_test_opened": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({
        "complete": True,
        "evaluated_rows": len(withheld),
        "family_supervised_rows": result["partitions"]["withheld_internal_test"]["family_supervised_rows"],
        "family_map": metrics.get("family_map"),
        "family_calibrated_macro_f1": metrics.get("family_calibrated_macro_f1"),
        "family_top1_accuracy": metrics.get("family_top1_accuracy"),
        "family_top3_accuracy": metrics.get("family_top3_accuracy"),
        "report": str(args.report),
    }), flush=True)


if __name__ == "__main__":
    main()
