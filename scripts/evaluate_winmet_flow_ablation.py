"""Measure how a frozen WinMET family-attention model responds to flow ablations.

This is a validation-only counterfactual experiment. It reconstructs the exact
group-isolated validation split from the prepared JSONL and saved experiment
manifest, applies one flow/modality ablation at a time, and never evaluates the
withheld test partition or changes model weights/calibration.
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
from malware_hybrid.evaluation import ENCODED_ABLATIONS, ablate_encoded_batch, ablate_report
from malware_hybrid.training import (
    _family_training_priors,
    _training_loss_weights,
    evaluate,
    load_cape_reports,
    load_checkpoint,
    research_split,
)


ABLATIONS = (
    "disable_cross_branch_attention",
    "shuffle_events",
    "reverse_events",
    "remove_timestamps",
    "break_targets",
    "remove_graph_edges",
    "static_only",
    "dynamic_only",
    *ENCODED_ABLATIONS,
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


def _family_metrics(
    model, reports, tensorizer, device, batch_size, weights,
    *, disable_cross_branch_attention: bool = False,
    encoded_ablation: str | None = None,
):
    loader = DataLoader(
        HybridDataset(reports, tensorizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )
    family_ids = [
        tensorizer.family_vocab.decode(index)
        for index in range(model.config.n_families)
    ]
    if encoded_ablation:
        source_loader = loader
        loader = (ablate_encoded_batch(batch, encoded_ablation) for batch in source_loader)
    metrics = evaluate(
        model,
        loader,
        device,
        family_ids,
        weights[0],
        weights[1],
        weights[2],
        behavior_head_trained=bool(getattr(model, "behavior_head_trained", False)),
        family_thresholds=getattr(model, "family_thresholds", None),
        disable_cross_branch_attention=disable_cross_branch_attention,
    )
    return {
        key: value for key, value in metrics.items()
        if key.startswith("family_") or key in {"per_family", "per_family_calibrated"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, help="Verified prepared WinMET JSONL")
    parser.add_argument("checkpoint", type=Path, help="Frozen WinMET family-attention checkpoint")
    parser.add_argument("source_report", type=Path, help="Matched training report with partition fingerprints")
    parser.add_argument("--report", type=Path, required=True, help="New ablation report; never overwritten")
    parser.add_argument(
        "--modes", nargs="+", choices=ABLATIONS, default=list(ABLATIONS),
        help="ablation modes to run; choose disable_cross_branch_attention for the direct cross-attention test",
    )
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
        args.reports.resolve(), args.checkpoint.resolve(),
        args.source_report.resolve(), manifest_path.resolve(),
    }:
        parser.error("result path must be distinct from every input")

    with args.source_report.open("r", encoding="utf-8") as source:
        experiment = json.load(source)
    with manifest_path.open("r", encoding="utf-8") as source:
        prepared_manifest = json.load(source)
    expected_source = Path(experiment.get("source", "")).resolve()
    if expected_source != args.reports.resolve():
        parser.error("prepared JSONL path differs from the source training report")
    if experiment.get("source_sha256") != _sha256_file(args.reports):
        parser.error("prepared JSONL checksum differs from the source training report")
    if experiment.get("source_manifest", {}).get("written") != prepared_manifest.get("written"):
        parser.error("prepared JSONL sidecar manifest does not match the training report")
    if Path(experiment.get("attention_checkpoint", "")).resolve() != args.checkpoint.resolve():
        parser.error("checkpoint path differs from the source training report")

    torch.set_num_threads(2)
    # Older experiment reports use one seed for sampling, splitting and model
    # training. New repeated-seed comparisons keep data/splits fixed while
    # varying model initialization.
    seed = int(experiment.get("sample_seed", experiment["seed"]))
    split_seed = int(experiment.get("split_seed", experiment["seed"]))
    sample_cap = int(experiment["sample_cap"])

    def loading_progress(state):
        if state["processed"] % 200 == 0 or state["processed"] == prepared_manifest["written"]:
            print(json.dumps({"loading": state}), file=sys.stderr, flush=True)

    reports = load_cape_reports(
        args.reports,
        max_reports=sample_cap,
        seed=seed,
        progress=loading_progress,
    )
    if len(reports) != sample_cap:
        parser.error("reconstructed WinMET sample size differs from the experiment report")
    train, validation, calibration, withheld, split_strategy = research_split(reports, split_seed)
    if split_strategy != experiment.get("split_strategy"):
        parser.error("reconstructed split strategy differs from the experiment report")
    partitions = {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "withheld_internal_test": withheld,
    }
    expected_fingerprints = {
        name: experiment["partitions"][name]["sample_ids_sha256"]
        for name in partitions
    }
    actual_fingerprints = {name: _fingerprint(rows) for name, rows in partitions.items()}
    if actual_fingerprints != expected_fingerprints:
        parser.error("reconstructed split fingerprints differ; refusing flow ablation")

    model, tensorizer = load_checkpoint(args.checkpoint, args.device)
    if not model.config.signature_branch_enabled:
        parser.error("cross-branch ablation requires the three-branch checkpoint")
    if model.config.family_head_variant != experiment.get("family_head_variant"):
        parser.error("checkpoint family-head variant differs from the source experiment")
    if _family_training_priors(tensorizer, train) != model.family_training_priors:
        parser.error("checkpoint training priors differ from the reconstructed training split")

    device = torch.device(args.device)
    model.to(device).eval()
    weights = _training_loss_weights(tensorizer, train, device)
    baseline = _family_metrics(model, validation, tensorizer, device, args.batch_size, weights)
    ablation_results = {}
    for mode in args.modes:
        if mode == "disable_cross_branch_attention":
            metrics = _family_metrics(
                model, validation, tensorizer, device, args.batch_size, weights,
                disable_cross_branch_attention=True,
            )
        elif mode in ENCODED_ABLATIONS:
            metrics = _family_metrics(
                model, validation, tensorizer, device, args.batch_size, weights,
                encoded_ablation=mode,
            )
        else:
            counterfactuals = [
                ablate_report(report, mode, seed=seed) for report in validation
            ]
            metrics = _family_metrics(
                model, counterfactuals, tensorizer, device, args.batch_size, weights
            )
        ablation_results[mode] = {
            "metrics": metrics,
            "delta_vs_original": {
                key: (
                    float(metrics[key]) - float(baseline[key])
                    if isinstance(metrics.get(key), (int, float))
                    and isinstance(baseline.get(key), (int, float))
                    else None
                )
                for key in (
                    "family_map",
                    "family_calibrated_macro_f1",
                    "family_top1_accuracy",
                    "family_top3_accuracy",
                )
            },
        }

    result = {
        "notice": (
            f"Validation-only counterfactuals on a clustered {len(reports):,}-report WinMET sample. "
            "AVClass family labels are weak supervision. Counterfactual metric changes are "
            "diagnostic, not causal proof or an independent benchmark. The withheld test split "
            "was reconstructed for integrity checks but never evaluated."
        ),
        "source": str(args.reports),
        "source_sha256": _sha256_file(args.reports),
        "source_report": str(args.source_report),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "seed": seed,
        "model_seed": int(experiment["seed"]),
        "device": args.device,
        "batch_size": args.batch_size,
        "sample_seed": seed,
        "split_seed": split_seed,
        "modality_dropout_scope": experiment.get(
            "modality_dropout_scope", "summary_only_legacy"
        ),
        "split_strategy": split_strategy,
        "partitions": {
            name: {"rows": len(rows), "sample_ids_sha256": actual_fingerprints[name]}
            for name, rows in partitions.items()
        },
        "evaluated_partition": "validation",
        "validation_family_supervised_rows": sum(report.family_supervised for report in validation),
        "baseline": baseline,
        "ablations": ablation_results,
        "test_metrics": None,
        "ablation_interpretation": {
            "break_targets": "Historical joint perturbation: also changes graph size, pruning and token coverage.",
            "remove_graph_edges": "Historical report-edge deletion: pruning can remove graph nodes too.",
            "remove_resource_relations": "Zero temporal resource identities after encoding; keep graph and tokens unchanged.",
            "remove_process_context": "Zero process/parent/thread context after encoding; keep graph and tokens unchanged.",
            "mask_graph_edges_keep_nodes": "Mask GAT/path edges after encoding; keep all encoded graph nodes and masks.",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({
        "complete": True,
        "validation_rows": len(validation),
        "family_supervised_rows": result["validation_family_supervised_rows"],
        "baseline": {
            key: baseline.get(key) for key in (
                "family_map", "family_calibrated_macro_f1",
                "family_top1_accuracy", "family_top3_accuracy",
            )
        },
        "ablation_deltas": {
            mode: result["ablations"][mode]["delta_vs_original"]
            for mode in result["ablations"]
        },
        "report": str(args.report),
    }), flush=True)


if __name__ == "__main__":
    main()
