"""Aggregate matched validation-only WinMET attention/bag seed reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, stdev


PARTITIONS = ("train", "validation", "calibration", "withheld_internal_test")
METRICS = {
    "family_map": "family_map",
    "calibrated_macro_f1": "family_calibrated_macro_f1",
    "top1_accuracy": "family_top1_accuracy",
    "top3_accuracy": "family_top3_accuracy",
}


def _experiment_identity(report: dict) -> dict:
    seed = int(report["seed"])
    return {
        "source_sha256": report["source_sha256"],
        "sample_cap": int(report["sample_cap"]),
        "sample_seed": int(report.get("sample_seed", seed)),
        "split_seed": int(report.get("split_seed", seed)),
        "split_strategy": report["split_strategy"],
        "family_min_support": int(report["family_min_support"]),
        "family_ranking_weight": float(report["family_ranking_weight"]),
        # Older reports predate the explicit field; the model config default
        # at that time was 0.15, so recover the effective setting.
        "modality_dropout": float(report.get("modality_dropout", 0.15)),
        "modality_dropout_scope": report.get(
            "modality_dropout_scope", "summary_only_legacy"
        ),
        "d_model": int(report["d_model"]),
        "batch_size": int(report["batch_size"]),
        "epochs": int(report["epochs"]),
        "family_head_variant": report["family_head_variant"],
        "architecture": report.get("architecture", "unspecified"),
        "signature_static_attention_enabled": report.get("model_config", {}).get(
            "signature_static_attention_enabled", False
        ),
        "yara_input_policy": report.get("tensorizer_config", {}).get(
            "yara_input_policy", "legacy_unverified_yara_v0"
        ),
        "partition_fingerprints": {
            name: report["partitions"][name]["sample_ids_sha256"]
            for name in PARTITIONS
        },
    }


def summarize_reports(reports: list[dict]) -> dict:
    """Validate matched reports, then return mean/sample-SD and paired deltas."""
    if len(reports) < 2:
        raise ValueError("at least two seed reports are required")
    seeds = [int(report["seed"]) for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("model seeds must be unique")

    identity = _experiment_identity(reports[0])
    for report in reports:
        if _experiment_identity(report) != identity:
            raise ValueError("reports differ in source, split, labels, or model settings")
        if report.get("test_metrics") is not None:
            raise ValueError("refusing to aggregate reports that evaluated a test partition")
        if "attention_validation" not in report or "bag_of_tokens_validation" not in report:
            raise ValueError("every report must contain matched attention and bag validation metrics")

    def metric_values(model_key: str, metric_key: str) -> list[float]:
        values = [report[model_key][metric_key] for report in reports]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise ValueError(f"metric {metric_key} must be finite for every seed")
        return [float(value) for value in values]

    def summarize_values(values: list[float]) -> dict:
        return {"mean": fmean(values), "sample_sd": stdev(values), "per_seed": values}

    result_metrics = {"attention": {}, "bag": {}, "paired_delta_attention_minus_bag": {}}
    for output_name, input_name in METRICS.items():
        attention = metric_values("attention_validation", input_name)
        bag = metric_values("bag_of_tokens_validation", input_name)
        deltas = [left - right for left, right in zip(attention, bag)]
        result_metrics["attention"][output_name] = summarize_values(attention)
        result_metrics["bag"][output_name] = summarize_values(bag)
        result_metrics["paired_delta_attention_minus_bag"][output_name] = {
            **summarize_values(deltas),
            "attention_wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "bag_wins": sum(delta < 0 for delta in deltas),
        }

    return {
        "notice": (
            "Matched validation-only seed summary on one fixed, clustered WinMET sample/split. "
            "AVClass family labels are weak supervision; seeds are not independent data samples, "
            "and this result is not a deployable-quality benchmark. No test metrics were used."
        ),
        "model_seeds": seeds,
        "seed_count": len(seeds),
        "experiment_identity": identity,
        "metrics": result_metrics,
        "test_metrics": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Matched validation-only seed JSON reports")
    parser.add_argument("--output", required=True, type=Path, help="New summary JSON path; never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
        summary = summarize_reports(reports)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps({
        "complete": True,
        "seed_count": summary["seed_count"],
        "model_seeds": summary["model_seeds"],
        "summary": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
