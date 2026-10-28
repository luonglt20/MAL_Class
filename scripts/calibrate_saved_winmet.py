"""Finalize uncertainty for a checksum-pinned saved candidate; never train.

Reconstruct the verified original split, calibrate on calibration only and
audit user-visible validation decisions. Test metrics remain unopened.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from evaluate_winmet_flow_ablation import _fingerprint, _sha256_file, _family_metrics
from malware_hybrid.deployment_audit import audit_family_decisions
from malware_hybrid.model import HybridAttentionConfig
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.training import (
    _family_training_priors, _training_loss_weights, fit_family_conformal,
    fit_family_thresholds, load_cape_reports, load_checkpoint, research_split,
    save_checkpoint,
)


def validate_saved_candidate(model, tensorizer, reference):
    if asdict(model.config) != asdict(HybridAttentionConfig.from_saved(reference["model_config"])):
        raise ValueError("saved candidate architecture differs from reference experiment")
    if asdict(tensorizer.config) != reference["tensorizer_config"]:
        raise ValueError("saved tensorizer differs from reference experiment")
    history = model.training_history or []
    if model.training_selection_metric != "family_map" or not history:
        raise ValueError("saved candidate lacks validation-mAP selection provenance")
    if any(not isinstance(row.get("family_map"), (int, float))
           or not math.isfinite(row["family_map"]) for row in history):
        raise ValueError("saved validation history contains missing or nonfinite mAP")
    if model.selected_epoch != max(history, key=lambda row: row["family_map"])["epoch"]:
        raise ValueError("selected checkpoint epoch disagrees with saved validation history")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    paths = [args.reports, args.checkpoint, args.reference, args.output, args.report,
             Path(str(args.reports) + ".manifest.json")]
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error("all source/output paths must be distinct")
    if args.output.exists() or args.report.exists() or args.batch_size < 1:
        parser.error("outputs must not exist and batch size must be positive")
    if _sha256_file(args.checkpoint) != args.checkpoint_sha256:
        parser.error("saved checkpoint checksum mismatch")
    reference = json.loads(args.reference.read_text())
    if (Path(reference["source"]).resolve() != args.reports.resolve()
            or reference["source_sha256"] != _sha256_file(args.reports)):
        parser.error("prepared dataset differs from frozen reference")
    manifest = json.loads(paths[-1].read_text())
    if manifest != reference["source_manifest"]:
        parser.error("prepared sidecar differs from frozen reference")
    torch.set_num_threads(2)
    reports = load_cape_reports(args.reports, max_reports=reference["sample_cap"],
                                seed=reference["sample_seed"])
    train, validation, calibration, withheld, strategy = research_split(reports, reference["split_seed"])
    partitions = dict(zip(("train", "validation", "calibration", "withheld_internal_test"),
                          (train, validation, calibration, withheld)))
    if strategy != reference["split_strategy"]:
        parser.error("split strategy differs")
    for name, rows in partitions.items():
        saved = reference["partitions"][name]
        if len(rows) != saved["rows"] or _fingerprint(rows) != saved["sample_ids_sha256"]:
            parser.error(f"partition differs: {name}")
    model, tensorizer = load_checkpoint(args.checkpoint, args.device)
    validate_saved_candidate(model, tensorizer, reference)
    if _family_training_priors(tensorizer, train) != model.family_training_priors:
        parser.error("training priors differ")
    if sorted(set(row.metadata.get("normalization_version") for row in train)) != model.normalization_versions:
        parser.error("normalization provenance differs")
    original_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    device = torch.device(args.device)
    print(json.dumps({"phase": "calibrating_frozen_weights", "selected_epoch": model.selected_epoch,
                      "calibration_rows": len(calibration), "weights_trainable": False}), flush=True)
    conformal = fit_family_conformal(model, tensorizer, calibration, device, batch_size=args.batch_size)
    if conformal is None or conformal.joint_threshold() is None:
        parser.error("no simultaneous calibration scores; refusing uncalibrated output")
    thresholds = fit_family_thresholds(model, tensorizer, calibration, device, batch_size=args.batch_size)
    model.family_conformal = conformal
    model.family_thresholds = {label: float(item["threshold"]) for label, item in thresholds.items()}
    weights = _training_loss_weights(tensorizer, train, device)
    metrics = _family_metrics(model, validation, tensorizer, device, args.batch_size, weights)
    audit = audit_family_decisions(
        HybridInferencePipeline(model, tensorizer, device=device), validation,
        tensorizer.family_vocab.token_to_id,
        lambda done, total: print(json.dumps({"audit": [done, total]}), flush=True) if done % 20 == 0 or done == total else None,
    )
    if any(not torch.equal(value, model.state_dict()[key].detach().cpu()) for key, value in original_state.items()):
        raise RuntimeError("weights changed during calibration or inference")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    save_checkpoint(args.output, model, tensorizer, history=payload.get("history", []),
                    family_conformal=conformal, family_thresholds=thresholds,
                    malware_calibration=payload.get("malware_calibration"))
    saved = torch.load(args.output, map_location="cpu", weights_only=True)
    if any(not torch.equal(value, saved["model_state"][key]) for key, value in original_state.items()):
        raise RuntimeError("output checkpoint did not preserve weights")
    result = {key: reference[key] for key in (
        "source", "source_sha256", "source_manifest", "seed", "sample_seed", "split_seed",
        "sample_cap", "split_strategy", "partitions", "family_head_variant",
    )}
    result.update(
        notice="Calibration/validation only of an interrupted-training candidate; no new training, no completed matched baseline, no release qualification.",
        source_checkpoint=str(args.checkpoint), source_checkpoint_sha256=args.checkpoint_sha256,
        reference_split=str(args.reference), reference_split_sha256=_sha256_file(args.reference),
        attention_checkpoint=str(args.output), checkpoint_sha256=_sha256_file(args.output),
        checkpoint_bytes=args.output.stat().st_size, model_config=asdict(model.config),
        tensorizer_config=asdict(tensorizer.config), attention_selected_epoch=model.selected_epoch,
        saved_history_epochs=len(payload.get("history", [])), weights_changed=False,
        conformal=conformal.as_dict(), family_thresholds=thresholds,
        attention_validation=metrics, deployment_family_decisions=audit,
        behavior_head_trained=model.behavior_head_trained, test_metrics=None,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as target:
        json.dump(result, target, indent=2)
    print(json.dumps({"complete": True, "checkpoint": str(args.output),
                      "map": metrics["family_map"], "f1": metrics["family_calibrated_macro_f1"],
                      "accepted": audit["accepted_primary_samples"], "total": audit["samples"],
                      "reference_precision": audit["accepted_primary_reference_precision"]}), flush=True)


if __name__ == "__main__":
    main()
