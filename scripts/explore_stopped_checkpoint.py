"""Exploratory, validation-only analysis of a stopped (uncalibrated) WinMET checkpoint.

Reports hybrid and PE-only (events removed) results separately with the label
scope, denominators and abstention coverage, and tests a frozen-embedding
nearest-prototype readout fitted on the train partition only (no gradient
updates). Thresholds are chosen on calibration and only read on validation.
The withheld internal test partition is never constructed or scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_winmet_flow_ablation import _sha256_file  # noqa: E402
from train_winmet_expansion import load_expansion  # noqa: E402
from malware_hybrid.dataset import HybridDataset, collate_samples  # noqa: E402
from malware_hybrid.evaluation import ablate_report  # noqa: E402
from malware_hybrid.prototype_readout import prototype_scores as _shared_prototype_scores  # noqa: E402
from malware_hybrid.training import _family_training_priors, _move, load_checkpoint  # noqa: E402

TARGET = 0.85


def collect(model, tensorizer, reports, device, batch_size, *, pe_only=False, no_cross=False):
    if pe_only:
        reports = [ablate_report(report, "static_only") for report in reports]
    captured = []

    # Pooled Signature/Static/Dynamic branch summaries fed to the verdict query.
    def grab(module, args):
        captured.append(args[1].detach().flatten(1).cpu())
    hook = model.malware_attention.register_forward_pre_hook(grab)
    loader = DataLoader(HybridDataset(reports, tensorizer), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_samples)
    scores = []
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = _move(batch, device)
                output = model(batch, return_attention=False,
                               **({"disable_cross_branch_attention": True} if no_cross else {}))
                scores.append(torch.sigmoid(output["family_logits"]).float().cpu())
    finally:
        hook.remove()
    vocab = tensorizer.family_vocab.token_to_id
    labels = []
    for report in reports:
        known = [vocab[f] for f in report.families if f in vocab]
        labels.append({"supervised": bool(report.family_supervised and report.families),
                       "known": known, "families": report.families})
    return torch.cat(scores).numpy(), torch.cat(captured).numpy(), labels


def split_rows(labels):
    eligible = np.array([bool(item["known"]) for item in labels])
    unknown_family = np.array([item["supervised"] and not item["known"] for item in labels])
    unlabeled = np.array([not item["supervised"] for item in labels])
    return eligible, unknown_family, unlabeled


def correct_top1(scores, labels):
    top = scores.argmax(1)
    return np.array([bool(item["known"]) and int(top[i]) in item["known"] for i, item in enumerate(labels)])


def selective(confidence, scores, labels, threshold):
    eligible, unknown_family, unlabeled = split_rows(labels)
    accept = confidence >= threshold
    correct = correct_top1(scores, labels)
    accepted_eligible = accept & eligible
    return {
        "threshold": float(threshold),
        "eligible_accepted": int(accepted_eligible.sum()), "eligible_total": int(eligible.sum()),
        "eligible_coverage": float(accepted_eligible.sum() / max(1, eligible.sum())),
        "selective_top1": float(correct[accepted_eligible].mean()) if accepted_eligible.any() else None,
        "unknown_family_accepted": int((accept & unknown_family).sum()),
        "unknown_family_total": int(unknown_family.sum()),
        "unlabeled_accepted": int((accept & unlabeled).sum()), "unlabeled_total": int(unlabeled.sum()),
        "abstention_rate_all_rows": float(1 - accept.mean()),
    }


def summary(scores, labels):
    eligible, unknown_family, unlabeled = split_rows(labels)
    correct = correct_top1(scores, labels)
    top3 = np.argsort(-scores, 1)[:, :3]
    hit3 = np.array([bool(set(top3[i]) & set(item["known"])) for i, item in enumerate(labels)])
    return {"rows": len(labels), "eligible_known_family": int(eligible.sum()),
            "unknown_family_supervised": int(unknown_family.sum()), "unlabeled": int(unlabeled.sum()),
            "top1": float(correct[eligible].mean()), "top3": float(hit3[eligible].mean()),
            "families_in_eligible_rows": len({f for item in labels for f in item["families"] if item["known"]})}


def fit_threshold(confidence, scores, labels):
    """Lowest calibration threshold whose selective top-1 reaches TARGET."""
    for threshold in np.unique(np.round(confidence, 4)):
        result = selective(confidence, scores, labels, threshold)
        if result["selective_top1"] is not None and result["selective_top1"] >= TARGET and result["eligible_accepted"] >= 10:
            return float(threshold)
    return None


def prototype_scores(train_embed, train_labels, query, n_families, k):
    """kNN vote over train rows with a family in the vocabulary (shared with the UI readout)."""
    rows = [i for i, item in enumerate(train_labels) if item["known"]]
    return _shared_prototype_scores(train_embed[rows], [train_labels[i]["known"] for i in rows], query, n_families, k)


def evaluate_view(name, model, tensorizer, reports, device, batch_size, ks, **flags):
    print(json.dumps({"view": name}), flush=True)
    data = {part: collect(model, tensorizer, reports[part], device, batch_size, **flags)
            for part in ("train", "calibration", "validation")}
    n_families = data["validation"][0].shape[1]
    readouts = {"model_sigmoid": {part: data[part][0] for part in ("calibration", "validation")}}
    for k in ks:
        readouts[f"prototype_knn{k}"] = {
            part: prototype_scores(data["train"][1], data["train"][2], data[part][1], n_families, k)
            for part in ("calibration", "validation")}
    result = {}
    for readout, parts in readouts.items():
        entry = {part: summary(parts[part], data[part][2]) for part in parts}
        conf_cal = parts["calibration"].max(1)
        threshold = fit_threshold(conf_cal, parts["calibration"], data["calibration"][2])
        entry["threshold_fit_on"] = "calibration"
        entry["calibration_threshold"] = threshold
        entry["validation_at_calibration_threshold"] = (
            selective(parts["validation"].max(1), parts["validation"], data["validation"][2], threshold)
            if threshold is not None else None)
        conf_val = parts["validation"].max(1)
        entry["validation_curve"] = [selective(conf_val, parts["validation"], data["validation"][2], q)
                                     for q in np.quantile(conf_val, [0, .25, .5, .75, .9])]
        result[readout] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--knn", type=int, nargs="*", default=[1, 5, 15])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    if _sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("checkpoint checksum mismatch")
    if _sha256_file(args.manifest) != args.manifest_sha256:
        raise SystemExit("manifest checksum mismatch")
    manifest = json.loads(args.manifest.read_text())
    torch.set_num_threads(4)
    reports = load_expansion(manifest.get("source"), manifest)  # verifies source + partition fingerprints
    model, tensorizer = load_checkpoint(args.checkpoint, args.device)
    if _family_training_priors(tensorizer, reports["train"]) != model.family_training_priors:
        raise SystemExit("model priors differ from preserved training partition")
    device = torch.device(args.device)
    views = {
        "hybrid_full": {},
        "hybrid_no_cross_attention": {"no_cross": True},
        "pe_only_events_removed": {"pe_only": True},
    }
    result = {
        "status": "exploratory_validation_uncalibrated_checkpoint",
        "checkpoint_sha256": args.checkpoint_sha256, "manifest_sha256": args.manifest_sha256,
        "source_sha256": manifest.get("source_sha256") or manifest.get("sources"),
        "partition_fingerprints": {k: v["sample_ids_sha256"] for k, v in manifest["partitions"].items()},
        "selected_epoch": getattr(model, "selected_epoch", None),
        "model_family_labels": len(tensorizer.family_vocab),
        "target_top1": TARGET,
        "views": {name: evaluate_view(name, model, tensorizer, reports, device, args.batch_size, args.knn, **flags)
                  for name, flags in views.items()},
        "test_metrics": None,
        "notice": ("Weak AVClass labels. top1/top3 use rows whose family is in the model vocabulary. "
                   "Selective top-1 is conditional on acceptance and must be read with coverage. "
                   "Prototype readouts use frozen embeddings and train-partition labels only; no weights changed. "
                   "Withheld internal test not constructed."),
    }
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"written": str(args.output)}))


if __name__ == "__main__":
    main()
