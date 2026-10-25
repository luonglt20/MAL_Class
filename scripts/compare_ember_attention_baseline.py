"""Matched validation-only family comparison of static attention and bag evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import torch
from torch.utils.data import DataLoader

from malware_hybrid.baselines import BagOfTokensBaseline
from malware_hybrid.dataset import HybridDataset, collate_samples
from malware_hybrid.dataset_sources import verify_dataset_file
from malware_hybrid.training import (
    _family_training_priors,
    _move,
    _snapshot_state,
    _training_loss_weights,
    evaluate,
    fit_family_thresholds,
    load_checkpoint,
    load_ember_win32_archive,
    research_split,
    set_seed,
)


def _fingerprint(reports):
    value = "\n".join(sorted(report.sample_hash for report in reports))
    return hashlib.sha256(value.encode()).hexdigest()


def _metrics(model, reports, tensorizer, device, batch_size, weights, thresholds):
    loader = DataLoader(
        HybridDataset(reports, tensorizer), batch_size=batch_size,
        shuffle=False, collate_fn=collate_samples,
    )
    family_ids = [tensorizer.family_vocab.decode(i) for i in range(len(tensorizer.family_vocab))]
    metrics = evaluate(
        model, loader, device, family_ids,
        weights[0], weights[1], weights[2],
        behavior_head_trained=False, family_thresholds=thresholds,
    )
    # This experiment compares family ranking/classification only. In
    # particular, the bag baseline's malware head is not trained, so reporting
    # its apparent malware FPR/recall would be meaningless.
    return {key: value for key, value in metrics.items() if key.startswith("family_")}


def _train_bag_baseline(
    config, tensorizer, train, validation, calibration, *,
    epochs, batch_size, seed, device,
):
    set_seed(seed)
    model = BagOfTokensBaseline(config).to(device)
    model.family_training_priors = _family_training_priors(tensorizer, train)
    model.behavior_head_trained = False
    model.malware_head_trained = False
    weights = _training_loss_weights(tensorizer, train, device)
    loader = DataLoader(
        HybridDataset(train, tensorizer), batch_size=batch_size,
        shuffle=True, collate_fn=collate_samples,
    )
    family_ids = [tensorizer.family_vocab.decode(i) for i in range(len(tensorizer.family_vocab))]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_map = float("-inf")
    best_state = None
    best_epoch = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_index, batch in enumerate(loader, start=1):
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch, return_attention=False)
            losses = model.compute_loss(
                output, batch,
                family_pos_weight=weights[0], behavior_pos_weight=weights[1],
                malware_pos_weight=weights[2], family_ranking_weight=0.0,
            )
            loss = losses["family_loss"] + losses["malware_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.detach())
            if batch_index % max(1, len(loader) // 10) == 0 or batch_index == len(loader):
                print(json.dumps({
                    "model": "bag_of_tokens",
                    "epoch": epoch,
                    "epochs": epochs,
                    "batch": batch_index,
                    "batches": len(loader),
                    "train_loss": running_loss / batch_index,
                }), file=sys.stderr, flush=True)

        validation_metrics = _metrics(
            model, validation, tensorizer, device, batch_size, weights, thresholds={}
        )
        validation_metrics["epoch"] = epoch
        validation_metrics["train_loss"] = running_loss / max(1, len(loader))
        history.append(validation_metrics)
        score = validation_metrics["family_map"]
        score = -float("inf") if score is None else float(score)
        updated = score > best_map
        if updated:
            best_map = score
            best_state = _snapshot_state(model)
            best_epoch = epoch
        print(json.dumps({
            "model": "bag_of_tokens",
            "epoch_end": epoch,
            "validation_family_map": validation_metrics["family_map"],
            "validation_family_macro_f1": validation_metrics["family_macro_f1"],
            "selected_epoch": best_epoch,
            "checkpoint_updated": updated,
        }), flush=True)

    if best_state is None:
        raise ValueError("bag baseline produced no finite validation family mAP")
    model.load_state_dict(best_state)
    threshold_report = fit_family_thresholds(
        model, tensorizer, calibration, device, batch_size=batch_size,
    )
    thresholds = {label: float(value["threshold"]) for label, value in threshold_report.items()}
    model.family_thresholds = thresholds
    return model, weights, thresholds, threshold_report, best_epoch, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Checksum-locked EMBER2024 Win32_train.zip")
    parser.add_argument("--attention-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=7000)
    parser.add_argument("--rows-per-week", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.report.exists():
        parser.error(f"refusing to overwrite existing report: {args.report}")
    if args.archive.name != "Win32_train.zip":
        parser.error("only the pinned Win32_train.zip is accepted; the test archive is forbidden")
    if args.max_samples < 100 or not 1 <= args.rows_per_week <= args.max_samples:
        parser.error("require max-samples >= 100 and 1 <= rows-per-week <= max-samples")

    torch.set_num_threads(2)
    verification = verify_dataset_file(args.archive, dataset_name="ember2024_win32_train")
    with zipfile.ZipFile(args.archive) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".jsonl")]
        if not members or any(not name.lower().endswith("_train.jsonl") for name in members):
            parser.error("archive contains a non-training JSONL member")

    attention, tensorizer = load_checkpoint(args.attention_checkpoint, args.device)
    if attention.config.family_head_variant != "labelwise":
        parser.error("attention checkpoint must use the labelwise family head for this comparison")
    trainable_reports = load_ember_win32_archive(
        args.archive, rows_per_week=args.rows_per_week,
        max_samples=args.max_samples, seed=args.seed,
    )
    train, validation, calibration, withheld, strategy = research_split(
        trainable_reports, args.seed,
    )
    if not train or not validation or not calibration or not withheld:
        parser.error("sample is too small for four disjoint partitions")

    train_priors = _family_training_priors(tensorizer, train)
    if train_priors != attention.family_training_priors:
        parser.error("sample/split family priors differ from the attention checkpoint")

    set_seed(args.seed)
    attention = attention.to(args.device).eval()
    attention_weights = _training_loss_weights(
        tensorizer, train, torch.device(args.device),
    )
    attention_thresholds = attention.family_thresholds
    attention_validation = _metrics(
        attention, validation, tensorizer, torch.device(args.device), args.batch_size,
        attention_weights, attention_thresholds,
    )

    print(json.dumps({
        "phase": "training_bag_baseline",
        "partitions": {
            "train": len(train), "validation": len(validation),
            "calibration": len(calibration), "withheld_internal_test": len(withheld),
        },
        "split_fingerprints": {
            "train": _fingerprint(train),
            "validation": _fingerprint(validation),
            "calibration": _fingerprint(calibration),
            "withheld_internal_test": _fingerprint(withheld),
        },
    }), flush=True)

    bag, bag_weights, bag_thresholds, bag_threshold_report, bag_epoch, bag_history = _train_bag_baseline(
        attention.config, tensorizer, train, validation, calibration,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        device=torch.device(args.device),
    )
    bag_validation = _metrics(
        bag, validation, tensorizer, torch.device(args.device), args.batch_size,
        bag_weights, bag_thresholds,
    )
    result = {
        "notice": (
            "Matched validation-only static comparison. The training archive was checksum-verified; "
            "the internal test partition and official test archive were not evaluated. "
            "Only family metrics are reported; malware/behavior metrics are out of scope."
        ),
        "dataset_verification": verification,
        "sample_cap": args.max_samples,
        "rows_per_week": args.rows_per_week,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "family_head_variant": attention.config.family_head_variant,
        "split_strategy": strategy,
        "split_fingerprints": {
            "train": _fingerprint(train),
            "validation": _fingerprint(validation),
            "calibration": _fingerprint(calibration),
            "withheld_internal_test": _fingerprint(withheld),
        },
        "attention_checkpoint": str(args.attention_checkpoint),
        "attention_validation": attention_validation,
        "bag_of_tokens_validation": bag_validation,
        "bag_selected_epoch": bag_epoch,
        "bag_validation_history": bag_history,
        "bag_family_thresholds": bag_threshold_report,
        "test_metrics": None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({
        "complete": True,
        "attention": {
            "family_map": attention_validation["family_map"],
            "family_macro_f1": attention_validation["family_calibrated_macro_f1"],
            "family_top1": attention_validation["family_top1_accuracy"],
            "family_top3": attention_validation["family_top3_accuracy"],
        },
        "bag_of_tokens": {
            "family_map": bag_validation["family_map"],
            "family_macro_f1": bag_validation["family_calibrated_macro_f1"],
            "family_top1": bag_validation["family_top1_accuracy"],
            "family_top3": bag_validation["family_top3_accuracy"],
        },
        "report": str(args.report),
    }), flush=True)


if __name__ == "__main__":
    main()
