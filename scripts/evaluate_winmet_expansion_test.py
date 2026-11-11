"""One-time frozen evaluation of a selected expansion candidate on the withheld test.

The readout (model head or frozen-embedding kNN) and its abstention threshold
must be chosen beforehand on validation/calibration and passed in explicitly;
this script fits nothing on test. It refuses to run if a result or lock for the
same checkpoint already exists, so the independent test is scored once.
Reports hybrid and PE-only (events removed) separately, with label scope,
denominators and abstention coverage. Labels are weak AVClass consensus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_winmet_flow_ablation import _sha256_file  # noqa: E402
from explore_stopped_checkpoint import collect, prototype_scores, selective, summary  # noqa: E402
from train_winmet_expansion import load_expansion  # noqa: E402
from malware_hybrid.training import _family_training_priors, load_checkpoint  # noqa: E402

TARGET = 0.85


def score(readout, train, part, n_families):
    if readout == "model_sigmoid":
        return part[0]
    k = int(readout.removeprefix("prototype_knn"))
    return prototype_scores(train[1], train[2], part[1], n_families, k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("training_report", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--selection-report", type=Path, required=True,
                        help="Validation/calibration report that fixed the readout and thresholds")
    parser.add_argument("--hybrid-readout", required=True)
    parser.add_argument("--hybrid-threshold", type=float, required=True)
    parser.add_argument("--pe-readout", required=True)
    parser.add_argument("--pe-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-partition", default="withheld_internal_test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    lock = args.checkpoint.with_suffix(f".{args.test_partition}.evaluated.lock")
    if args.output.exists() or lock.exists():
        raise SystemExit("test already evaluated for this checkpoint; refusing to rescore")
    for readout in (args.hybrid_readout, args.pe_readout):
        if readout != "model_sigmoid" and not readout.startswith("prototype_knn"):
            raise SystemExit(f"unknown readout {readout}")
    if _sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("checkpoint checksum mismatch")
    if _sha256_file(args.manifest) != args.manifest_sha256:
        raise SystemExit("manifest checksum mismatch")
    manifest = json.loads(args.manifest.read_text())
    training = json.loads(args.training_report.read_text())
    result = training.get("result") or {}
    if (training.get("manifest") != manifest or result.get("epochs_completed") != result.get("epochs_requested")
            or not result.get("conformal")
            or Path(result.get("checkpoint", "")).resolve() != args.checkpoint.resolve()):
        raise SystemExit("completed, calibrated training report for this checkpoint required")
    selection = json.loads(args.selection_report.read_text())
    if selection.get("checkpoint_sha256") != args.checkpoint_sha256 or selection.get("test_metrics") is not None:
        raise SystemExit("selection report must be validation-only for the same checkpoint")
    lock.write_text(json.dumps({"output": str(args.output), "checkpoint_sha256": args.checkpoint_sha256}))
    torch.set_num_threads(4)
    reports = load_expansion(manifest.get("source"), manifest, include_test=args.test_partition)
    model, tensorizer = load_checkpoint(args.checkpoint, args.device)
    if _family_training_priors(tensorizer, reports["train"]) != model.family_training_priors:
        raise SystemExit("model priors differ from preserved training partition")
    device = torch.device(args.device)
    views = {"hybrid": ({}, args.hybrid_readout, args.hybrid_threshold),
             "pe_only_events_removed": ({"pe_only": True}, args.pe_readout, args.pe_threshold)}
    output = {"status": "one_time_independent_test", "checkpoint_sha256": args.checkpoint_sha256,
              "manifest_sha256": args.manifest_sha256, "source_sha256": manifest.get("source_sha256") or manifest.get("sources"),
              "training_report_sha256": hashlib.sha256(args.training_report.read_bytes()).hexdigest(),
              "selection_report_sha256": hashlib.sha256(args.selection_report.read_bytes()).hexdigest(),
              "test_partition_name": args.test_partition,
              "test_partition": manifest["partitions"][args.test_partition],
              "label_scope": f"{len(tensorizer.family_vocab)} families in model vocabulary (train support >= 5)",
              "target_top1": TARGET, "views": {}}
    for name, (flags, readout, threshold) in views.items():
        needs_train = readout != "model_sigmoid"
        train = collect(model, tensorizer, reports["train"], device, args.batch_size, **flags) if needs_train else None
        test = collect(model, tensorizer, reports[args.test_partition], device, args.batch_size, **flags)
        scores = score(readout, train, test, test[0].shape[1])
        entry = summary(scores, test[2])
        entry.update(readout=readout, frozen_threshold=threshold,
                     at_frozen_threshold=selective(scores.max(1), scores, test[2], threshold),
                     full_coverage_top1_meets_target=bool(entry["top1"] > TARGET))
        output["views"][name] = entry
    output["notice"] = ("Independent group-isolated test scored once with readout and thresholds fixed on "
                        "validation/calibration. top1/top3 denominators are test rows whose family is in the "
                        "model vocabulary; out-of-vocabulary and unlabelled rows are reported separately.")
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({name: {k: v for k, v in e.items() if k in ("top1", "top3", "eligible_known_family", "rows")}
                      for name, e in output["views"].items()}))


if __name__ == "__main__":
    main()
