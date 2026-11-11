"""Freeze the validation-selected top-1 readout for a checkpoint (no test access).

Builds train-partition prototypes when a kNN readout is chosen, fits the
abstention threshold on calibration with the same rule used in the exploratory
validation report, and writes the checksum-bound sidecar read by the UI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_stopped_checkpoint import collect, fit_threshold, prototype_scores  # noqa: E402
from train_winmet_expansion import load_expansion  # noqa: E402
from malware_hybrid.prototype_readout import save_readout, sha256_file  # noqa: E402
from malware_hybrid.training import _family_training_priors, load_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--readout", required=True, help="model_sigmoid or prototype_knn<k>")
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise SystemExit("checkpoint checksum mismatch")
    selection = json.loads(args.selection_report.read_text())
    if selection.get("checkpoint_sha256") != args.checkpoint_sha256 or selection.get("test_metrics") is not None:
        raise SystemExit("selection report must be validation-only for this checkpoint")
    manifest = json.loads(args.manifest.read_text())
    torch.set_num_threads(4)
    reports = load_expansion(manifest.get("source"), manifest)
    model, tensorizer = load_checkpoint(args.checkpoint, args.device)
    if _family_training_priors(tensorizer, reports["train"]) != model.family_training_priors:
        raise SystemExit("model priors differ from preserved training partition")
    device = torch.device(args.device)
    calibration = collect(model, tensorizer, reports["calibration"], device, args.batch_size)
    n_families = calibration[0].shape[1]
    embeddings = labels = None
    if args.readout == "model_sigmoid":
        scores = calibration[0]
    else:
        k = int(args.readout.removeprefix("prototype_knn"))
        train = collect(model, tensorizer, reports["train"], device, args.batch_size)
        scores = prototype_scores(train[1], train[2], calibration[1], n_families, k)
        rows = [i for i, item in enumerate(train[2]) if item["known"]]
        embeddings, labels = train[1][rows], [train[2][i]["known"] for i in rows]
    threshold = fit_threshold(scores.max(1), scores, calibration[2])
    if threshold is None:
        raise SystemExit("no calibration threshold reaches the target selective top-1")
    config = {"readout": args.readout, "threshold": threshold, "threshold_fit_on": "calibration",
              "label_scope": f"{n_families} families in model vocabulary",
              "selection_report_sha256": sha256_file(args.selection_report),
              "manifest_sha256": sha256_file(args.manifest)}
    print(json.dumps({"sidecar": str(save_readout(args.checkpoint, config, embeddings, labels)),
                      "threshold": threshold}))


if __name__ == "__main__":
    main()
