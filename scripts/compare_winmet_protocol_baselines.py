"""Train protocol baselines on the exact validation-only WinMET experiment.

No held-out test metric is computed. All models use the frozen train vocabulary,
same optimizer, family objective, epoch budget and calibration partition. This
is a clustered weak-label pilot, not independent dataset qualification.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from compare_ember_attention_baseline import _metrics, _train_neural_baseline
from compare_winmet_attention_baseline import _file_sha256, _fingerprint
from malware_hybrid.training import (
    _family_training_priors, _training_loss_weights, load_cape_reports,
    load_checkpoint, research_split,
)


NEURAL_BASELINES = (
    "bag_of_tokens", "bilstm", "temporal_transformer", "graph_attention", "sequence_graph",
)


def validate_partitions(experiment, partitions, strategy):
    if strategy != experiment["split_strategy"]:
        raise ValueError("split strategy differs from attention experiment")
    seen = set()
    for name, rows in partitions.items():
        expected = experiment["partitions"][name]
        if (not rows or len(rows) != expected["rows"]
                or _fingerprint(rows) != expected["sample_ids_sha256"]):
            raise ValueError(f"partition fingerprint/size differs: {name}")
        identities = {report.sample_hash for report in rows}
        if len(identities) != len(rows) or seen.intersection(identities):
            raise ValueError(f"duplicate/overlapping sample identities: {name}")
        seen.update(identities)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="Existing matched attention comparison JSON")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=NEURAL_BASELINES,
                        default=["temporal_transformer", "graph_attention", "sequence_graph"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("refusing to overwrite a result")
    if len(set(args.models)) != len(args.models):
        parser.error("duplicate baseline names")
    experiment = json.loads(args.experiment.read_text())
    source = Path(experiment["source"])
    checkpoint = Path(experiment["attention_checkpoint"])
    manifest = json.loads(Path(str(source) + ".manifest.json").read_text())
    if (_file_sha256(source) != experiment["source_sha256"]
            or manifest != experiment["source_manifest"]):
        parser.error("prepared data or sidecar differs from attention experiment")
    torch.set_num_threads(2)
    reports = load_cape_reports(
        source, max_reports=experiment["sample_cap"], seed=experiment["sample_seed"],
    )
    train, validation, calibration, withheld, strategy = research_split(reports, experiment["split_seed"])
    partitions = dict(train=train, validation=validation, calibration=calibration,
                      withheld_internal_test=withheld)
    validate_partitions(experiment, partitions, strategy)
    attention, tensorizer = load_checkpoint(checkpoint, args.device)
    if (asdict(attention.config) != experiment["model_config"]
            or asdict(tensorizer.config) != experiment["tensorizer_config"]
            or _family_training_priors(tensorizer, train) != attention.family_training_priors):
        parser.error("checkpoint config, tensorizer or priors differ from attention experiment")
    if not tensorizer.signature_policy_compatible:
        parser.error("checkpoint uses an incompatible signature policy")
    device = torch.device(args.device)
    weights = _training_loss_weights(tensorizer, train, device)
    attention_metrics = _metrics(
        attention, validation, tensorizer, device, experiment["batch_size"], weights,
        attention.family_thresholds,
    )
    result = {
        "notice": "Validation only, same clustered sample/split and weak AVClass labels; not deployment qualification.",
        "source_experiment_sha256": _file_sha256(args.experiment),
        "source_sha256": experiment["source_sha256"],
        "attention_checkpoint_sha256": _file_sha256(checkpoint),
        "seed": experiment["seed"],
        "partitions": experiment["partitions"],
        "split_strategy": strategy,
        "family_ranking_weight": experiment["family_ranking_weight"],
        "epochs": experiment["epochs"],
        "batch_size": experiment["batch_size"],
        "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-4},
        "attention_validation": attention_metrics,
        "baselines": {},
        "test_metrics": None,
        "behavior_head_trained": False,
    }
    # The reserved partition is fingerprinted above but never encoded/evaluated.
    del attention, reports, partitions, withheld
    for name in args.models:
        print(json.dumps({"phase": "training_protocol_baseline", "model": name}), flush=True)
        model, weights, thresholds, threshold_report, epoch, history = _train_neural_baseline(
            name, tensorizer_config_model(experiment), tensorizer, train, validation, calibration,
            epochs=experiment["epochs"], batch_size=experiment["batch_size"],
            seed=experiment["seed"], device=device,
            family_ranking_weight=experiment["family_ranking_weight"],
        )
        metrics = _metrics(model, validation, tensorizer, device, experiment["batch_size"], weights, thresholds)
        result["baselines"][name] = {
            "selected_epoch": epoch, "history": history, "family_thresholds": threshold_report,
            "validation": metrics,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }
        print(json.dumps({"complete_model": name, "family_map": metrics["family_map"],
                          "macro_f1": metrics["family_calibrated_macro_f1"]}), flush=True)
        del model
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({"complete": True, "report": str(args.report)}), flush=True)


def tensorizer_config_model(experiment):
    from malware_hybrid.model import HybridAttentionConfig
    return HybridAttentionConfig(**experiment["model_config"])


if __name__ == "__main__":
    main()
