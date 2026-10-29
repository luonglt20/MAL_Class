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
from malware_hybrid.faithfulness import family_token_faithfulness
from malware_hybrid.deployment_audit import audit_family_decisions, explore_candidate_thresholds, select_candidate_threshold
from malware_hybrid.pipeline import HybridInferencePipeline
from malware_hybrid.pair_intervention import compare_attention_pairs
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
    "shuffle_events_keep_timestamps",
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


def _paired_family_logit_change(model, loader, device, mode):
    """Separate score sensitivity from unchanged ranking/F1 metrics."""
    if mode not in ENCODED_ABLATIONS:
        raise ValueError("paired logits require an encoded node-preserving control")
    count = samples = argmax_changes = 0
    absolute_sum = max_logit_delta = max_sigmoid_delta = 0.
    known_resources = event_count = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            original = model(batch, return_attention=False)["family_logits"]
            altered = model(ablate_encoded_batch(batch, mode), return_attention=False)["family_logits"]
            if not torch.isfinite(original).all() or not torch.isfinite(altered).all():
                raise ValueError("nonfinite family logits during paired control")
            difference = (original - altered).abs()
            absolute_sum += float(difference.sum())
            count += difference.numel()
            samples += difference.shape[0]
            argmax_changes += int((original.argmax(-1) != altered.argmax(-1)).sum())
            max_logit_delta = max(max_logit_delta, float(difference.max()))
            max_sigmoid_delta = max(max_sigmoid_delta, float((original.sigmoid() - altered.sigmoid()).abs().max()))
            event_count += int(batch["event_mask"].sum())
            known_resources += int(((batch["resource_ids"] > 0) & batch["event_mask"]).sum())
    return {"notice": "Same-sample logit changes with fixed weights; metric invariance is not proof of input insensitivity.",
            "samples": samples, "family_logit_values": count,
            "mean_absolute_logit_delta": absolute_sum / count if count else None,
            "maximum_absolute_logit_delta": max_logit_delta,
            "maximum_absolute_sigmoid_delta": max_sigmoid_delta,
            "argmax_changed_samples": argmax_changes,
            "valid_events": event_count, "events_with_known_resource_identity": known_resources}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, help="Verified prepared WinMET JSONL")
    parser.add_argument("checkpoint", type=Path, help="Frozen WinMET family-attention checkpoint")
    parser.add_argument("source_report", type=Path, help="Matched training report with partition fingerprints")
    parser.add_argument("--report", type=Path, required=True, help="New ablation report; never overwritten")
    parser.add_argument(
        "--modes", nargs="*", choices=ABLATIONS, default=list(ABLATIONS),
        help="ablation modes to run; choose disable_cross_branch_attention for the direct cross-attention test",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--faithfulness-samples", type=int, default=0,
                        help="optional seeded validation subset for four-group token-identity controls")
    parser.add_argument("--faithfulness-random-repeats", type=int, default=5)
    parser.add_argument("--faithfulness-ranking", choices=("attention", "signed_readout"), default="attention")
    parser.add_argument("--faithfulness-intervention", choices=("identity", "static_feature"), default="identity")
    parser.add_argument("--pair-samples", type=int, default=0,
                        help="validation samples for cross-branch pair interventions")
    parser.add_argument("--pair-random-controls", type=int, default=3)
    parser.add_argument("--deployment-audit", action="store_true",
                        help="measure actual guarded family decisions on the same validation partition")
    parser.add_argument("--calibration-candidate-study", action="store_true",
                        help="explore an empirical candidate threshold on calibration; apply to validation only")
    parser.add_argument("--paired-logits", action="store_true",
                        help="also measure exact paired score changes for encoded controls")
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
    if not 0 <= args.faithfulness_samples <= 100 or not 1 <= args.faithfulness_random_repeats <= 100:
        parser.error("faithfulness samples must be 0..100 and random repeats 1..100")
    if not 0 <= args.pair_samples <= 100 or not 1 <= args.pair_random_controls <= 20:
        parser.error("pair samples must be 0..100 and random controls 1..20")
    if not args.modes and not args.faithfulness_samples and not args.pair_samples and not args.deployment_audit and not args.calibration_candidate_study:
        parser.error("request at least one ablation or faithfulness sample")
    if args.report.resolve() in {
        args.reports.resolve(), args.checkpoint.resolve(),
        args.source_report.resolve(), manifest_path.resolve(),
    }:
        parser.error("result path must be distinct from every input")

    with args.source_report.open("r", encoding="utf-8") as source:
        experiment = json.load(source)
    checkpoint_sha256 = _sha256_file(args.checkpoint)
    if experiment.get("checkpoint_sha256") and experiment["checkpoint_sha256"] != checkpoint_sha256:
        parser.error("checkpoint checksum differs from source experiment")
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
    if args.faithfulness_samples > len(validation):
        parser.error("faithfulness sample count exceeds validation partition size")
    if args.pair_samples > len(validation):
        parser.error("pair sample count exceeds validation partition size")

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
        print(json.dumps({"ablation_started": mode}), file=sys.stderr, flush=True)
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
        print(json.dumps({"ablation_completed": mode,
                          "delta": ablation_results[mode]["delta_vs_original"]}),
              file=sys.stderr, flush=True)
        if args.paired_logits and mode in ENCODED_ABLATIONS:
            loader = DataLoader(HybridDataset(validation, tensorizer), batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_samples)
            ablation_results[mode]["paired_logit_changes"] = _paired_family_logit_change(
                model, loader, device, mode,
            )

    faithfulness = None
    if args.faithfulness_samples:
        selected = sorted(validation, key=lambda row: hashlib.sha256(
            f"{seed}:faithfulness:{row.sample_hash}".encode()
        ).hexdigest())[:args.faithfulness_samples]
        rows = []
        for index, row in enumerate(selected):
            batch = collate_samples([tensorizer.encode(row)])
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            diagnostic = family_token_faithfulness(
                model, batch, seed=seed + index,
                random_repeats=args.faithfulness_random_repeats,
                ranking=args.faithfulness_ranking,
                intervention=args.faithfulness_intervention,
            )
            diagnostic["sample_hash"] = row.sample_hash
            diagnostic["target_family"] = tensorizer.family_vocab.decode(diagnostic["target_index"])
            rows.append(diagnostic)
            print(json.dumps({"faithfulness_completed": index + 1, "requested": len(selected)}),
                  file=sys.stderr, flush=True)
        summary = {}
        for branch in ("signature", "static", "temporal_events", "graph"):
            measured = [row["branches"][branch] for row in rows if row["branches"][branch]["evaluated"]]
            summary[branch] = {
                "evaluated_samples": len(measured),
                "mean_top_minus_random_aopc": (
                    sum(row["top_minus_random_aopc"] for row in measured) / len(measured)
                    if measured else None
                ),
                "top_beats_random_samples": sum(row["top_minus_random_aopc"] > 0 for row in measured),
            }
        faithfulness = {
            "notice": (
                "Token-identity-only diagnostic; numeric values, masks, context, edges and cross-branch duplicates remain. Not event deletion, full evidence deletion, rollout or causal proof. Graph rankings cover only the family evidence-head prefix. Sigmoid scores are not calibrated probabilities."
                if args.faithfulness_intervention == "identity" else
                "Signature/static feature token, value and mask are removed; dynamic branches and duplicate evidence remain. This is not event deletion, a causal proof, or evidence of semantic suspicion. Sigmoid scores are not calibrated probabilities."
            ),
            "selection": "smallest SHA256(sample_seed:faithfulness:sample_hash), validation only, no label selection",
            "ranking": args.faithfulness_ranking,
            "intervention": args.faithfulness_intervention,
            "sample_ids_sha256": _fingerprint(selected),
            "summary": summary, "samples": rows,
        }

    pairs = None
    if args.pair_samples:
        selected = sorted(validation, key=lambda row: hashlib.sha256(
            f"{seed}:faithfulness:{row.sample_hash}".encode()
        ).hexdigest())[:args.pair_samples]
        pipeline = HybridInferencePipeline(model, tensorizer, device=args.device)
        pair_rows = []
        for index, row in enumerate(selected):
            measured = compare_attention_pairs(pipeline, row, seed=seed + index,
                                               random_controls=args.pair_random_controls)
            measured['sample_hash'] = row.sample_hash
            pair_rows.append(measured)
            print(json.dumps({'pair_completed': index + 1, 'requested': len(selected)}),
                  file=sys.stderr, flush=True)
        pair_summary = {}
        for branch in ('signature', 'static'):
            measured = [row['branches'][branch] for row in pair_rows if row['branches'][branch]['evaluated']]
            differences = [row['top_minus_random_absolute_interaction'] for row in measured]
            pair_summary[branch] = {
                'evaluated_samples': len(measured),
                'mean_top_minus_random_absolute_interaction': sum(differences) / len(differences) if differences else None,
                'top_beats_random_samples': sum(value > 0 for value in differences),
            }
        pairs = {'selection': 'smallest SHA256(sample_seed:faithfulness:sample_hash), validation only',
                 'sample_ids_sha256': _fingerprint(selected), 'summary': pair_summary, 'samples': pair_rows,
                 'notice': 'Absolute non-additive logit sensitivity; graph reconstruction, truncation, '
                           'duplicate evidence and top-k candidate pruning confound interpretation. '
                           'Not a calibrated maliciousness score or causal proof.'}

    candidate_study = None
    if args.calibration_candidate_study:
        candidate_pipeline = HybridInferencePipeline(model, tensorizer, device=args.device)
        def study_progress(partition):
            def update(done, total):
                if done % 20 == 0 or done == total:
                    print(json.dumps({'candidate_study': partition, 'done': done, 'total': total}),
                          file=sys.stderr, flush=True)
            return update
        calibration_audit = audit_family_decisions(candidate_pipeline, calibration,
                                                    tensorizer.family_vocab.token_to_id,
                                                    study_progress('calibration'))
        selection = select_candidate_threshold(calibration_audit['rows'], target_accuracy=.85, min_selected=20)
        validation_result = None
        if selection['selected'] is not None:
            validation_audit = audit_family_decisions(candidate_pipeline, validation,
                                                      tensorizer.family_vocab.token_to_id,
                                                      study_progress('validation'))
            validation_result = explore_candidate_thresholds(
                validation_audit['rows'], (selection['selected']['threshold'],)
            )['results'][0]
        candidate_study = {
            'notice': 'Exploratory calibration-only selection applied to a validation set previously '
                      'inspected in other analyses; no conformal guarantee or independent test. '
                      'AVClass labels are weak. Model and deployed policy remain unchanged.',
            'selection': selection, 'validation_result': validation_result,
            'calibration_audit_errors': calibration_audit['errors'],
        }

    deployment = None
    if args.deployment_audit:
        pipeline = HybridInferencePipeline(model, tensorizer, device=args.device)
        def audit_progress(done, total):
            if done % 20 == 0 or done == total:
                print(json.dumps({"deployment_audit": {"done": done, "total": total}}),
                      file=sys.stderr, flush=True)
        deployment = audit_family_decisions(
            pipeline, validation, tensorizer.family_vocab.token_to_id, audit_progress,
        )

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
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_hash_verified_against_source_report": bool(experiment.get("checkpoint_sha256")),
        "seed": seed,
        "model_seed": int(experiment["seed"]),
        "device": args.device,
        "batch_size": args.batch_size,
        "sample_seed": seed,
        "split_seed": split_seed,
        "modality_dropout_scope": experiment.get(
            "modality_dropout_scope", getattr(model, "modality_dropout_scope", None) or "summary_only_legacy"
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
        ("token_identity_faithfulness" if args.faithfulness_intervention == "identity"
         else "static_feature_faithfulness"): faithfulness,
        "pair_interaction": pairs,
        "calibration_candidate_study": candidate_study,
        "deployment_family_decisions": deployment,
        "test_metrics": None,
        "ablation_interpretation": {
            "break_targets": "Historical joint perturbation: also changes graph size, pruning and token coverage.",
            "shuffle_events": "Historical shuffle also replaces timestamp gaps with integer positions.",
            "shuffle_events_keep_timestamps": "Shuffle event objects, retain original timestamp slots; graph order still changes.",
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
