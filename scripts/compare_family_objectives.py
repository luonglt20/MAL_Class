"""Matched TRAIN-archive objective experiment; not a final accuracy benchmark."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from malware_hybrid.dataset import collate_samples
from malware_hybrid.dataset_sources import verify_dataset_file
from malware_hybrid.training import iter_ember_win32_archive, research_split, train_model, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        parser.error("output already exists; choose a new experiment name")
    if args.rows < 32 or args.epochs < 1:
        parser.error("rows must be >=32 and epochs >=1")
    torch.set_num_threads(2)
    verification = verify_dataset_file(args.archive)
    reports = list(iter_ember_win32_archive(args.archive, max_samples=args.rows))
    train, validation, calibration, test, strategy = research_split(reports, 42)
    result = {
        "notice": "Exploratory TRAIN-prefix cohort; internal test has been examined before, not a final holdout.",
        "verification": verification, "split_strategy": strategy,
        "seed": 42, "epochs": args.epochs, "d_model": 32, "batch_size": 64,
        "family_min_support": 10, "family_head_variant": "labelwise",
        "selection_metric": "family_map", "device": args.device,
        "partitions": {name: {"count": len(rows), "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(r.sample_hash for r in rows)).encode()).hexdigest()}
            for name, rows in zip(("train", "validation", "calibration", "test"),
                                  (train, validation, calibration, test))},
        "runs": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"phase": "data_verified", "partitions": result["partitions"]}), flush=True)
    for weight in (0., .5):
        checkpoint = Path("models") / f"{output_path.stem}-rank{weight:g}.pt"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        run = train_model(
            train, validation, checkpoint, epochs=args.epochs, batch_size=64, seed=42,
            device=args.device, d_model=32, family_min_support=10, behavior_min_support=2,
            calibration_reports=calibration, test_reports=test, train_behavior_head=False,
            family_head_variant="labelwise", family_ranking_weight=weight,
            selection_metric="family_map",
        )
        model, tensorizer = load_checkpoint(checkpoint)
        model.eval()
        counts = Counter()
        with torch.no_grad():
            for start in range(0, len(validation), 32):
                batch = collate_samples([tensorizer.encode(r) for r in validation[start:start + 32]])
                ids = model(batch, return_attention=False)["family_logits"].argmax(-1).tolist()
                counts.update(tensorizer.family_vocab.decode(i) for i in ids)
        run.update(family_ranking_weight=weight, validation_top1_counts=dict(counts))
        result["runs"].append(run)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"weight": weight, "selected_epoch": run["selected_epoch"],
            "validation_top1_counts": dict(counts), "test": {k: v for k, v in run["test_metrics"].items()
            if k.startswith(("family_map", "family_calibrated_macro", "family_top", "family_prior"))}}), flush=True)


if __name__ == "__main__":
    main()
