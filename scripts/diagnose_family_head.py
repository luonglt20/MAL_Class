"""Head-capacity diagnostic on TRAIN rows; never an independent benchmark.

The frozen checkpoint may have seen every row. This compares head optimization
on identical cached attention contexts, not model generalization or acceptance.
Run with PYTHONPATH=src python scripts/diagnose_family_head.py --help.
"""
import argparse
import json
import zipfile
from collections import Counter

import torch
from torch import nn

from malware_hybrid.dataset import collate_samples
from malware_hybrid.metrics import multilabel_metrics
from malware_hybrid.model import LabelwiseFamilyHead
from malware_hybrid.training import iter_ember_win32_archive, load_checkpoint, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="Training archive only")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.rows < 8 or args.steps < 1:
        parser.error("rows must be >=8 and steps >=1")
    with zipfile.ZipFile(args.archive) as archive:
        shards = [name for name in archive.namelist() if name.endswith(".jsonl")]
        if not shards or any(not name.endswith("_train.jsonl") for name in shards):
            parser.error("diagnostics require training shards; test archives are not allowed")
    torch.set_num_threads(2)
    set_seed(42)
    model, tensorizer = load_checkpoint(args.checkpoint)
    model.eval()
    contexts, targets, masks, original = [], [], [], []
    hook = model.family_head.register_forward_pre_hook(
        lambda _module, inputs: contexts.append(inputs[0].detach().clone())
    )
    reports = list(iter_ember_win32_archive(args.archive, max_samples=args.rows))
    with torch.no_grad():
        for start in range(0, len(reports), 32):
            batch = collate_samples([tensorizer.encode(r) for r in reports[start:start + 32]])
            original.append(model(batch, return_attention=False)["family_logits"])
            targets.append(batch["family_targets"])
            masks.append(batch["family_target_mask"])
    hook.remove()
    x, y, mask = torch.cat(contexts), torch.cat(targets), torch.cat(masks)
    boundary = int(len(x) * .75)
    labels = [tensorizer.family_vocab.decode(i) for i in range(y.shape[1])]

    def summarize(logits):
        result = multilabel_metrics(
            y[boundary:].numpy(), logits.sigmoid().detach().numpy(),
            mask[boundary:].numpy(), labels, prefix="family",
        )
        return {
            "family_map": result["family_map"],
            "supported_labels": result["family_map_labels"],
            "top1_counts_diagnostic_rows": dict(Counter(labels[i] for i in logits.argmax(-1).tolist())),
        }

    results = {"original": summarize(torch.cat(original)[boundary:])}
    for name, head in (
        ("shared", nn.Linear(x.shape[-1], 1)),
        ("labelwise", LabelwiseFamilyHead(y.shape[1], x.shape[-1])),
    ):
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
        for _ in range(args.steps):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                head(x[:boundary]).squeeze(-1), y[:boundary], reduction="none"
            )
            loss = (loss * mask[:boundary]).sum() / mask[:boundary].sum().clamp_min(1)
            loss.backward()
            optimizer.step()
        results[name] = summarize(head(x[boundary:]).squeeze(-1))
        results[name]["train_bce"] = float(loss.detach())
    print(json.dumps({"notice": "TRAIN-only frozen-context diagnostic, not generalization",
                      "rows": len(reports), "fit_rows": boundary,
                      "diagnostic_rows": len(reports) - boundary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
