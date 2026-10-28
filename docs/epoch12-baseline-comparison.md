# Epoch-12 Attention versus a completed 14-epoch bag baseline

The result is mixed. On the same validation reports, Hybrid Attention improves
calibration-threshold macro-F1, but the order-free bag wins mAP and top-k accuracy.
This does not establish general superiority of the attention architecture.

| Validation metric | Frozen Hybrid Attention | Bag of tokens | Attention minus bag |
|---|---:|---:|---:|
| Family mAP | 0.658705 | 0.676361 | −0.017656 |
| Calibrated macro-F1 | 0.454011 | 0.387302 | +0.066709 |
| Top-1 | 0.777778 | 0.809524 | −0.031746 |
| Top-3 | 0.904762 | 0.936508 | −0.031746 |

One new bag baseline was fitted on MPS; Attention weights were reused unchanged.
The bag completed all 14 epochs and selected epoch 14 using validation mAP.
Attention selected epoch 12 from 14 completed epochs before its earlier shutdown
interruption during epoch 15. Its saved history contains only epochs 1–12.
The command's `--epochs 14` describes the new bag budget, not new Attention
training. The checkpoint SHA-256 before and after this run is
`d95f6cf5b599740e72bb0892d9b64a0a0ea45dcca1fa13d9a9fc8860807a1b93`.

Both models used the same 1,000-report reservoir, archive-group split, train-only
vocabulary, width 32, batch size 4, ranking-loss weight 0.5 and seed 44. Train,
validation, calibration and withheld partitions contain 637/107/101/155 reports.
Thresholds use calibration only. Validation has 100 family-labeled rows, 17
positive-supported labels and 63 rows for known-family top-k evaluation. No test
partition was evaluated. AVClass labels are weak reference labels.

Bag receives Signature/Static numeric evidence plus order-free event tokens.
It lacks graph edges, process/thread and time context, and has different model
capacity. This contrast cannot isolate attention alone. The frozen
[attention controls](epoch12-attention-validation.md) remain separate evidence:
cross-attention removal hurts scores, but Signature/event identity faithfulness
still fails. Neither result cancels the other.

The runner started at commit `69ff6f2`, before reporting improvements were edited.
Its original full report is preserved unchanged; the committed compact record
adds provenance and caveats. The runner did not persist the bag weights, so this
is a metric/history record, not a deployable bag checkpoint. The current runner
now labels reuse explicitly, records checkpoint hashes and refuses a checkpoint
change during baseline fitting. Seed summaries reject mixing fresh Attention
training with reused checkpoints.

No default UI model or thresholds were changed. Next architecture work must
address evidence faithfulness and independently reviewed suspicious links;
further training epochs alone are not justified by these results. The separate
five accepted-label disagreements and missing benign hybrid validation remain.

[Compact metrics and split fingerprints](experiments/winmet-epoch12-vs-bag14-20261029.json).
