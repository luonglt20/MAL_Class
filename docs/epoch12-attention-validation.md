# Frozen epoch-12 attention validation

This evaluates the actual calibrated candidate, without new training, changed
thresholds or test-set access. The model stays unregistered/unqualified. Source
and checkpoint hashes plus all four partition identities were checked.

Full validation has 107 reports, 100 family-labeled rows and only 63 known-family
ranking rows. Baseline mAP/calibrated macro-F1 are 0.6587/0.4540. The following
changes are ablated minus full, **not** independent baseline-training results:

| Frozen intervention | mAP change | Macro-F1 change |
|---|---:|---:|
| Disable token-level cross-branch attention | −0.0846 | −0.2147 |
| Shuffle events, preserve timestamp slots | −0.1437 | −0.1250 |
| Zero temporal resource identities | 0.0000 | 0.0000 |
| Zero process/parent/thread context | −0.1846 | −0.2885 |
| Mask graph edges, keep nodes | −0.0907 | −0.0481 |

These controls support contribution from cross-branch paths, execution order,
process context and graph edges on this split. They do not establish semantic
correctness of specific suspicious-evidence links. Cross-branch removal leaves
self/fusion/label attention present; shuffling also changes graph ordering, and
the process control removes embeddings as well as pairwise relations.

## Why unchanged F1 does not mean zero resource use

A second frozen run compared the 2,461 family logits pairwise for the same 107
reports. Resource removal changes logits by 0.00460 mean absolute / 0.12995
maximum, and sigmoid scores by at most 0.01223. No top-family argmax changes.
There are 62,464 known resource identities among 155,222 valid events. This is
not an all-zero input channel. The control removes **temporal relation IDs**,
not resource nodes or duplicated information in the graph. It proves some
score sensitivity, not a useful resource-level decision effect or causality.

The evaluator now has optional `--paired-logits` for encoded controls. This
avoids interpreting unchanged ranking/threshold metrics as exact insensitivity.
Tests include a model whose score changes but top class stays the same. The
runner also refuses a checkpoint hash mismatch when the source report pins it,
records whether this was verified, and recovers dropout metadata from the
checkpoint rather than incorrectly labeling a missing report field as legacy.

## Attention-weight identity faithfulness is still insufficient

Ten deterministic validation samples were evaluated with five random controls
at each 5/10/20% masking fraction. Mean top-minus-random AOPC is −0.03943 for
Signature (1/10 wins), +0.03752 Static (5/10), −0.01694 temporal events (3/10),
+0.02138 graph (7/10). Identity-only masking preserves numeric evidence, context,
edges and cross-branch duplicates. The raw family-head weights are not rollout
or signed value contributions. Still, these mixed/negative results do **not**
pass the requested attention-faithfulness gate, particularly for Signature and
events. A cross-attention ablation gain cannot substitute for that gate.

Next decisions must retain these failures, analyze resource coverage/handle
identity and independently check evidence links. Do not force explanatory maps
to look plausible or select masking thresholds from these validation outcomes.
No default UI model change is justified by these diagnostics; the previous
25/30 AVClass agreement among accepted predictions also remains insufficient.

[Compact metrics, fingerprints and caveats](experiments/winmet-epoch12-attention-controls-20261029.json).
