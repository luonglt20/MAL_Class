# Resource v4: matched validation, not release qualification

The v4 Attention model completed 14 epochs; epoch 14 was selected by validation
mAP and thresholds were calibrated on the disjoint calibration partition. The
paired 2,500-report preparation has identical sample order, family labels,
atomic evidence and archive groups to v3; all four 1,000-sample partition
fingerprints match. No held-out test was evaluated. Family labels are weak
AVClass consensus, not independently adjudicated ground truth.

On 107 validation reports (100 with family reference labels, 63 eligible for
known-family top-k), mAP is 0.6924 and calibrated macro-F1 0.5007. The earlier
v3 candidate scored 0.6587 and 0.4540 on the matched validation partition.
This single-seed difference is exploratory, not proof of generalization or a
causal effect from the new file resolver.

The guarded pipeline accepted a primary family on 22/107 reports; all 22 agreed
with AVClass. It abstained on 85. The v3 candidate accepted 30 and agreed on 25.
The higher observed agreement on fewer predictions does not establish accuracy
or a false-positive guarantee; independent annotations and benign hybrid
validation remain needed.

Frozen interventions relative to v4 baseline: removing temporal resource IDs
changed mAP by +0.00018 and calibrated macro-F1 by +0.0392; masking graph edges
changed them by −0.1007 and −0.1174; disabling token-level cross-branch attention
changed them by −0.1172 and −0.2388. Removing temporal resource IDs leaves
other resource information in the graph. These results show dependency on graph
edges and cross-branch paths in this split, but do not establish semantic
correctness of an individual attention link. Attention faithfulness, independent
behavior labels and benign false-positive evaluation remain open.

The checkpoint is research-only and is not registered in the UI. The first
ablation attempt rejected missing family-head metadata; the reference was fixed
from the saved training report before the successful read-only run.

Raw CAPE uploads now select the v4 parser when a manually loaded checkpoint
records only the v4 schema, or when a research registry explicitly specifies
that schema. Prepared JSONL keeps its recorded schema and incompatibilities are
still rejected. A real v4-checkpoint smoke test on a synthetic four-event CAPE
fixture used the compatible schema, returned `UNKNOWN`, and did not execute
the sample. This is integration verification, not model qualification.

[Metrics, partition fingerprints and artifact checksums](experiments/winmet-resourcev4-validation-20261029.json).

## Reserved internal holdout

A subsequent one-time frozen evaluation on 155 group-isolated volume-2 reports
(145 with family supervision; 91 known-family top-k rows) achieved family mAP
0.6030 over 19 supported labels, calibrated macro-F1 0.3980, top-1 0.7582 and
top-3 0.9231. No weights or thresholds changed. These results are lower than
the validation mAP 0.6924 and calibrated macro-F1 0.5007. This is an internal
clustered holdout with weak AVClass labels, not an independent ground-truth or
benign false-positive study; the model remains unqualified for release.

The original result notice accidentally says volume 1; its source path and
manifest identify volume 2. The result remains unchanged and the evaluator's
notice has been corrected for future runs. Its checkpoint checksum guard now
refuses a mismatch before loading model or data.

[Holdout metrics and checksums](experiments/winmet-resourcev4-holdout-20261029.json).

## Token-identity attention diagnostic

On the same ten deterministic validation samples used for the earlier v3
diagnostic, top-attention minus random deletion AOPC averaged −0.01391 for
Signature (top beats random in 3/10 samples), −0.00528 for Static (4/10),
−0.01165 for temporal events (1/10), and −0.00775 for graph tokens (4/10).
All four averages are negative. This diagnostic does not pass the requested
attention-faithfulness criterion; weights should not be presented as reliable
evidence importance. It masks token identities only, retains numeric values,
context, edges and duplicated information, and covers only the graph evidence
head's prefix. Ten validation samples are insufficient for broad conclusions;
the results do not establish that the model ignores those branches.

[Diagnostic summary and full-report checksum](experiments/winmet-resourcev4-faithfulness-20261029.json).

The same ten validation samples were then ranked by signed contextualized
readout contribution, using the same identity-only masking and five random
controls per fraction. Top-minus-random AOPC averaged +0.01130 for Signature
(6/10 wins), −0.02987 for Static (1/10), +0.00981 for events (9/10), and
+0.00034 for graph (5/10). This improves the diagnostic for Signature and
events over raw attention ranking, but Static remains negative and graph is
near zero. Signed readout decomposes contextualized values, not causal effects
of raw inputs; the small validation diagnostic does not qualify explanations.
No model weights or thresholds changed.

[Signed diagnostic and checksum](experiments/winmet-resourcev4-signed-faithfulness-20261029.json).

## Expanded raw-attention diagnostic

The frozen checkpoint was evaluated on 30 hash-selected validation samples,
including the original ten, with five matched random controls per fraction.
Mean top-attention minus random AOPC was **−0.00733** for Signature (11/30
positive), **+0.00234** for Static (9/30), **−0.01328** for temporal events
(3/30), and **−0.00367** for graph (14/30). These results do not establish the
required attention-faithfulness gate. Static attention scores tied on 19/30
samples; event and graph scores had ties on all 30. A small positive mean in
Static should not be confused with consistent per-sample improvement.

This is the same identity-only counterfactual described above: numeric values,
context and edges remain, information may be duplicated across branches, and
graph rankings only cover the evidence-head prefix. The samples come from the
same clustered validation split with weak AVClass labels; they are not an
independent annotation set, and the means do not establish statistical
significance. Weights, thresholds and the held-out test were not changed.
[Reproducible aggregate and report checksum](experiments/winmet-resourcev4-faithfulness30-20261029.json).

## Static feature-removal diagnostic

On the identical 30 validation samples and frozen checkpoint, removing the
highest-attention feature's token, value and mask produced mean top-minus-random
AOPC **−0.01375** for Signature (11/30 positive) and **−0.00804** for Static
(7/30 positive). Five matched random controls per fraction were used. Both
averages are negative, so this stronger intervention also fails to establish
that direct attention reliably identifies influential features. It does not
evaluate Dynamic or Graph. Other branches remain intact and may contain
duplicate signals; masking may create out-of-distribution inputs. These are
clustered validation diagnostics, not independent ground truth or proof of a
model defect in a particular sample.

The original local report uses the legacy JSON key `token_identity_faithfulness`
because the field was renamed after the process started; its stored intervention
and notice correctly specify `static_feature`. Future runs use the corrected
`static_feature_faithfulness` key. No model weight, threshold, or test split was
changed. [Aggregate and full-report checksum](experiments/winmet-resourcev4-feature-removal30-20261029.json).

### Signed readout on the same intervention

Using the same samples, fixed predicted targets, masks and random controls,
ranking by signed contextualized readout gave mean top-minus-random AOPC
**+0.01273** on Signature (18/30 positive) and **−0.00936** on Static (6/30).
Against raw attention on the same samples, the paired mean change was +0.02648
for Signature (22 samples improved, six worsened, two tied) and −0.00131 for
Static (19 improved, ten worsened, one tied). The reconstruction error of the
linear logits was at most 8.59e-6. Positive Signature results on this small
validation diagnostic justify further evaluation, not a general faithfulness
claim; Static remains negative despite more per-sample improvements.

The signed score is a decomposition of **already contextualized** values and
can include cross-branch information; it is not a causal attribution to the
original token or a score for maliciousness. The model, calibration and test
partition remain unchanged. `scripts/compare_faithfulness_rankings.py` checks
matching sample IDs, targets, budgets, random controls and checkpoint before
computing paired differences. [Aggregate and report hashes](experiments/winmet-resourcev4-signed-feature30-20261029.json).
