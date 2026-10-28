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
