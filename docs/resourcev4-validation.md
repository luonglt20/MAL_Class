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
