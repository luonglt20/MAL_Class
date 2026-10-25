# MalBehavD-V1 auxiliary dynamic sequences

Source: [the dataset authors' public repository](https://github.com/mpasco/MalbehavD-V1),
which declares an MIT license. The CSV is stored locally at
`Data/raw/malbehavd/MalBehavD-V1-dataset.csv`; its byte size and SHA-256 are pinned
in `configs/datasets.lock.json`.

The acquired file has 2,570 rows and 2,543 unique SHA-256 values. The 27 repeated
hash rows include 16 exact duplicate rows and 11 additional trace variants;
different traces for one executable are retained. An audit found 1,601 distinct
exact API sequences, with 179 repeated-sequence groups covering 1,132 rows,
1,439 API-call multisets, and 176 repeated-multiset groups covering 1,289 rows.
Split assignment groups connected files by SHA-256, exact normalized sequence,
and API-call multiset; this also prevents an order-free baseline from seeing an
identical bag representation in different splits. The cross-label connected
group (12 hashes) is quarantined because API-only evidence cannot resolve its
contradictory labels. Remaining traces of a hash stay in one split and
probabilities are averaged by hash for evaluation. Hash-level source-label
counts are 1,276 and 1,267. Ordered API sequences have length 2–175 (median 37).
Numeric header columns are sequence positions;
trailing empty columns are padding. The authors document balanced benign and
malicious populations, but do not specify the numeric 0/1 mapping in the README,
so experiments preserve raw label IDs and do not call label 1 “malware.”

The dataset contains API names only—no executable bytes, arguments, return
values, process/resource relationships, family, or ATT&CK/MBC labels. It is
therefore an auxiliary dynamic-only binary classification experiment, not
supervision for the multimodal family or behavior heads.

Before any train/validation/test split, remove exact duplicate rows and group
related hashes/traces to prevent leakage. Preserve alternate traces as separate
auxiliary dynamic-only samples; never pair them with a PE or CAPE report from
another source to manufacture a hybrid sample. Use it only for
an explicitly labeled sequence/verdict experiment. Missing APIs in these short
sequences are not strong negative evidence for individual ATT&CK techniques, and
the dataset cannot supervise malware-family classification. Its traces also have
unknown API return status, so an adapter must preserve that uncertainty rather
than mark every call successful or failed.

The adapter implements deterministic stratified train/validation/test splits
after connected hash/sequence/bag grouping, a label-query self-attention model
gated with mean-embedding API features, a mean-embedding
bag-of-API baseline, and a test-only shuffled-order counterfactual. Attention
weights are exposed as ranked API calls but are not causal explanations. Run
from the repository root with:

```bash
PYTHONPATH=src python -m malware_hybrid.cli train-malbehavd \
  Data/raw/malbehavd/MalBehavD-V1-dataset.csv \
  --output models/malbehavd_api_attention.pt --seed 42
```

The single-seed experiment and its limitations are recorded in
`docs/implementation-status.md`. It is not a hybrid/family acceptance result;
independent-source testing and multiple seeds remain required. No executable is
run by this adapter or experiment.
