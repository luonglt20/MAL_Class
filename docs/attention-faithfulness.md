# Family attention: frozen identity-occlusion diagnostic

`scripts/evaluate_winmet_flow_ablation.py` can now evaluate token-identity
sensitivity in Signature, Static, temporal events and the graph prefix exposed
by the family evidence head. It verifies the prepared source checksum, recreates
and checks **all** partition fingerprints, and evaluates validation only. No
weight, threshold or withheld-test metric is changed.

For a deterministic hash-selected validation subset, fix the original top-logit
family before perturbation. Rank each input group by direct
`family_to_evidence` attention averaged across heads. Set the top 5/10/20% token
ids to PAD (the zero token embedding), and compare the confidence drop with
matched-count random subsets. Report signed AOPC, top-minus-random AOPC, random
variation, attention-score ties, and an insertion curve restoring the ranked
identities from the all-zero-identity group. Scores are **uncalibrated sigmoid
logits**, not calibrated posterior probabilities.

Example using an existing three-branch checkpoint and its matching experiment:

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_winmet_flow_ablation.py \
  artifacts/winmet/volume2_random_blocks_2500_seed44_relv3.jsonl \
  models/winmet_volume2_relv3_samplewindows_hybrid3_attention_1000_seed44.pt \
  artifacts/winmet/volume2_relv3_samplewindows_hybrid3_attention_1000_seed44.json \
  --report artifacts/winmet/new_identity_diagnostic.json \
  --device cpu --batch-size 4 --modes \
  --faithfulness-samples 5 --faithfulness-random-repeats 3
```

Output files are created exclusively; existing results are never overwritten.
`--modes` with no values skips flow ablations, but still evaluates the frozen
validation baseline. Omit the faithfulness flag for the original flow-only
workflow. Use a new output path for each experiment.

## Interpretation limits

- This changes **token identity only**. Numeric features, process/thread/time,
  success/repeat context, masks, graph edges and duplicate information in other
  branches remain. A small drop may reflect redundancy, not useless attention.
- Direct family-head attention is not rollout through self/cross-attention;
  relation-value messages and other classification paths are not explained by
  these weights alone.
- Graph rankings cover at most the first 128 encoded graph nodes exposed by
  this head. Both ranked and total encoded counts are reported; this is not an
  exhaustive graph faithfulness test.
- A zero identity with a true mask is a counterfactual input and may be out of
  distribution. This is neither true event deletion nor a causal intervention.
- Tied attention can make a stable positional ranking arbitrary. One-token
  groups force top and random deletion to coincide. Negative results and ties
  must remain visible; no passing gate is manufactured from these cases.
- Five clustered validation samples are a smoke diagnostic, not acceptance of
  attention faithfulness. Independent evidence-pair annotations, full-evidence
  deletion, IG agreement and broader paired uncertainty estimates remain.

Tests include an analytic model with known token importance, intentionally
misleading attention (which must score worse than random), deterministic random
controls, unchanged context/source tensors/parameters, mode restoration,
argument guards and a real three-branch model contract check.

## First frozen-data diagnostic (2026-10-27)

The corrected-window seed-44 checkpoint was evaluated on five deterministic
validation samples (three predicted redline, one agenttesla, one berbew), without
changing weights/thresholds or opening tests. Signature top-minus-random AOPC
averaged **−0.04390**, beating random on only 1/5 samples. Static, temporal and
graph averages were +0.00159/+0.00287/+0.00012, each beating random on 3/5. These
small, mixed results do not qualify faithfulness, particularly for Signature.
The graph head exposes 128/1,024 nodes here, and one static group has just one
token. See [the compact diagnostic](experiments/winmet-identity-faithfulness-20261027.json)
for provenance, limits and the full local report fingerprint.

This evidence changes the next action: do not present direct attention weights
as suspiciousness or causal contribution. Evaluate the longer-trained model
with the same frozen controls and retain negative results. Independent labels,
signed value/readout contributions and broader interventions are still needed.
