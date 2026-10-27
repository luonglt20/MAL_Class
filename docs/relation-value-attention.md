# Experimental relation-value attention

This is an opt-in architecture experiment, not a new qualified classifier.
Signature/Static self-attention, their bidirectional cross-attention and the
temporal/graph/family evidence paths remain present.

The bias-only temporal block adds learned pairwise scores before softmax. A
row-constant bias cancels, so that component alone cannot distinguish a uniformly
known relation from an unknown relation. A regression test reproduces this
specific limitation; it is not a claim that the whole hybrid ignores relations.

Inspired by [Shaw et al. (2018), Eq. 3](https://aclanthology.org/N18-2074.pdf),
the experiment adds a learned relation vector to the attention-weighted values.
It retains our existing score bias, not the paper's entire architecture. The
five binary coordinates are same process, same thread, parent/child, same
resource and both-success. No raw PID, file digest or verdict becomes a feature.

For each head, attention mass is summed by relation type, then multiplied by
that head's relation-value vectors. The result uses the existing output
projection without adding its bias twice. This avoids a dense
`batch × heads × query × key × head_dimension` value tensor. It still needs
attention weights internally even on the non-explanatory path; performance and
memory must be measured on real traces.

Vectors initialize at zero, preserving the initial bias-only mathematical
function without consuming extra RNG. This does not guarantee bitwise identical
training: the value path materializes weights instead of using the same fused
fast path. The opt-in is `temporal_relation_values=True`, exposed by the WinMET
runner's `--temporal-relation-values`. Old checkpoint configs default to false;
model/CSV metadata records the flag and seed aggregation rejects mixed variants.
The registered UI model is not switched merely because unit tests pass.

Tests cover initial-function parity, a learned response to uniform known versus
unknown relations, masked-key isolation, padding invariance, a nonzero family-head
gradient and checkpoint/fast-explanation parity. A small MPS forward/backward
check has finite outputs and finite nonzero relation-vector gradients. These
checks establish implementation/capacity, not classification or false-positive
quality. Existing evidence-chain and abstention guards remain unchanged.

The real-data comparison must use the corrected `per_sample_v1` window policy,
identical cohort/partitions/loss/epoch budget and unchanged test exclusion.
Evaluate node-preserving resource/context/edge controls, attention faithfulness,
additional seeds and independent labels. Attention weights alone do not capture
the value-message contribution and must not be presented as causal explanations.
