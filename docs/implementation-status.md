# MAL_Class: evidence and remaining work

This is an independent multimodal research project. Legacy reports and scores are reference material, not experimental evidence for this project.

| Objective | Current evidence | Required next evidence |
|---|---|---|
| Pairwise relation/time bias | Executable model; gradient and elapsed-time counterfactual test | Real sandbox entity-resolution and flow ablations |
| Window overlap | Encoder and CLI support overlap, event folding | Padding invariance and long-report benchmarking |
| Fourier graph / causal pooling / pruning | Fourier encoder, bounded chronological multi-hop paths, duplicate-parent removal and conservative pruning | Contextual validation on large real graphs |
| Event-level cross-attention | Both directions query event/static tokens; inference trace test | Faithfulness on independent annotations |
| Imphash/TLSH/ssdeep train prototypes | Exact imphash and optional fuzzy matching against train-only prototypes; self-match exclusion, anonymized tokens | Large-scale nearest-neighbor indexing and real-hash validation |
| Dashboard / Sankey / Navigator | Family×modality attention, execution graph, Sankey and Navigator 4.5 export | Visual QA on real reports and large-graph sampling |
| 15 evidence specifications | Fifteen multi-event rules; order/success/process/target guards and negative fixtures | Validate every rule on independently annotated reports; add handle resolution |
| Golden Set 50 per technique | Missing | Independent manual annotations with provenance; automated rules cannot substitute |
| STIX / OSCAL version locking | ATT&CK 19.2 STIX and NIST OSCAL content 1.2.1 URLs/SHA-256 locked; automated ID/name validation passes | Expert review of semantic crosswalk relationships |
| EMBER2024 static-data path | Train/test archives checksum-verified; old 6,648-row checkpoint on 359,994 unique official test rows: family mAP 0.0212, calibrated macro-F1 0.0121, top-1 0.540, top-3 0.589; malware precision 0.965, recall 0.0658, FPR 0.00240 (432/179,994 benign; Wilson 95% CI 0.00218–0.00264). The 26k/support-50 candidate on 349,994 remaining test rows (first 10k held as development): family mAP 0.0302, calibrated macro-F1 0.0198, top-1 0.607, top-3 0.667; malware precision 0.284, recall 0.00264, FPR 0.00672 (Wilson 95% CI 0.00635–0.00712) | Improve calibrated family decisions and malware sensitivity; no acceptance gate is met yet |
| Attention inference fast path | `return_attention=False` avoids materializing attention maps; parity test matches all logits to 1e-6, while explanatory inference retains maps | Benchmark throughput and memory against the old path on full hybrid traces |
| WinMET bounded acquisition/preparation | Pinned metadata/license; consensus label file downloaded and checked; optional encrypted-7z reader tested on synthetic archives with per-report/decompression/output bounds and no executable extraction | Finish and verify report-volume download, inspect real schema, prepare representative cohorts and train |
| Resource identity and global event order | Anonymous report-local resources stay distinct despite identical semantic path tokens; graph features exclude resource IDs, PIDs and event indices; global chronological cap and datetime parsing have regression tests | Re-prepare old traces; real handle/alias resolution and large-report validation |
| Direct PE upload | Added | Bounded static-only PE parser/UI (never executes sample); dynamic behavior still requires CAPE report |
| Dashboard dataset training | Added; Streamlit health endpoint `ok`, page HTTP 200 | Local CAPE folder/JSONL or verified EMBER Win32 archive; interactive upload/visual QA remains |
| MalBehavD auxiliary data | Missing | Verify the schema: API-presence features can support bag-of-API/hard-negative experiments, not ordered-flow pretraining |
| Conformal / OOD | Per-family Mondrian split-conformal binary sets, persisted in checkpoint; four-way data split | Real held-out coverage/set-size evaluation and distribution-shift tests |
| Seven baselines / WinMET experiment | Seven executable protocol models share one tensor contract; no scores claimed | Identical leakage-controlled real-data runs and multi-seed report |
| Metrics | F1, balanced accuracy, masked mAP/per-label recall, localization, event Layer-IG and attention/IG rank agreement | Independent event annotations and measured acceptance results |

## Acceptance gates

The full-hybrid empirical gates remain unproven: family macro-F1 >= .88; ATT&CK mAP >= .82; hybrid improvement >= .02; Wilson 95% upper bound FPR <= .01; chain completeness >= .90; attention/IG agreement >= .75; shuffle macro-F1 relative decrease >= .15. Static EMBER evaluations meet the FPR bound, but with very low malware recall; this does not establish acceptable detection or the full-hybrid gate.

Even zero false positives among all 268 benign reports gives Wilson two-sided 95% upper bound about 1.41%, above 1%. At least 381 independent benign observations with zero false positives are needed for this particular bound. A validation subset needs its own denominator. Dataset access and labels, including benign versus undetected, must be verified from original sources.

The 1,000-row EMBER run is an execution smoke test only: it demonstrates the locked-source loader, temporal partition, model training, checkpoint, and inference path, but its held-out metrics fail the project gates and must not be presented as a research result. No manual annotations or accepted accuracy are inferred from unit tests. Dataset roles in the objective are preserved: WinMET/CAPE for hybrid experiments, MALVADA for report processing, EMBER for static pretraining, MalBehavD for dynamic pretraining, Oliveira/Catak for sequence baselines, BODMAS for temporal drift and capa for auxiliary evidence comparison. Each requires source/schema/license verification before import; static capa findings alone do not establish observed dynamic execution chains.

## Measured EMBER runs (not acceptance results)

- `ember2024_calibrated.pt`: 6,648-row weekly sample, 5 epochs, 33 family labels. Official test is temporal, not used to fit this checkpoint. Behavior metrics are null because EMBER does not supply strong observed behavior negatives; the behavior head is disabled.
- `ember2024_26k_mps.pt`: 26,000-row weekly sample, 8 epochs, 104 family labels (support threshold 10). Internal temporal test: family mAP 0.0153, top-1 0.348, top-3 0.376; a 10,000-row official-test development prefix: mAP 0.0134, top-1 0.480, top-3 0.520. This candidate was not promoted.
- `ember2024_26k_support50_mps.pt`: same 26,000-row sample and split, 25 family labels (support threshold 50). Internal temporal test: family mAP 0.0507 over 21 evaluable labels, top-1 0.461, top-3 0.526, macro-F1 0.0019. On the 349,994-row official-test remainder after skipping the 10,000-row development prefix: family mAP 0.0302, calibrated macro-F1 0.0198, top-1 0.607, top-3 0.667. Malware recall is only 0.00264 (precision 0.284, FPR 0.00672); behavior head is disabled.

These are exploratory evaluations, not comparable controlled improvements: the old checkpoint uses 33 labels and all 359,994 test rows, whereas support-50 uses 25 labels and the 349,994-row remainder. The old checkpoint had already been evaluated on the entire official test archive before later development. Excluding the 10,000-row development prefix does not restore an untouched final holdout. Calibrated family F1 remains far below the acceptance gate and the malware auxiliary head is not robust. Further tuning must use training-archive validation only; final scientific claims require a separately reserved evaluation cohort and a fixed label space.

## Family-head diagnosis and repair

A diagnostic of the support-50 checkpoint on the first 64 unique TRAIN archive rows selects `berbew` as top-1 for all 64. Logits vary between samples, so the input path is not simply constant, but family ranking is collapsed on this cohort. The legacy head uses one shared scalar projection across family attention contexts; with only one modality its modality contexts are identical across queries. An optional `--family-head-variant labelwise` uses separate per-family projections/biases over the existing label-query attention contexts. Legacy checkpoints retain their original layout and outputs, and the default remains `legacy_shared` pending stronger evidence. This is a capacity hypothesis, not a verified solution to ranking collapse.

CPU best-checkpoint snapshots now clone tensor storage so subsequent optimizer steps cannot overwrite the selected epoch. A regression test covers this storage-aliasing bug; prior MPS snapshots copied to CPU and were not affected by this particular bug.

Frozen-context diagnostic: 512 TRAIN-prefix rows, first 384 fit the head, final 128 diagnose it, seed 42, 200 AdamW steps at 0.01, unweighted masked BCE. Shared-head refit: mAP 0.1624, all 128 top-1 predictions `berbew`. Labelwise refit: mAP 0.2188, top-1 counts `wacatac` 70, `berbew` 36, `flystudio` 21, `sfone` 1 (seven supported diagnostic labels). These are head-capacity diagnostics on a previously trained encoder, not held-out generalization scores. They justify a matched end-to-end experiment, not model promotion.

Matched end-to-end diagnostic: first 2,048 TRAIN-archive reports, seed 42, temporal 65/10/10/15 split (1,331/205/204/308), five supported training families, five epochs, batch 64, d_model 32, MPS, family_min_support 10. Shared-head selected validation mAP 0.1238 versus labelwise 0.1009; internal-test mAP 0.0875 versus 0.1127 (four supported labels), and calibrated macro-F1 0.0689 versus 0.0139. Both predict `berbew` for all 205 validation rows. The labelwise variant therefore has mixed results and does NOT resolve collapse end to end. Neither checkpoint is promoted. Next investigate loss imbalance, label supervision and model selection on training-only validation; do not infer a research improvement from this small cohort.

Label audit of that same cohort: only 124/1,331 training rows contain a retained family (354 have any family); `berbew` has 42 of those positives. Validation contains only 27 retained-family positives, including 14 `berbew`. Thus top-1 around 52% can arise from a majority-family predictor without discrimination. Future comparisons need a train-frequency-prior sanity baseline and fixed label space, not just top-k accuracy. This audit does not justify discarding benign samples or treating unknown families as benign.

### Matched objective experiment: rejected candidate

`scripts/compare_family_objectives.py` verifies the pinned TRAIN archive, retains the same 2,048-row cohort/splits above, and fits two labelwise-head models with seed 42, five epochs, d_model 32 and batch 64 on MPS. Both select their checkpoint using validation family mAP; neither uses official-test rows. Weight 0 retains BCE; weight 0.5 adds auxiliary cross-entropy only on fully known single-family rows. Benign, unknown and multilabel rows retain BCE without the auxiliary term. Inference remains sigmoid/multilabel. The default ranking weight stays zero and default checkpoint-selection metric stays loss.

| Method | Internal-test mAP | Calibrated macro-F1 | Top-1 | Top-3 |
|---|---:|---:|---:|---:|
| Training-frequency prior, no features | 0.05085 | not measured | 0.5167 | 0.9833 |
| Labelwise, ranking weight 0 | 0.12629 | 0.01515 | 0.3667 | 0.5333 |
| Labelwise, ranking weight 0.5 | 0.14860 | 0.03509 | 0.0167 | 0.5333 |

There are 60 known-family-positive internal-test rows and four supported evaluation labels. Both checkpoints select epoch 5. Validation mAP is 0.10933 versus 0.11504, but validation top-1 is 0.1852 versus 0.0. The auxiliary term has mixed/worse results and neither candidate is promoted. This is exploratory, single-seed evidence on an already-examined internal cohort, not an untouched holdout or a scientific improvement claim. Full local results, including train-only priors, split-ID fingerprints, history and input checksum, are in `artifacts/family_objectives_2048_v1.json` (ignored research artifact).

The trainer now persists train-only family priors and selected-epoch/selection-metric metadata; evaluation reports the no-feature prior baseline. Old checkpoints without stored priors report those baseline metrics as unavailable rather than fitting a prior on evaluation labels.

Checkpoint loading now uses restricted `weights_only=True` without an unsafe-pickle fallback. Existing support-50 and both objective-experiment checkpoints load successfully in restricted mode. A regression fixture containing a Python callback is rejected before callback execution. Stable PyTorch >= 2.6 is required at dependency and runtime level because [PyTorch's CVE-2025-32434 advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6) identifies earlier restricted loaders as affected. This is defense in depth, not a guarantee against all malformed-model/parser attacks; keep loading limited to project-generated checkpoints.

Feature audit of these 2,048 raw TRAIN-prefix reports: all truncated static feature vectors are distinct (no exact cross-family collisions), evidence lengths range from 64 to 396, and 101 reports exceed the 256-token cap. This rules out identical feature vectors as the sole explanation; truncation coverage, feature scaling, temporal distribution shift and sparse family supervision still need investigation. Real dynamic trace training remains pending the WinMET download.

## Trace-normalization corrections

Previously, unrelated files normalized to the same path category could share a graph node; the temporal resource relation was also unavailable for placeholder-style targets. Parsing now allocates anonymous report-local entity IDs while graph vocabulary uses only semantic resource tokens. Named objects are distinct; repeat access retains equality; handles are process-scoped but not resolved to underlying objects. Process IDs and event indices are no longer graph features under `semantic_v2`. Old checkpoints missing the new tensorizer field explicitly retain legacy graph-token lookup; previously prepared traces must be rebuilt to recover lost identities.

Event truncation previously returned after the first process reached the cap, before global sorting. It now selects globally earliest calls using a bounded heap, then compresses repeats. Numeric and ISO/date-time timestamps retain elapsed-time information. These repairs are tested on counterexamples, not yet validated on the downloading real volume. See [the bounded workflow](winmet-workflow.md).

Dynamic trace evidence is still missing from real-data training. WinMET provides 31,844 CAPE traces with process/API/resource sequence detail and reports roughly 13 GB compressed / 750 GB uncompressed. Although its rendered page leaves the license field blank, the [official record API](https://zenodo.org/api/records/16414116) explicitly returns `metadata.license.id = gpl-3.0-or-later`. License/source metadata, one volume's size/checksum and the label file are recorded in `configs/winmet.lock.json`. Bounded report extraction remains required before training; the workstation currently has about 14 GiB free, so full-volume extraction is not feasible. No dynamic training results are claimed. See [the official WinMET record](https://zenodo.org/records/16414116).
