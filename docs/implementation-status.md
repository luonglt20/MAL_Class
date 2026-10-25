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
| EMBER2024 static-data path | Train/test archives checksum-verified; old 6,648-row checkpoint evaluated on the official later-time holdout (359,994 unique rows): family mAP 0.0212, calibrated macro-F1 0.0121, top-1 0.540, top-3 0.589; malware precision 0.965, recall 0.0658, FPR 0.00240 (432/179,994 benign; Wilson 95% CI 0.00218–0.00264) | 26k/support-50 candidate is in final evaluation on the remaining holdout after excluding the 10k development prefix; no acceptance gate is met yet |
| Attention inference fast path | `return_attention=False` avoids materializing attention maps; parity test matches all logits to 1e-6, while explanatory inference retains maps | Benchmark throughput and memory against the old path on full hybrid traces |
| Direct PE upload | Added | Bounded static-only PE parser/UI (never executes sample); dynamic behavior still requires CAPE report |
| Dashboard dataset training | Added; Streamlit health endpoint `ok`, page HTTP 200 | Local CAPE folder/JSONL or verified EMBER Win32 archive; interactive upload/visual QA remains |
| MalBehavD pretraining | Missing | Separate API-flow pretraining dataset adapter and measured hard-negative iteration |
| Conformal / OOD | Per-family Mondrian split-conformal binary sets, persisted in checkpoint; four-way data split | Real held-out coverage/set-size evaluation and distribution-shift tests |
| Seven baselines / WinMET experiment | Seven executable protocol models share one tensor contract; no scores claimed | Identical leakage-controlled real-data runs and multi-seed report |
| Metrics | F1, balanced accuracy, masked mAP/per-label recall, localization, event Layer-IG and attention/IG rank agreement | Independent event annotations and measured acceptance results |

## Acceptance gates

All empirical gates remain unproven: family macro-F1 >= .88; ATT&CK mAP >= .82; hybrid improvement >= .02; Wilson 95% upper bound FPR <= .01; chain completeness >= .90; attention/IG agreement >= .75; shuffle macro-F1 relative decrease >= .15.

Even zero false positives among all 268 benign reports gives Wilson two-sided 95% upper bound about 1.41%, above 1%. At least 381 independent benign observations with zero false positives are needed for this particular bound. A validation subset needs its own denominator. Dataset access and labels, including benign versus undetected, must be verified from original sources.

The 1,000-row EMBER run is an execution smoke test only: it demonstrates the locked-source loader, temporal partition, model training, checkpoint, and inference path, but its held-out metrics fail the project gates and must not be presented as a research result. No manual annotations or accepted accuracy are inferred from unit tests. Dataset roles in the objective are preserved: WinMET/CAPE for hybrid experiments, MALVADA for report processing, EMBER for static pretraining, MalBehavD for dynamic pretraining, Oliveira/Catak for sequence baselines, BODMAS for temporal drift and capa for auxiliary evidence comparison. Each requires source/schema/license verification before import; static capa findings alone do not establish observed dynamic execution chains.

## Measured EMBER runs (not acceptance results)

- `ember2024_calibrated.pt`: 6,648-row weekly sample, 5 epochs, 33 family labels. Official test is temporal, not used to fit this checkpoint. Behavior metrics are null because EMBER does not supply strong observed behavior negatives; the behavior head is disabled.
- `ember2024_26k_mps.pt`: 26,000-row weekly sample, 8 epochs, 104 family labels (support threshold 10). Internal temporal test: family mAP 0.0153, top-1 0.348, top-3 0.376; a 10,000-row official-test development prefix: mAP 0.0134, top-1 0.480, top-3 0.520. This candidate was not promoted.
- `ember2024_26k_support50_mps.pt`: same 26,000-row sample and split, 25 family labels (support threshold 50). Internal temporal test: family mAP 0.0507 over 21 evaluable labels, top-1 0.461, top-3 0.526, macro-F1 0.0019. Malware recall on that internal test was 0.0031 at 1.06% FPR. The remaining official test rows are being used once as a final, prefix-excluded evaluation; no result is claimed yet.

These runs show that label-space/long-tail handling materially changes rankings, but family decisions are still too weak for acceptance and the malware auxiliary head is not calibrated robustly. The official full-test score above belongs only to the earlier 33-family checkpoint. The 10,000-row prefix is now development data and must not be included in the support-50 final holdout metric.

Dynamic trace evidence is still missing from real-data training. WinMET provides 31,844 CAPE traces with process/API/resource sequence detail, but its current Zenodo page shows no license value and reports roughly 13 GB compressed / 750 GB uncompressed; do not redistribute it or claim dynamic training results until licensing and a bounded extraction workflow are resolved. See [the official WinMET record](https://zenodo.org/records/16414116).
