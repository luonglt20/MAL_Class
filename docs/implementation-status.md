# MAL_Class: evidence and remaining work

This is an independent multimodal research project. Legacy reports and scores are reference material, not experimental evidence for this project.

| Objective | Current evidence | Required next evidence |
|---|---|---|
| Pairwise relation/time bias | Executable model; gradient and elapsed-time counterfactual test | Real sandbox entity-resolution and flow ablations |
| Window overlap | Encoder and CLI support overlap, event folding | Padding invariance and long-report benchmarking |
| Fourier graph / causal pooling / pruning | Fourier encoder, bounded chronological multi-hop paths, duplicate-parent removal and conservative pruning | Contextual validation on large real graphs |
| Event-level cross-attention | Both directions query event/static tokens; inference trace test | Faithfulness on independent annotations |
| TLSH/ssdeep train prototypes | Train-only index, self-match exclusion, anonymized prototype tokens; optional official-compatible backends | Large-scale nearest-neighbor indexing and real-hash validation |
| Dashboard / Sankey / Navigator | Family×modality attention, execution graph, Sankey and Navigator 4.5 export | Visual QA on real reports and large-graph sampling |
| 15 evidence specifications | Fifteen multi-event rules; order/success/process/target guards and negative fixtures | Validate every rule on independently annotated reports; add handle resolution |
| Golden Set 50 per technique | Missing | Independent manual annotations with provenance; automated rules cannot substitute |
| STIX / OSCAL version locking | ATT&CK 19.2 STIX and NIST OSCAL content 1.2.1 URLs/SHA-256 locked; automated ID/name validation passes | Expert review of semantic crosswalk relationships |
| EMBER / MalBehavD pretraining | Missing | Dataset adapters, training artifacts and measured hard-negative iteration |
| Conformal / OOD | Per-family Mondrian split-conformal binary sets, persisted in checkpoint; four-way data split | Real held-out coverage/set-size evaluation and distribution-shift tests |
| Seven baselines / WinMET experiment | Seven executable protocol models share one tensor contract; no scores claimed | Identical leakage-controlled real-data runs and multi-seed report |
| Metrics | F1, balanced accuracy, masked mAP/per-label recall, localization, event Layer-IG and attention/IG rank agreement | Independent event annotations and measured acceptance results |

## Acceptance gates

All empirical gates remain unproven: family macro-F1 >= .88; ATT&CK mAP >= .82; hybrid improvement >= .02; Wilson 95% upper bound FPR <= .01; chain completeness >= .90; attention/IG agreement >= .75; shuffle macro-F1 relative decrease >= .15.

Even zero false positives among all 268 benign reports gives Wilson two-sided 95% upper bound about 1.41%, above 1%. At least 381 independent benign observations with zero false positives are needed for this particular bound. A validation subset needs its own denominator. Dataset access and labels, including benign versus undetected, must be verified from original sources.

No manual annotations, real-data accuracy or training completion are inferred from unit tests. Dataset roles in the objective are preserved: WinMET/CAPE for hybrid experiments, MALVADA for report processing, EMBER for static pretraining, MalBehavD for dynamic pretraining, Oliveira/Catak for sequence baselines, BODMAS for temporal drift and capa for auxiliary evidence comparison. Each requires source/schema/license verification before import; static capa findings alone do not establish observed dynamic execution chains.
