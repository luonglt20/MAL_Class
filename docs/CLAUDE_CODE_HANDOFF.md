# MAL_Class handoff for Claude Code

Open this repository as the working directory. Read README.md and docs/SHUTDOWN_STATE_20261029.md first. The prior handoff was pushed at a4158b1. Preserve existing untracked user files and keep KEY_Secert.txt out of reads, logs and commits.

## Objective

Deliver a local UI that accepts Windows PE files and datasets, extracts features safely, and classifies malware families with an attention model. The performance target is top-1 accuracy above 85% with an independent evaluation. Report PE-only and hybrid results separately, with the label scope, denominator, abstention coverage and split provenance. Use trusted dataset sources and verify the complete workflow.

## Current state

- Launch UI with `PYTHONPATH=src .venv/bin/python -m streamlit run Source_Code/hybrid_attention_dashboard.py`. Run tests with `PYTHONPATH=src .venv/bin/python -m pytest -q`.
- Direct PE inference uses static features extracted in a restricted subprocess. Hybrid inference uses supplied execution reports. Uploaded binaries are treated as data and are never executed on the host.
- The shutdown copy `models/winmet_resourcev4_expanded2500_shutdown_20261029_e4.pt` is local. SHA-256 is `a564c9e8e903860c568541236d6f8c71add43bc29ea7b16a34d704eaa7857c03`. Epoch 4 had 60.22% top-1 and 72.93% top-3 on 181 eligible validation samples. Calibration is incomplete. The file stores weights and history, with no full optimizer/RNG resume state.
- The expanded split manifest is `artifacts/winmet/resourcev4_expansion_training_manifest_20261029.json`. Local model and dataset artifacts may be absent in a fresh clone. Check availability and checksums before evaluation.
- Eight distinct local pip/setuptools launcher controls returned UNKNOWN for verdict and family, with no family candidate. This small convenience cohort cannot establish a population false-positive rate. A synthetic minimal PE previously produced a virlock candidate with UNKNOWN malware verdict. The UI and exports distinguish those outcomes.
- The last dashboard suite recorded 8 passing tests. Browser verification of the latest wording remains pending. Re-run relevant checks against the current tree.

## Execution order

1. Inspect git status, the model registry, dataset provenance, manifest and checkpoint hashes. Verify registered static/hybrid routing with safe fixture inputs and exercise UI upload, dataset analysis, per-file errors, and export.
2. Evaluate existing weights against preserved validation groups. Reuse metrics and ablation code in `scripts/evaluate_winmet_flow_ablation.py`. The expansion evaluator requires a completed calibrated training report. For the stopped checkpoint, use a separately labeled exploratory validation run with explicit checksum and split checks. Keep the independent test untouched during model selection.
3. Compare baseline, cross-branch attention disabled, and graph edges masked while retaining graph nodes. Report accuracy, macro-F1 and coverage with family support. Track confidence and abstention on benign and unknown-family controls. Attention associations require separate faithfulness evidence.
4. The user previously prohibited restarting training. Continue evaluation and implementation under that constraint. Obtain explicit authorization before launching training. If authorized, record the new run configuration, choose changes on validation, calibrate on the designated calibration partition, then evaluate the selected frozen candidate on the independent test set.
5. Promote a checkpoint only after its provenance and measured behavior support it. Verify the normal UI path with that checkpoint, including file safety, failure handling and exports. Keep the goal open while the >85% target or required verification remains unmet.

## Epoch 4 validation ablation completed

The frozen checkpoint and expansion source matched their recorded checksums. All three runs used the same preserved validation partition (246 reports, 181 eligible for top-k). The independent test remains reserved.

| Configuration | Top-1 | Top-3 | Macro-F1 | mAP |
| --- | ---: | ---: | ---: | ---: |
| Full checkpoint | 60.22% | 72.93% | 24.13% | 33.55% |
| Cross-attention disabled | 60.22% | 70.72% | 14.38% | 30.07% |
| Graph edges masked, nodes retained | 57.46% | 72.93% | 19.88% | 29.73% |

These are inference-time ablations of the uncalibrated checkpoint on weak-label validation. Cross-attention improves macro-F1 and mAP in this comparison while aggregate top-1 is unchanged. Retaining graph edges improves top-1 by 2.76 percentage points. See artifacts/winmet/epoch4_shutdown_validation_ablation.json for metrics and fingerprints.

## 2026-11-11 Claude Code session

Checkpoint, manifest (`e9ca…4671`) and source (`503e…835a`) checksums matched; partition fingerprints were re-verified on load and the withheld test was never constructed.

### Exploratory validation (`artifacts/winmet/epoch4_shutdown_exploratory_selective_20261111.json`)

Script: `scripts/explore_stopped_checkpoint.py`. Validation 246 rows = 181 with a family in the 51-label vocabulary, 47 labelled with out-of-vocabulary families, 18 unlabelled. Top-1 uses the 181 eligible rows (41 families present). Thresholds were fitted on calibration (216 rows, 149 eligible) and only read on validation.

| View / readout | Val top-1 | Val top-3 | Selective top-1 at calibration threshold | Eligible coverage | OOV families accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid, model head | 60.22% | 72.93% | 87.78% | 49.7% (90/181) | 4/47 |
| Hybrid, frozen-embedding kNN-5 | 60.22% | 72.93% | 90.91% | 48.6% (88/181) | 8/47 |
| Hybrid without cross-attention, kNN-5 | 65.19% | 79.56% | 91.92% | 54.7% (99/181) | 7/47 |
| PE-only (events removed), model head | 21.55% | 34.81% | not reachable | – | – |
| PE-only, kNN-15 | 56.91% | 74.03% | 92.19% | 35.4% (64/181) | 11/47 |

Training-prior baseline top-1 is 22.10%. Selective accuracy is conditional on acceptance and is not the >85% full top-1 target. kNN readouts use train-partition labels with frozen weights; they were compared on validation only.

### PE upload fixes

- The local PE parser emitted EMBER-style static tokens while WinMET checkpoints learned CAPE `static.pe` tokens (44% known tokens on benign launchers). `src/malware_hybrid/static_dialect.py` rebuilds a CAPE static block from the parsed structure and tokenizes it with `CapeReportParser._static_evidence`; it is applied only for CAPE-vocabulary checkpoints (coverage 89%). Gaps: no PEiD/YARA locally, approximate string extraction.
- With correct tokens, the e4 model ranked malware families first (sigmoid ≈0.70–0.78) on 14/14 benign pip/setuptools launchers. Because PE-only validation top-1 is below the prior baseline, `pipeline.py` now abstains (`static_only_input_not_validated_for_dynamic_checkpoint`) when a checkpoint trained on sandbox reports receives static-only input, unless it records `static_only_validated`. EMBER static routing is unaffected.
- Live dashboard check with the e4 checkpoint on a safe folder (2 launchers, 2 malformed PE, 1 CAPE fixture): 5/5 processed, per-file errors retained, CSV button present. Full test suite: 334 passed, 94 subtests.

### Training evidence

The stopped run improved every epoch (validation loss 1.605→1.380→1.204→1.118; mAP 0.153→0.221→0.279→0.336). No training was launched in this session.

## 2026-11-13 session (state at shutdown)

### Results (validation-selected, test scored once)
- Manifest v2 (`artifacts/winmet/resourcev4_expansion_v2_manifest_20261111.json`): 8,500 train / 783 validation (670 with an in-vocabulary family) / 216 calibration / test v1 362 / test v2 779.
- Hybrid Attention, 6 epochs, `models/winmet_ab_v2_baseline.pt`: validation top-1 64.2%, top-3 78.1%, mAP 0.389. History: `docs/results/20261113/attention_v2_6epoch_history.json`.
- TF-IDF + one-vs-rest logistic regression (`scripts/classical_family_baseline.py`, all features, C=16): validation top-1 71.6%, top-3 85.7%, mAP 0.557.
- Late fusion (`scripts/fuse_attention_tfidf.py select`) chose attention weight 0 (also in group cross-fitting). Frozen selection: `docs/results/20261113/fusion_v2_selection.json`.
- Test scored once (`fuse_attention_tfidf.py test`, lock `artifacts/winmet/fusion_v2_selection.test_evaluated.lock`): TF-IDF 73.2% (v1, 302 rows) / 70.2% (v2, 668 rows); attention 61.3% / 64.1%. These test partitions are now spent for this selection; any later model must be reported as a second evaluation.
- The cosine_balanced A/B variant was dropped. The Mac sleeps on lid close; use `caffeinate -i -s -w <pid>` with the lid open on AC.

### Fixes
- `isolated_pe.py`: the PE sandbox now works from `/Users/Shared` by denying reads of the whole user directory (home or `/Users`) and re-allowing only `src/` and `.venv`; the probe also checks a real home file. Previously PE upload failed closed outside home.
- Slides and report corrected (suspected needs at least half a chain; static-only TF-IDF reaches 63.6%).

### Behavior direction (in progress)
- Census (`scripts/behavior_label_census.py`): the v4 prepared data has no CAPE TTP labels; rule labels are skewed (T1055 confirmed 3/8,500; T1547.001 never confirmed).
- Raw WinMET reports do contain 6-20 CAPE signatures each (no TTP field). T1547.001 failed because `handles.py` marked a Run-key value write (`...\Run\<value>` through the handle opened on `...\Run`) as CONFLICT.
- Normalization v5 (`cape-resource-relations-v5`, opt-in `registry_value_resolution` / `--registry-value-resolution`; v4 stays default): links a value write to its opened key, rejects writes whose path contradicts the handle (a v4 false-positive path), and stores behavioral CAPE signature names in `metadata["cape_signatures"]` (family/YARA/IDS detections excluded).
- On 300 raw reports: T1547.001 confirmed v4 0 -> v5 15, all 15 with `persistence_autorun`, 0 without (`docs/results/20261113/v4_v5_rule_check_300_raw_reports.log`; signature-to-technique pairing is by name, not an official CAPE mapping).
- v5 data prepared by replaying the v4 blocks and samples (`scripts/run_v5_reprepare.sh`): `artifacts/winmet/*resourcev5*.jsonl`. v5 manifest `artifacts/winmet/resourcev5_expansion_v2_manifest_20261113.json` (copy in `docs/results/20261113/`) reloads with every partition fingerprint identical to v4.

### v5 behavior results (2026-11-13, late)
- Census v5 (`docs/results/20261113/behavior_label_census_v5.json`): 8,464/8,500 train reports carry behavioral CAPE signatures; 218 kinds, 52 with >= 200 train rows. T1547.001 confirmed 0 -> 417; T1112 confirmed 1,540 -> 944 (stricter handle rule).
- `scripts/behavior_signature_baseline.py` (52 signatures from events only, TF-IDF + LR, C=4 fixed, validation): unigram mAP 0.796 / micro-F1 0.854 / macro-F1 0.696; with adjacent event pairs mAP 0.803 / 0.858 / 0.702. Pairs help 34/50 labels; paired bootstrap delta mAP +0.007 (95% CI +0.002..+0.013). Rule T1547.001 on validation: 41 confirmed, all with `persistence_autorun`; 128 signature reports missed.

- Attention with a behavior head on v5 signature targets (`--behavior-targets cape-signatures`, 6 epochs, `models/winmet_v5_behavior_signatures_e6.pt`): behavior mAP 0.626 / micro-F1 0.703 / macro-F1 0.599 on the same 52 labels and 782 validation reports, below the bigram baseline on all 52 labels; still rising each epoch (0.45 -> 0.63). Family on the same model: top-1 62.5%, mAP 0.413. History: `docs/results/20261113/attention_v5_behavior_signatures_history.json`.

- 10 epochs selected by behavior mAP (`--selection-metric behavior_map`, `models/winmet_v5_behavior_signatures_e10_bsel.pt`): epoch 10, behavior mAP 0.663 / micro-F1 0.727 / macro-F1 0.619 (beats the bigram baseline on 2/52 labels); family top-1 65.5%, mAP 0.464. Epochs 1-6 reproduce the 6-epoch run exactly. The curve flattens (0.646 -> 0.663 over the last 4 epochs).

- d_model 64 run (same settings, `--d-model 64`) started 2026-11-14 08:08 and was stopped at shutdown during epoch 1 (batch 1696/2125, no checkpoint yet). It fit in memory on MPS (swap ~1.8 GB) and its train loss at batch 1696 was 2.09 versus 2.37 for d_model 32. Log: `artifacts/winmet/winmet_v5_behavior_d64_e10_bsel.stopped_e1.log`. Restart command:
  `PYTHONPATH=src nohup nice -n 10 .venv/bin/python scripts/train_winmet_expansion.py artifacts/winmet/resourcev5_expansion_v2_manifest_20261113.json --checkpoint models/winmet_v5_behavior_d64_e10_bsel.pt --report artifacts/winmet/winmet_v5_behavior_d64_e10_bsel.json --epochs 10 --patience 3 --selection-metric behavior_map --d-model 64 --device mps --pre-encode --mps-empty-cache-every 50 --behavior-targets cape-signatures > artifacts/winmet/winmet_v5_behavior_d64_e10_bsel.log 2>&1 &` (about 50 min preparation + ~30 min/epoch; keep the lid open and run `caffeinate -i -s -w <pid>`).

### Family error analysis (2026-11-14)
- `scripts/family_cheap_boosts.py`: 28 training-light TF-IDF variants (C, trigrams, argument-free tokens, cosine kNN, rank fusion) do not raise validation top-1 (best 71.9% vs 71.6%; kNN 68.2%; rank fusion 70.4%). Results: `docs/results/20261113/family_cheap_boosts_v2.json`.
- `scripts/family_error_analysis.py`: top-1 89.1% for families with >= 200 train rows vs 53-58% below; imphash seen in train 73.8% vs unseen 59.2%; TLSH nearest neighbour 86.1% at distance <= 30 (26% coverage). Errors collapse into agenttesla/redline (snakelogger, taskun, gamarue, noon, msilzilla, injuke, amadey...). Results: `docs/results/20261113/family_error_analysis_v2.json`. `py-tlsh` (declared in the `fuzzy` extra) is now installed in `.venv`.
- Tried (validation, exact McNemar per report): TLSH routing at distance <= 30 (`scripts/tlsh_routed_hybrid.py`) 72.8% vs 71.6% (+18/-10, p=0.18); class-balanced LR and logit adjustment (`scripts/long_tail_boosts.py`) 71.5-71.8%, tau=1 drops to 62.1%; cross-fitted choice 72.7% (p=0.28). Conclusion: family top-1 is capped near 72-73% with the current features and labels.
- Proposed new approach: TLSH-routed hybrid (fix the threshold first, then verify), long-tail handling (more data, group-then-family, prototypes), separating packer from payload using CAPE payload data with family names removed.

### Next steps
0. Re-run the d_model 64 behavior run above and compare with d_model 32 (behavior mAP 0.663) and TF-IDF (0.803).
1. Behavior with attention: a larger model on a GPU or an architecture that uses longer event context; keep the bigram TF-IDF + LR (mAP 0.803) as the behavior baseline.
2. Map CAPE signature names to ATT&CK (needs the CAPE community signatures repository; ask before downloading), build behavior targets from signatures, train the behavior head on v5 and compare an order-free bag model with the ordered/graph model on behavior.
3. Train the family model on v5 (new checkpoint; v4 checkpoints reject v5 traces) and keep TF-IDF + LR as the main baseline.

Presentation sources: report doc https://claude.ai/artifact/SbeDLD93Gc2CWCsD3cTR4X ; slides https://claude.ai/artifact/5Uc72GtfmKPt9gmwg6xTZv (source copy in `docs/presentation/slides/`).
