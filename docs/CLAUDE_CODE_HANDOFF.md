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
- Behavioral CAPE signatures as extra family features (`scripts/signature_feature_family.py`, v5): 71.8% vs 71.6% (+1/-0, p=1.0); signatures alone 64.6%, redundant with events. v5 baseline family top-1 equals v4 (71.64%).
- Unused WinMET data (`scripts/winmet_unused_census.py`): 2,098 of 12,738 reports were skipped because they exceed 64 MiB (median 103 MiB); 1,693 sit in train blocks but 1,298 of them are `vbclone`, and the most-confused families gain almost nothing. WinMET has only two volumes, so more WinMET data does not address the current errors.
- Remaining options need new information: more samples for rare families, unpacked payload data (CAPE section, family names removed), or handling packer-versus-payload AVClass labels (changes the task; must be reported).
- Proposed new approach: TLSH-routed hybrid (fix the threshold first, then verify), long-tail handling (more data, group-then-family, prototypes), separating packer from payload using CAPE payload data with family names removed.

### Next steps
0. Re-run the d_model 64 behavior run above and compare with d_model 32 (behavior mAP 0.663) and TF-IDF (0.803).
1. Behavior with attention: a larger model on a GPU or an architecture that uses longer event context; keep the bigram TF-IDF + LR (mAP 0.803) as the behavior baseline.
2. Map CAPE signature names to ATT&CK (needs the CAPE community signatures repository; ask before downloading), build behavior targets from signatures, train the behavior head on v5 and compare an order-free bag model with the ordered/graph model on behavior.
3. Train the family model on v5 (new checkpoint; v4 checkpoints reject v5 traces) and keep TF-IDF + LR as the main baseline.

Presentation sources: report doc https://claude.ai/artifact/SbeDLD93Gc2CWCsD3cTR4X ; slides https://claude.ai/artifact/5Uc72GtfmKPt9gmwg6xTZv (source copy in `docs/presentation/slides/`).

## 2026-11-14 shutdown state

- Submission set: `final/` (Word report with real TOC and page numbers, report PDF, 22-slide PDF, speaker-notes PDF, result JSON), pushed to `main`.
- d_model 64 behavior run (`models/winmet_v5_behavior_d64_e10_bsel.pt`, log `artifacts/winmet/winmet_v5_behavior_d64_e10_bsel.log`): stopped cleanly after epoch 6 of 10 (checkpoint verified loadable). The checkpoint holds epoch 6: family top-1 64.33%, family mAP 0.476 (d_model 32 at epoch 6: 62.54%, 0.413); behavior metrics are only in the final report, which was not written. It cannot resume (no optimizer state); a rerun of the same command restarts from epoch 1. At the same epoch d_model 64 beats d_model 32 on family mAP (epoch 4: 0.416 vs 0.331).
- Avast-CTU pass 1 (reports <= 128 MiB, `artifacts/avast/avast_ctu_v5_le128.jsonl.partial`) was about 45% done (21,500 / 48,309). Continue it with:
  `PYTHONPATH=src .venv/bin/python scripts/prepare_avast_ctu.py Data/raw/avast_ctu/public_full_reports.zip Data/raw/avast_ctu/public_labels.csv --output artifacts/avast/avast_ctu_v5_le128.jsonl --max-report-mib 128 --workers 2 --resume`
  then pass 2 (128-512 MiB, one worker, nothing else running): `... --output artifacts/avast/avast_ctu_v5_128to512.jsonl --min-report-mib 128 --max-report-mib 512 --workers 1`.
- Next on Avast-CTU: build a time-ordered split from `archive_group_id` (detection month), run the TF-IDF + LR baseline first, then attention and the behavior head with CAPE `cape_ttps`; report next to WinMET, never instead of it.

## 2026-11-17 overnight results

- Avast-CTU fully prepared with normalization v5: pass 1 (<= 128 MiB, 48,309 reports) and pass 2 (128-512 MiB, 613 reports); 54 reports > 512 MiB skipped; 0 family-name leaks. Files: `artifacts/avast/avast_ctu_v5_le128.jsonl`, `artifacts/avast/avast_ctu_v5_128to512.jsonl`.
- Time-ordered manifest `artifacts/avast/avast_ctu_time_manifest.json` (copy in docs/results): train <= 2019-06 (35,778), validation 2019-07..09 (4,796), calibration 2019-10 (3,717), withheld test 2019-11..2020-01 (4,631).
- TF-IDF + LR on Avast-CTU validation (`artifacts/avast/avast_classical_validation.json`): all features C=16 top-1 94.8%, top-3 99.3%, mAP 0.974, macro-F1 0.888; events only 94.9%; static evidence only 86.4-87.6%; prior 7.9%. Test not scored yet.
- Speed-ups without result changes: `--encoding-cache` (encoded data reused, losses identical in tests), streamed TF-IDF vectorization, fast `--resume` scan.
- d_model 64, 10 epochs, behavior-selected (`models/winmet_v5_behavior_d64_e10_bsel_full.pt`, encoding cache `artifacts/cache/`): epoch 10 family top-1 66.4%, family mAP 0.498, behavior mAP 0.696, behavior macro-F1 0.645 (d_model 32: 65.5% / 0.464 / 0.663 / 0.619; TF-IDF 71.6% / 0.557 / 0.803 / 0.702). Beats TF-IDF on 1/52 signatures. History: `docs/results/20261113/attention_v5_behavior_d64_e10_history.json`.
- Next: train attention + behavior head on Avast-CTU (uses `cape_ttps`/signatures; ~4x WinMET size), freeze the configuration on validation, then score `withheld_test_time` once.

- Ready-to-run next step: `scripts/run_avast_attention.sh` (training plus validation-only selection; the one-time test command is left commented until the selection is reviewed). Start it when the machine can stay on for several hours.

## 2026-11-17 evening: handover (Claude Code -> Codex)

- Running now: `scripts/run_avast_attention.sh` (PID 29673, started ~05:20; log `artifacts/avast/avast_attention_d64_bsel.log`). At 21:40 it was still in preparation (vocabulary/encoding of 35,778 train reports; swap ~6.9/8 GB). The Avast encoding cache is not saved yet (`artifacts/cache/` holds only the WinMET one). Do not start another heavy job while it runs. Check with `ps -p 29673` and `tail -2` of the log.
- When it ends, the script runs `fuse_attention_tfidf.py select` -> `artifacts/avast/avast_selection.json` (validation only). Review it against TF-IDF (94.8% top-1, mAP 0.974) before anything else.
- Only after that review: score `withheld_test_time` exactly once for both models (`fuse_attention_tfidf.py test ... --partitions withheld_test_time --output artifacts/avast/avast_test_once.json`, see the commented line in the script). Never rerun the test to pick a model.
- Then: add the Avast attention/behavior results to report section 8.4 and to this handoff, commit, push (only when the user asks).
- If the run fails or is killed: rerun `scripts/run_avast_attention.sh` after removing the partial checkpoint/report it refuses to overwrite (move them to Trash, do not hard-delete).
- Update 21:50: the first run (PID 29673) was thrashing swap (25 GB resident on a 16 GB Mac) in the encoding step: each `EncodedSample` is lists of Python numbers (~0.39 MB live), so 40k samples needed ~15.6 GB. `training.py` now stores pre-encoded samples zlib-pickled in `PackedReports` (~14 KB each, ~0.5 GB total; decoded samples compare equal, tests pass) and logs `encoding_train`/`encoding_validation` progress. The run was restarted (wrapper PID 44008, output `artifacts/avast/run_avast_attention.out`); the thrashed log is kept as `avast_attention_d64_bsel.thrashed_20261117.log`.

## 2026-11-17 Codex takeover and selection memory fix

- Codex read AGENTS.md, this handoff, README.md and the shutdown state directly in the local repository. The current chat now has write access. No second Codex chat was created (the earlier creation request was denied); takeover continues here.
- Verified live: wrapper PID 44008 and training child PID 44013. At elapsed 09:45, the child used ~2.63 GiB RSS and ~100% CPU; the log reached loaded_rows 14750. This is source loading, not encoding or an epoch. Earlier swap check showed ~2.12 GiB used, on AC and charging. No restart or extra training was launched. Recheck processes rather than relying on these snapshots.
- Fixed scripts/fuse_attention_tfidf.py: tfidf_scores now streams token documents into the existing vectorizer instead of retaining token lists for the complete train/evaluation corpus. Hyperparameters, event scope, train filtering, priors, evaluation grouping and fusion selection are unchanged. This also applies to the later one-time test command. The running shell imports this script only after training exits, so no training restart is needed.
- Validation: PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_fuse_attention_tfidf.py tests/test_classical_family_baseline.py -> 9 passed in 4.92s. Added regression coverage for deferred tokenization and exact score/group equality against materialized inputs across all/events/evidence, including an unsupervised training row. git diff --check passed.
- Verified the existing Avast baseline JSON: 35778 supervised train, 4796 validation, 10 labels; selected all features / C=16 by family mAP. Validation top-1 0.9482902418682235, top-3 0.9931192660550459, mAP 0.9735639759386526, macro-F1 0.8882869671008198. Its max_events is null (all events), whereas attention tensorization caps at 2048; report this context difference and do not attribute differences solely to architecture.
- Both docs/report/TAKA10_bao_cao.docx and TAKA10_bao_cao_word.docx currently describe Avast data in section 8.4 without the new results. No document has been edited or rendered in this takeover. Wait for training/report/validation selection, review the frozen selection, then perform the one-time withheld_test_time evaluation and update the report with verified metrics. No test partition was scored here. No commit/push was made. Preserve all unrelated dirty/untracked files.

- 2026-11-18 00:54 +0700 check: PID 44013 remains live (~185 CPU minutes at 3h12 elapsed), no selection/report yet. Log still ends at encoding_cache_saved. System swap rose to ~7.65 GiB. Native stack sampling shows torch.mps.empty_cache / MPSGraph cache destruction, so execution has reached MPS work despite sparse progress logs; do not classify this as a dead process. Sampled physical footprint ~14.4 GiB, peak ~17.5 GiB, much larger than RSS. No restart or second job launched; recheck at the next scheduled interval.

- 2026-11-18 17:24 +0700 Codex check: original PID 44013 exited after completing all 8 epochs and saving selected epoch 8. Failure occurred afterward in fit_family_conformal (MPS OOM), not in training; final report/selection absent. A separate Claude-started recovery is already running: PID 28092, scripts/finalize_calibration.py, log artifacts/avast/finalize_calibration.log; wrapper 28081 will then run validation selection into artifacts/avast/avast_selection.json with log artifacts/avast/avast_selection.log. Do not start duplicate calibration, selection or training. Recovery script and loader-only partition option were added externally; preserve those edits. No withheld test scored by Codex.

- 2026-11-18 report update requested by user while validation selection is running: created docs/report/TAKA10_BaoCao_cap_nhat_20261118.docx (also in the current Codex chat outputs). Based on final/TAKA10_BaoCao.docx, preserving the original. Updated overview, normalization versions, selection policy, hardware/calibration status, WinMET v5 d64 history, Avast temporal split and validation/behavior metrics, limitations and conclusions. Explicitly labels Avast test as pending; no invented test/fusion score. Preserved the real TOC field with verified cached page numbers; rendered all 19 pages with bundled LibreOffice and inspected every page. Builder in chat work/update_report.py; QA in work/report_render_final. After test finishes, produce a new version with verified test/fusion results rather than overwriting this snapshot. No commit/push.

- 2026-11-18 19:08 +0700: validation selection finished. Reviewed frozen selection: family mAP selects attention weight 0.0 (TF-IDF all/C=16), validation top-1 94.829%, mAP 0.973564 vs attention 94.183%, 0.921200. Group cross-fitting top-1 95.058%, fold weights 0.1/0.0; this is validation, not independent test. Confirmed selection status, null test_metrics, manifest hash, and test lock selection hash. Another agent already launched the one-time withheld_test_time evaluation, PID 33873, output artifacts/avast/avast_test_once.json, log artifacts/avast/avast_test_once.log. The lock already exists: DO NOT run test again or remove the lock. Await this process and result, then create an updated report snapshot with verified test/fusion metrics.

## 2026-11-18 Avast-CTU attention results (test spent)

- Training finished all 8 epochs (behavior_map selection, selected epoch 8) but the in-run calibration ran out of MPS memory. The selected weights were finalized with `scripts/finalize_calibration.py --device cpu` (same calibration functions, calibration partition only; uncalibrated copy kept as `models/avast_attention_d64_bsel.uncalibrated.pt`). On MPS the finalize also ran out of memory (inference cache growth), hence CPU.
- Validation (2019-07..09, 4,796): attention top-1 94.2%, top-3 96.9%, mAP 0.921, macro-F1 0.855; TF-IDF 94.8% / 99.3% / 0.974 / 0.888. Behavior head (87 evaluated CAPE signatures): mAP 0.456 -> 0.648 over 8 epochs, micro-F1 0.819, macro-F1 0.567. History: `docs/results/20261118/avast_attention_d64_bsel_history.json`.
- Frozen selection (`avast_selection.json`, family mAP): attention weight 0.0, i.e. TF-IDF alone.
- One-time test `withheld_test_time` (2019-11..2020-01, 4,631; `docs/results/20261118/avast_test_once.json`): TF-IDF (selected) top-1 90.1%, top-3 92.8%, mAP 0.953; attention top-1 84.0%, top-3 90.9%, mAP 0.881. Both drop under time drift; attention drops more (-10.2 vs -4.7 top-1 points). This test partition is now spent for family selection.
- Next: TF-IDF behavior baseline on Avast validation (fair comparison for the behavior head), then report section 8.4.

- Sleep handoff (2026-11-18 22:38 +0700): verified calibrated epoch-8 backup `models/avast_attention_d64_bsel.epoch8_sleep_20261118.pt` (498613903 bytes, SHA-256 `a5e936bd8bbc2d8b4f165f15e87c2a73a9f1960f0a7af054ca05c591bd84e570`) matches the checkpoint recorded in the frozen selection. Checked selection hash against both the independent test result and the lock. Family test is complete and must not be rerun. Separate validation-only behavior baseline was live as PID 37334 before sleep (log `artifacts/avast/avast_behavior_signature_baseline.log`). On waking, check that process and its output before doing any new work.

- Report with verified one-time family test results: `docs/report/TAKA10_BaoCao_ket_qua_test_20261118.docx`. Updated the overview, section 8.4, limitations and conclusion. Attention top-1 84.02%, selected TF-IDF top-1 90.11%, n=4,631, frozen attention weight 0. Verified selection SHA-256 against the result and lock. Rendered 19 pages; reviewed the five changed pages and verified the other 14 are byte-identical to the reviewed render. Comparison against the validation-only behavior baseline remains pending. Clarification: the epoch-8 history JSON records 87 evaluable CAPE signature labels and 4,763 annotated validation reports; use these counts for the reported attention metrics.

- Behavior baseline follow-up: sampled active PID 37334 at ~14.6 GiB physical footprint. The running process uses the original implementation and was left alone. Updated `scripts/behavior_signature_baseline.py` for subsequent runs to stream tokenized documents into TF-IDF, skip building unused calibration report objects and log each fit start. The behavior baseline protocol and model settings are unchanged. Focused behavior/TF-IDF/loader suites: 9 passed. If the current process finishes, use its result and compare epoch-8 Attention and the baseline on shared CAPE labels before revising the report.

## 2026-11-19 Family classification priority

- The thesis centers on malware-family classification with Hybrid Attention. CAPE signature prediction is an auxiliary validation experiment. TF-IDF remains a comparison baseline for family classification. The Avast independent family test remains spent (Attention 84.02%, TF-IDF 90.11%, 4,631 reports). Do not use it for another round of model selection.
- Epoch 8 Avast validation, thresholded per-label metrics: Emotet recall 60.0% (380 positives), Ursnif recall 60.3% (73 positives), Zeus precision 36.4% (48 positives), and Swisyn accounts for 2,410 of 4,796 positive family annotations. These are validation diagnostics, not top-1 confusion counts or test statistics.
- Enabled the existing expansion ablation evaluator to accept the complete recovered Avast report and to load train/validation in packed form. Preflight passed using the actual frozen checkpoint hash and recovered report. Evaluator, loader and dashboard tests: 12 passed. Once the running behavior process finishes, run the following without a concurrent heavy job. This is inference on validation using the existing weights and includes full-model, disabled cross-attention and masked graph-edge conditions. Keep Apple MPS memory limits intact.

```sh
PYTHONPATH=src .venv/bin/python scripts/evaluate_winmet_expansion.py \
  artifacts/avast/avast_ctu_time_manifest.json \
  artifacts/avast/avast_attention_d64_bsel.json \
  models/avast_attention_d64_bsel.pt \
  --checkpoint-sha256 a5e936bd8bbc2d8b4f165f15e87c2a73a9f1960f0a7af054ca05c591bd84e570 \
  --device cpu --batch-size 4 \
  --output artifacts/avast/avast_family_attention_ablation_validation.json
```

### User-confirmed acceptance scope

- Main deliverable: a local antivirus-style scanner that predicts malware families using deep attention. The Hybrid Attention and cross-attention experiment is the central thesis result. Inputs are prepared data or execution reports and uploaded Windows PE/EXE files. The visible output names the candidate family, its score or an explicit UNKNOWN result, and which model handled the input.
- Keep the demo honest about available evidence. An EXE alone is statically parsed in isolation and routed to the Signature–Static Attention checkpoint. A compatible report with observed events supplies the full hybrid branches. Do not execute uploaded malware or fabricate an execution trace. The current default hybrid registry entry is WinMET v4, distinct from the Avast v5 research checkpoint used in the reported benchmark.
- Verify both input paths in the actual dashboard before handoff, using benign or synthetic PE fixtures and prepared report fixtures. Check family output or explicit abstention, routing/model identity, per-file errors, and batch CSV. Keep coverage and test data provenance in the final report. Treat CAPE behavior prediction as supplemental; the family scanner and its attention contribution are the priority.

### Defer the expensive behavior comparison

- User prioritized the family scanner and allowed the costly behavior experiment to move to a later phase. Preserve `artifacts/avast/avast_behavior_signature_baseline.log`. It contains a completed unigram validation run (mAP 0.72098, micro-F1 0.86829, macro-F1 0.64476). Bigram had no result after more than 3h40 elapsed and system swap was ~13.7 GiB. Sent SIGTERM to the matching baseline process before starting family ablation. Keep the unigram as a preliminary auxiliary result. If the comparison is revisited, use the streaming implementation and compare on matched CAPE labels.

- Family ablation started on CPU as PID 45168 (Codex execution session 63367), logging to `artifacts/avast/avast_family_attention_ablation_validation.cpu.log`. Previous behavior PID is gone. Check the new job before any restart.

- Dashboard smoke check (2026-11-19): used Streamlit AppTest on the actual dashboard with the installed checkpoints from the default registry. Both local-folder and batch-upload flows passed on a synthetic PE, a CAPE fixture and malformed JSON. The synthetic PE used the EMBER Signature–Static checkpoint and showed a conditional family (virlock) with UNKNOWN malware verdict. The CAPE fixture used the WinMET Hybrid checkpoint and returned UNKNOWN verdict and primary family, with a top-1 candidate and score. Malformed JSON produced a per-file error and CSV download contained all three rows with correct model IDs. These fixtures verify routing and output semantics, not detection accuracy. Avast ablation remains a separate validation experiment.

- Batch dashboard and CSV now show verdict, family, family score and model route together near the front. The batch caption explains that a family shown beside UNKNOWN is conditional. Dashboard suite: 8 passed.

### Avast family ablation results and report

- Completed the CPU validation ablation on 4,796 reports (all eligible, 10 families). Full Hybrid Attention: top-1 94.18%, mAP 0.9212. Disable cross-branch attention: 92.66%, 0.9162. Mask graph edges and retain graph nodes: 93.87%, 0.9134. Differences relative to full top-1: -1.52 and -0.31 percentage points. Family macro-F1 at the default threshold was 0.8547, 0.8649 and 0.8555 respectively. With the original calibration thresholds it was 0.8844, 0.7621 and 0.8408. These are inference-time results on one checkpoint, not a test-set or retraining ablation.
- Archived the result JSON in `docs/results/20261118/avast_family_attention_ablation_validation.json`. Verified manifest and checkpoint hashes and validation-only status. The previous independent test remains 84.02% top-1 on 4,631 reports for Attention. The >85% independent-test research target remains open.
- Report: `docs/report/TAKA10_BaoCao_Attention_20261119.docx` and matching chat output. Updated section 8.4 and the conclusion. Rendered all 19 pages. Pages 1–16 match the already-reviewed render byte for byte, and pages 17–19 were visually checked. Expensive behavior comparison remains deferred. The dashboard uses the research routing entries described above. No commit or push.

### Evaluation audit for the two scanner inputs

- The static registry SHA-256 matches the checkpoint used in the historical EMBER2024 internal temporal holdout report `artifacts/ember_hybrid3_sigstatic_nodrop_34k_d64_seed42_e15_internal_holdout.json`. Top-1 is 89.45% on 1,005 in-vocabulary family rows out of 5,045 holdout rows. This measures prepared EMBER features from a holdout inside the training archive. The official later-time archive was unopened by that run. Do not present this as measured accuracy on arbitrary uploaded EXE files or as a qualified malware verdict.
- The Avast Hybrid Attention independent test produced 3,891 correct top-1 values out of 4,631 eligible reports. Strictly exceeding 85% at that sample count would require 3,937 correct, a gap of 46. This is a description of the existing result, not a target for tuning on the spent test. The available MOTIF source is an EMBER feature dataset, so it cannot independently measure the full dynamic hybrid path.
- Validation-only ablation effects vary by family. Full minus cross-disabled AP rises most for njrat (+0.0952, 418 positives) and emotet (+0.0792, 380), but drops for ursnif (-0.2235, 73). Use class-level metrics alongside aggregate top-1 when diagnosing any later model. A fresh independently held-out cohort with compatible dynamic features and family labels is required for a new hybrid milestone claim after model development.

### External hybrid test cohort feasibility

- Audited only the public WinMET consensus-label metadata from [Zenodo record 16414116](https://zenodo.org/records/16414116), matching its published checksum. The published first/last SHA-256 values partition volumes 3–5 into 6,369, 6,369 and 6,368 label records. These boundaries are identifier ranges, not chronological cuts.
- Exact case-insensitive matches to the 10 frozen Avast family names are sparse. Across volumes 3–5, AVClass matches 29/30/27 records (86 combined, five families). CAPE matches 76/78/70 (224 combined, nine families), of which 104 are qakbot. These are metadata counts before cross-source deduplication, schema checks or label reconciliation. No report archives or predictions were used for this audit.
- WinMET volumes 3–5 can inform an explicitly scoped cross-source stress test. The exact-name cohort is too sparse and imbalanced to replace the 10-family Avast acceptance benchmark. To continue toward a new >85% claim, obtain a representative untouched cohort of compatible static + dynamic reports with labels and deduplicate it against all prior splits. Keep that cohort sealed while selecting any candidate on validation.
- Avast behavior baseline (validation, `docs/results/20261118/avast_behavior_signature_baseline_full.json`): TF-IDF + LR over events, unigram mAP 0.721 (micro-F1 0.868, macro-F1 0.645), bigram mAP 0.730 (0.874 / 0.648); bigram - unigram paired bootstrap ΔmAP +0.009, CI95 [0.005, 0.015]. Attention behavior head: mAP 0.648. Event order helps on Avast too, but the linear baseline still beats the attention head.

## 2026-11-19 Handoff to local Claude Code

- Start in the PPNCPLMD checkout and read `AGENTS.md`. Keep the thesis centered on family classification with Hybrid Attention and cross-attention. The demo routes `.exe` to restricted macOS static PE parsing and EMBER Attention. CAPE JSON with execution evidence uses the registered WinMET Hybrid Attention checkpoint. The Avast research checkpoint supplies the thesis benchmark separately. The demo never executes uploaded PE files.
- The frozen Avast test is complete. On 4,631 eligible reports, Hybrid Attention reached 84.02% top-1 and TF-IDF reached 90.11%. On validation, full versus cross-attention disabled reached 94.18% versus 92.66%. See `docs/results/20261118/` and `docs/report/TAKA10_BaoCao_Attention_20261119.docx`.
- Claude bridge review recommended verifying the lock and split provenance. The local manifest SHA-256 matched the archived selection, and the archived selection SHA-256 matched both the local test lock and archived result. The gap between validation and test alone cannot establish leakage. Preserve the lock. A new >85% claim needs a representative untouched cohort with compatible reports and family labels, split and deduplicated against development data before evaluation. PE-only accuracy on real labeled `.exe` inputs remains to be measured separately.
- The full auxiliary TF-IDF behavior baseline result is now available in `docs/results/20261118/avast_behavior_signature_baseline_full.json` (validation signature mAP 0.7210 unigram and 0.7302 bigram). Preserve the result. Further behavior work is deferred, and new training requires user authorization. Report meaningful changes only and keep any live job checks 30–60 minutes apart.
