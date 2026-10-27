# Saved before shutdown — 2026-10-27

Work paused at the user's explicit request to shut down the Mac. Do not start
training automatically. First inspect saved artifacts, finish pipeline/UI tests
and explain accepted versus abstained family predictions. The user specifically
objected to repeatedly training without a usable verified deliverable.

## Preserved checkpoint

- Local path: `models/winmet_volume2_relv3_samplewindows_hybrid3_attention_1000_seed44_e15.pt`
- Bytes: 22,666,499; SHA-256: `e938b973a8b351cbd78ee76927288e68c8b0966d5336e110839dc6329a36af65`.
- Training intentionally interrupted by SIGINT in epoch 15, after batch 112/160.
- Epochs 1–14 completed. Latest observed best checkpoint was epoch 12, selected
  on validation mAP 0.6587045409655704; uncalibrated-threshold macro-F1 at that
  epoch was 0.4540105863635275. These are weak-label validation observations,
  **not** final calibrated metrics, accuracy, or a release claim.
- Checkpoint writes use temporary files and atomic replacement. Its hash was
  verified after the process exited (130). No optimizer/RNG state is stored:
  this is **not** an exact mid-epoch-resume checkpoint.
- End-of-run calibration, matched 15-epoch bag training and the final experiment
  JSON have **not** completed. Do not silently register this intermediate model
  or claim the planned 15-epoch comparison is done.
- Model training source revision: `9e4b91f`; fixed `per_sample_v1` windows,
  relation values off, 1,000-report reservoir from the verified v3 2,500-report
  cohort. Seeds/model/sample/split 44, d32, batch4, ranking .5, dropout0, MPS.
- Reference split: `artifacts/winmet/volume2_relv3_samplewindows_hybrid3_attention_1000_seed44.json`.
  Source SHA-256 `c6b13028505f7537dc39bd27a7bdb6850128675b46e3d0f0484adfb30f28882a`.

## Code and completed diagnostics

Signed value/readout decomposition and optional `--faithfulness-ranking
signed_readout` preserve classifier predictions. The same five validation
samples show Signature/event ranking gains but graph ranking remains negative;
see `docs/experiments/winmet-signed-readout-20261027.json`. Exact logit
reconstruction is not raw-input causality or a model-quality improvement.

Last fully completed suite: 231 tests + 67 subtests, before adding the deployment
audit. Two new deployment-audit tests and CLI integration are saved but the
latest full test process stalled on cold library imports and was terminated at
shutdown; rerun it. `git diff --check` passed before saving.

`--deployment-audit` now measures the actual guarded pipeline's accepted family
coverage, reference agreement, abstention reasons and per-file failures on the
verified validation split. It has not yet been run on the saved model. Unknown
WinMET references must never be treated as benign.

## UI/runtime

The old Streamlit process answered health checks but stopped updating UI.
After reload/restart its startup stalled on cloud-backed library reads. Some
runtime/source files have `compressed,dataless` flags. Finder Keep Downloaded
was enabled **only for `.venv`** (962 MiB apparent size); its menu changed from
`cmdPin` to `cmdUnpin` and download progress appeared. No whole dataset was
pinned. This is mitigation, not proof the UI issue is fixed.

Screenshot: `artifacts/runtime_keep_downloaded_20261027.png`.
Apple reference: https://support.apple.com/en-ie/guide/mac-help/mchl1a02d711/27/mac/27

The replacement server started successfully with file watching disabled but
has now been stopped at the user's shutdown request. Start only after resuming:

```sh
PYTHONPATH=src .venv/bin/python -m streamlit run Source_Code/hybrid_attention_dashboard.py \
  --server.port 8501 --server.address 127.0.0.1 --server.headless true --server.fileWatcherType none
```

Wait for actual health readiness before opening a fresh browser tab; stale IAB
tabs 1/2 are error pages. New tab 3 was still loading, not verified functional.
Do not bypass browser error-page URL policy. Do not change the registered UI
model merely because a diagnostic or training checkpoint exists.

Prepared end-to-end inputs: `artifacts/e2e_inputs_20261027.Sd0IdQ/` contains a
copy of the installed setuptools PE launcher (read-only analysis), a synthetic
injection CAPE fixture, and an intentionally invalid PE. The UI test on these
three files has **not** completed. No sample was executed. These are integration
inputs, not evidence of malware classification accuracy.

All three owned processes (training PID63920, dashboard PID70063, pytest
PID70066) were confirmed absent after shutdown signals. Other apps and user
processes were left untouched. Models/data remain local and are ignored by Git;
source and compact diagnostic records are committed/pushed separately.
