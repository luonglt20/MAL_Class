# Resume: inference usability and saved-checkpoint audit

No new model training or LLM API requests were started. The saved epoch-12
checkpoint was checksum-verified, finalized with the **original calibration
partition only**, then evaluated with the actual guarded pipeline on validation.
All original partition fingerprints matched. Both in-memory and serialized
state tensors were checked unchanged. The source checkpoint remains untouched;
the calibrated output is a separate file.

[Compact evidence](experiments/winmet-epoch12-calibrated-audit-20261028.json):
validation family mAP/macro-F1 0.6587/0.4540, top-1/top-3 0.7778/0.9048. Ranking
covers 63 known-family validation rows, not all 107. The actual decision path
accepted 30/107 samples; only 25 of those 30 matched AVClass references. This
conditional 83.3% agreement is not release-quality evidence. Most other samples
abstain. The simultaneous conformal level must not be presented as accepted-set
precision. The UI registry is **not** changed to this candidate.

`scripts/calibrate_saved_winmet.py` reproduces this calibration-only procedure.
It requires an explicit source-checkpoint SHA-256, exact prepared source and
sidecar matches, all partition identities, matching architecture/tensorizer,
training priors and normalization provenance. It refuses existing outputs and
does not evaluate tests or train a baseline. Unit tests reject mismatched or
missing selection provenance and nonfinite validation histories.

## Browser integration reproduction

Using the Playwright CLI against localhost and the installed pinned research
models, the mixed local directory `artifacts/e2e_inputs_20261027.Sd0IdQ` processed
3/3 files: CAPE fixture and installed setuptools PE launcher were parsed, while
the deliberately invalid PE produced a per-file `PEFormatError`. CSV download
recorded hybrid/static routing respectively. Both valid inputs abstained at
family UNKNOWN; this demonstrates honest integration, **not** correct malware
classification. The fixture is synthetic, and the launcher is not a malware
family test. No executable was launched or externally submitted.

Real-browser testing found two usability defects:

1. Downloading CSV caused a Streamlit rerun, removing the just-computed table.
   The download now uses `on_click="ignore"`, also asserted in AppTest.
2. Some conformal family abstentions lacked an explicit top-level reason.
   Decisions now supply a reason without changing probabilities, prediction
   sets, family statuses or thresholds. CSV and single-sample UI expose it.

The server was restarted to load the changed Python module rather than relying
on a UI rerun. It binds 127.0.0.1:8501 and disables source-file watching for the
demo runtime. This does not by itself prove the former idle-session issue is
permanently fixed. Generated browser logs, screenshots and downloads are ignored
by Git. API key files remain ignored and were not opened or used.

Post-fix browser verification completed: downloading CSV leaves the 3/3 table
and its 2-parsed/1-error counters visible. The two valid inputs route to the
registered hybrid/static models; each abstains with
`no_confirmed_family_in_prediction_sets`. The invalid PE has
`input_parse_error`. No family statuses or probabilities were changed by this
usability fix.

- CSV: `.playwright-cli/mal-class-batch-predictions.csv`, SHA-256
  `9f5fad3f754abf9c3ea00cb58ddeac48be904c1aa5594c14f30829787bd64c1e`.
- Screenshot: `output/playwright/e2e-csv-retained-20261028.png`, SHA-256
  `cad03ff6e7161df37a7925c841a404ba458a12e81dced5a0780d4242f84aed3c`.
- Complete regression suite: **237 passed + 73 subtests**, one third-party
  Streamlit/Pandas deprecation warning; `git diff --check` passes.

## Remaining work before any completion claim

- Current-checkpoint flow/faithfulness controls, accepted-label error analysis,
  independent validation and benign false-positive qualification remain.
- Behavior neural head and independently reviewed evidence labels remain
  incomplete. A stronger family ranking score does not complete these tasks.
- Complete the actual EXE/dataset user-facing workflow, including broader input
  formats and malformed-input safeguards; do not substitute toy fixture success
  for real-family classification quality.
