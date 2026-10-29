# Saved for shutdown — 2026-10-29

User requested saving before powering off. Training PID 8622 was terminated
and verified absent. No new training/retraining is authorized. Do not restart
or extend training when resuming this chat without explicit user authorization.

## Preserved candidate

- Original: `models/winmet_resourcev4_expanded2500_seed44_e14.pt`
- Shutdown copy: `models/winmet_resourcev4_expanded2500_shutdown_20261029_e4.pt`
- SHA-256: `a564c9e8e903860c568541236d6f8c71add43bc29ea7b16a34d704eaa7857c03`
- Selected and saved epoch: 4. Epoch 5 stopped after batch 287/419;
  unsaved epoch-5 updates are not recoverable from this checkpoint.
- Validation top-1: 60.220994%; top-3: 72.928177%, on 181 eligible samples.
  This is conditional known-family weak-label validation, not test accuracy.
- Candidate is uncalibrated, not registered for UI, not suitable for promotion.
- Checkpoint contains weights/history, not a full optimizer/RNG resume state.
- Log: `artifacts/winmet/expanded_training_stopped_20261029.log`.
- Machine-readable snapshot: `artifacts/shutdown_checkpoint_20261029.json`.

## Completed and pushed

Latest application commit before saving: `27b28e7`. Improvements include exact
imphash indexing, bounded fuzzy evidence cache, bounded directory enumeration,
nonblocking file open, bounded attention link ranking, source process/time/event
provenance for cross-attention, and explicit conditional-family interpretation
in JSON/CSV and UI when the malware verdict is UNKNOWN.

Latest checks: model/batch 22 tests + 2 subtests passed; dashboard 8 tests passed.
Earlier complete suite: 329 tests + 94 subtests passed, before the latest changes.
UI static audit passed. Live browser verification unavailable because local
browser access is disabled. Do not claim full browser verification.

Synthetic mixed-batch inference exposed an unconfirmed virlock assignment to
minimal PE input while the malware verdict remains UNKNOWN. This is an unresolved
model limitation, not a correct malware-family classification.

## Next work without training

Continue inference/evaluation, attention explanation and false-positive analysis.
A proposed local setuptools/pip launcher control cohort has NOT been evaluated
in this turn. Its source files were found in the Homebrew Python 3.11 installation.
Do not report its outcomes or FPR without running the analysis. No malware was
executed or submitted externally. Do not read or print KEY_Secert.txt.

Project remains incomplete: hybrid top-1 above 85%, independent test evidence,
attention faithfulness and required flow ablations are not proven.
