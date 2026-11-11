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
