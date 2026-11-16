#!/bin/zsh
# Next session: attention + behavior head on Avast-CTU, then the one-time locked test.
# Step 1 builds the encoding cache on its first run (about 2 h of preparation), then trains on MPS.
set -e
cd "$(dirname "$0")/.."
M=artifacts/avast/avast_ctu_time_manifest.json
PYTHONPATH=src .venv/bin/python scripts/train_winmet_expansion.py $M \
  --checkpoint models/avast_attention_d64_bsel.pt --report artifacts/avast/avast_attention_d64_bsel.json \
  --epochs 8 --patience 3 --selection-metric behavior_map --d-model 64 --device mps \
  --pre-encode --mps-empty-cache-every 50 --behavior-targets cape-signatures --encoding-cache artifacts/cache \
  > artifacts/avast/avast_attention_d64_bsel.log 2>&1
# Step 2 (validation only): compare with TF-IDF and freeze the selection.
PYTHONPATH=src .venv/bin/python scripts/fuse_attention_tfidf.py select $M models/avast_attention_d64_bsel.pt \
  --selection artifacts/avast/avast_selection.json --classical-report artifacts/avast/avast_classical_validation.json
# Step 3 (run once, after reviewing step 2): score the withheld time-ordered test.
# PYTHONPATH=src .venv/bin/python scripts/fuse_attention_tfidf.py test $M models/avast_attention_d64_bsel.pt \
#   --selection artifacts/avast/avast_selection.json --output artifacts/avast/avast_test_once.json --partitions withheld_test_time
