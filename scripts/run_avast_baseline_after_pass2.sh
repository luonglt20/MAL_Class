#!/bin/zsh
# After Avast-CTU pass 2 finishes: build the time-ordered manifest and run the TF-IDF + LR baseline (validation only).
cd "$(dirname "$0")/.."
log() { print -r -- "$(date '+%F %T') $*" >> artifacts/chain_20261116.log; }
until grep -q "pass2 finished" artifacts/chain_20261116.log 2>/dev/null; do sleep 60; done
[ -f artifacts/avast/avast_ctu_v5_128to512.jsonl.manifest.json ] || { log "pass2 manifest missing; avast baseline not started"; exit 1; }
PYTHONPATH=src .venv/bin/python scripts/build_avast_manifest.py artifacts/avast/avast_ctu_v5_le128.jsonl artifacts/avast/avast_ctu_v5_128to512.jsonl --output artifacts/avast/avast_ctu_time_manifest.json > artifacts/avast/avast_ctu_time_manifest.log 2>&1 || { log "avast manifest failed"; exit 1; }
log "avast time manifest built"
PYTHONPATH=src .venv/bin/python scripts/classical_family_baseline.py artifacts/avast/avast_ctu_time_manifest.json --output artifacts/avast/avast_classical_validation.json > artifacts/avast/avast_classical_validation.log 2>&1 && log "avast TF-IDF baseline finished" || log "avast TF-IDF baseline failed"
