#!/bin/zsh
cd "/Users/Shared/iCloud Drive - Local/PPNCPLMD"
log() { print -r -- "$(date '+%F %T') $*" >> artifacts/chain_20261116.log; }
while kill -0 1933 2>/dev/null; do sleep 30; done
log "pass1 finished: $(tail -c 300 artifacts/avast/avast_ctu_v5_le128.log | tr '\n' ' ')"
[ -f artifacts/avast/avast_ctu_v5_le128.jsonl.manifest.json ] || { log "pass1 manifest missing, stop"; exit 1; }
PYTHONPATH=src nohup .venv/bin/python scripts/train_winmet_expansion.py artifacts/winmet/resourcev5_expansion_v2_manifest_20261113.json --checkpoint models/winmet_v5_behavior_d64_e10_bsel_full.pt --report artifacts/winmet/winmet_v5_behavior_d64_e10_bsel_full.json --epochs 10 --patience 3 --selection-metric behavior_map --d-model 64 --device mps --pre-encode --mps-empty-cache-every 50 --behavior-targets cape-signatures --encoding-cache artifacts/cache > artifacts/winmet/winmet_v5_behavior_d64_e10_bsel_full.log 2>&1 &
TP=$!; caffeinate -i -s -w $TP &
log "d64 started pid $TP"
until grep -q '"stage": "batch"' artifacts/winmet/winmet_v5_behavior_d64_e10_bsel_full.log || ! kill -0 $TP 2>/dev/null; do sleep 60; done
log "d64 reached training batches (or exited)"
R=Data/raw/avast_ctu
PYTHONPATH=src nohup .venv/bin/python scripts/prepare_avast_ctu.py $R/public_full_reports.zip $R/public_labels.csv --output artifacts/avast/avast_ctu_v5_128to512.jsonl --min-report-mib 128 --max-report-mib 512 --workers 1 --resume > artifacts/avast/avast_ctu_v5_128to512.log 2>&1 &
PP=$!; caffeinate -i -s -w $PP &
log "pass2 started pid $PP"
wait $PP; log "pass2 finished"
wait $TP; log "d64 finished"
