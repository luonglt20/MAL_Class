#!/bin/zsh
# Re-normalize the three v4 WinMET sources with the v5 schema, replaying exactly
# the same solid blocks and samples. Each output's manifest is written last.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
W=artifacts/winmet
LABELS=Data/raw/winmet/reports_consensus_label.json
log() { print -r -- "$(date '+%F %T') $*"; }
prep() {  # archive  v4_source  v5_output
  if [[ -f $3.manifest.json ]]; then log "$3 exists, skipped"; return 0; fi
  [[ -f $3 ]] && mv $3 $3.partial-$(date +%s)
  log "preparing $3"
  PYTHONPATH=src $PY -m malware_hybrid.cli prepare-winmet $1 --labels $LABELS --output $3 \
    --sampling replay --replay-from $2 --registry-value-resolution \
    --max-reports 100000 --raw-budget-mib 400000 --output-budget-mib 9000 | grep -v '"progress"' | tail -c 1500
  [[ -f $3.manifest.json ]] || { log "FAILED $3"; exit 4; }
}
V2=Data/raw/winmet/.volume2.pwgIP0/WinMET_volume_2.7z
V1=Data/raw/winmet/WinMET_volume_1.7z
prep $V2 $W/volume2_random_blocks_2500_seed44_resourcev4.jsonl $W/volume2_random_blocks_2500_seed44_resourcev5.jsonl
prep $V2 $W/volume2_remaining_blocks_resourcev4_20261111.jsonl $W/volume2_remaining_blocks_resourcev5_20261113.jsonl
prep $V1 $W/volume1_all_blocks_resourcev4_20261111.jsonl $W/volume1_all_blocks_resourcev5_20261113.jsonl
log "all v5 sources prepared"
