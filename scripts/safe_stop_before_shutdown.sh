#!/bin/zsh
# At the given time, stop running jobs at safe points before a planned shutdown.
cd "$(dirname "$0")/.."
log() { print -r -- "$(date '+%F %T') $*" >> artifacts/chain_20261116.log; }
target=$(date -j -f "%Y-%m-%d %H:%M" "$1" +%s)
while [ $(date +%s) -lt $target ]; do sleep 60; done
L=artifacts/winmet/winmet_v5_behavior_d64_e10_bsel_full.log
for pid in $(pgrep -f "d64_e10_bsel_full.pt"); do
  # Only stop mid-epoch, so a checkpoint is never being written.
  for i in {1..120}; do tail -1 $L | grep -q '"stage": "batch"' && break; sleep 10; done
  kill $pid && log "safe-stop: d64 stopped mid-epoch at $(tail -1 $L | cut -c1-80)"
done
for pid in $(pgrep -f "prepare_avast_ctu.py|classical_family_baseline.py|build_avast_manifest.py"); do
  kill $pid && log "safe-stop: stopped $pid (Avast prep resumes with --resume; baseline/manifest rerun)"
done
log "safe-stop done"
