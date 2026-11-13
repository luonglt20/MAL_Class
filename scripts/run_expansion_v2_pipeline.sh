#!/bin/zsh
# Unattended expansion v2 pipeline. Each stage is skipped when its output
# already exists, so the script can be re-run after an interruption.
#   1. wait for the verified WinMET volume 1 archive
#   2. prepare volume 1 (resource v4 normalization, read from the archive)
#   3. build the v2 manifest (volume 1 -> withheld_test_v2 / extra validation / train;
#      remaining volume 2 -> train)
#   4. wait for any training process passed via WAIT_PID, then run A/B variants one at a time
#   5. summarize validation results
set -euo pipefail
ROOT="${0:A:h:h}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
PY="$ROOT/.venv/bin/python"
W=artifacts/winmet
ARCHIVE=Data/raw/winmet/WinMET_volume_1.7z
V1_OUT=$W/volume1_all_blocks_resourcev4_20261111.jsonl
V2_REMAINING=$W/volume2_remaining_blocks_resourcev4_20261111.jsonl
V1_MANIFEST=$W/resourcev4_expansion_training_manifest_20261029.json
V2_MANIFEST=$W/resourcev4_expansion_v2_manifest_20261111.json
TAG=v2
VARIANTS=(baseline cosine_balanced)
log() { print -r -- "$(date '+%F %T') $*"; }

log "stage 1: waiting for verified $ARCHIVE"
until [[ -f $ARCHIVE ]]; do sleep 60; done

# prepare-winmet writes its manifest last, so the manifest marks a complete output.
if [[ ! -f $V1_OUT.manifest.json ]]; then
  [[ -f $V1_OUT ]] && mv $V1_OUT $V1_OUT.partial-$(date +%s)
  free=$(df -g . | tail -1 | awk '{print $4}')
  (( free >= 16 )) || { log "LOW DISK ${free}G"; exit 3; }
  log "stage 2: preparing volume 1"
  "$PY" -m malware_hybrid.cli prepare-winmet $ARCHIVE \
    --labels Data/raw/winmet/reports_consensus_label.json \
    --output $V1_OUT \
    --max-reports 6369 --max-solid-blocks 1000 --raw-budget-mib 400000 --output-budget-mib 9000 \
    --file-handle-resolution --sampling random-solid-blocks --seed 46 | grep -v '"progress"' | tail -c 2000
  [[ -f $V1_OUT.manifest.json ]] || { log "volume 1 prepare did not finish"; exit 4; }
else
  log "stage 2: $V1_OUT exists, skipped"
fi

if [[ ! -f $V2_MANIFEST ]]; then
  log "stage 3: building v2 manifest"
  "$PY" scripts/build_expansion_v2_manifest.py $V1_MANIFEST \
    --train-source $V2_REMAINING --second-test-source $V1_OUT --output $V2_MANIFEST
else
  log "stage 3: $V2_MANIFEST exists, skipped"
fi

if [[ -n "${WAIT_PID:-}" ]]; then
  log "stage 4: waiting for training process $WAIT_PID to finish"
  while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done
fi
log "stage 4: A/B variants ${VARIANTS[*]}"
"$PY" scripts/run_winmet_ab.py $V2_MANIFEST --tag $TAG --epochs 6 --patience 3 --only ${VARIANTS[@]}

log "stage 5: summary"
"$PY" scripts/summarize_winmet_ab.py --tag $TAG --output $W/winmet_ab_${TAG}_summary
log "pipeline finished"
