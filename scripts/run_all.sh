#!/usr/bin/env bash
# End-to-end per-host repro: extract -> vectors -> auto-calibrate -> MC GPU task ->
# summary. One model per host, selected via GULA_MODEL_ID (set by the caller or a
# host-local env file). Resume-safe: each phase writes a sentinel and is skipped on
# re-run; gpu_task additionally checkpoints per trial.
#
#   GULA_MODEL_ID=Qwen/Qwen3.5-27B nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
#
# A monitor can watch results/REPRO_DONE (written last).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
[ -f env.host ] && source env.host          # optional host-local GULA_MODEL_ID / CUDA_VISIBLE_DEVICES
: "${GULA_MODEL_ID:?set GULA_MODEL_ID (or provide env.host)}"
export GULA_MODEL_ID
mkdir -p logs results
echo "=== run_all start $(date -u +%FT%TZ) model=$GULA_MODEL_ID cuda=${CUDA_VISIBLE_DEVICES:-all} ==="

run_phase () {  # $1=sentinel $2=logname $3...=command
  local sentinel="results/$1"; local log="logs/$2"; shift 2
  if [ -f "$sentinel" ]; then echo "skip: $sentinel exists"; return 0; fi
  echo "--- phase $sentinel: $* ($(date -u +%T)) ---"
  if "$@" > "$log" 2>&1; then
    date -u +%FT%TZ > "$sentinel"
  else
    echo "PHASE FAILED: $* (see $log; tail follows)"; tail -5 "$log"
    date -u +%FT%TZ > results/REPRO_FAILED
    exit 1
  fi
}

run_phase EXTRACT_DONE extract.log  python3 -m src.persona_extract
run_phase VECTORS_DONE vectors.log  python3 -m src.persona_vectors
run_phase CALIB_DONE   calib.log    python3 -m src.auto_calibrate
run_phase GPUTASK_DONE gpu_task.log python3 -m src.gpu_task --vectors gluttony sloth random --trials 5
python3 -m src.repro_summary > results/REPRO_SUMMARY.txt 2>&1 || true

date -u +%FT%TZ > results/REPRO_DONE
echo "=== run_all DONE $(date -u +%FT%TZ) ==="
cat results/REPRO_SUMMARY.txt
