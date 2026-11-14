#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/khoa/Projects/New-Method-Research
BASE=experiments/sm13_three_directions_20261113
DATA=$BASE/data_expansion
cd "$ROOT"
for spec in '1 101' '1 202' '2 101' '2 202' '3 101' '3 202'; do
  read -r fold seed <<<"$spec"
  log="$DATA/logs/fit_fold${fold}_seed${seed}.log"
  echo "START fold=$fold seed=$seed $(date -u +%FT%TZ)" | tee -a "$DATA/logs/remaining_fit_queue.log"
  flock -x "$BASE/gpu.lock" -c "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python $DATA/train_eval.py --fold $fold --seed $seed > $log 2>&1"
  echo "DONE fold=$fold seed=$seed $(date -u +%FT%TZ)" | tee -a "$DATA/logs/remaining_fit_queue.log"
done
