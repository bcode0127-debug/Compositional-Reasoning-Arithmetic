#!/usr/bin/env bash
# Tuning-control runner (design: docs/design/tuning-control.md). Six cells x seeds 0,1,2 on Study 1, transformer.
# Training flags are run_pos_sweep.sh's (--epochs 200 --patience 50, schedule and batch size from
# TRAINING_CONFIG_V2) plus --learning-rate and --weight-decay per cell. At most 2 runs at once, each with
# OMP_NUM_THREADS and MKL_NUM_THREADS = max(1, nproc/2).
#
#   scripts/training/run_tuning_control.sh --dry-run     # print the 18 train commands and exit
#   PYTHON=/path/to/python scripts/training/run_tuning_control.sh
#
# POD_IMAGE must be exported: train_v2.py records it in each run's environment.json.
set -uo pipefail  # deliberately NOT -e: one run's failure must not abort the rest

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
OUT_ROOT="results_v2/tuning_control"
LOG_DIR="$REPO_DIR/logs/tuning_control"
SUMMARY="$REPO_DIR/$OUT_ROOT/summary.txt"

# cell:encoding:learning_rate:weight_decay, in run order
CELLS=(E1a:sinusoidal:1e-4:0 E1b:sinusoidal:1e-3:0 E3a:sinusoidal:3e-4:0.01 E2a:rope:1e-4:0 E2b:rope:1e-3:0 E3b:sinusoidal:3e-4:0.1)
SEEDS=(0 1 2)

train_cmd() {
    # $1=encoding $2=lr $3=wd $4=seed $5=run_dir
    echo "$PYTHON -u train_v2.py --model transformer --study 1 --seed $4 --epochs 200 --patience 50 --positional_encoding $1 --learning-rate $2 --weight-decay $3 --output-dir $5"
}

if [ "${1:-}" = "--dry-run" ]; then
    for spec in "${CELLS[@]}"; do
        IFS=: read -r cell enc lr wd <<< "$spec"
        for seed in "${SEEDS[@]}"; do
            train_cmd "$enc" "$lr" "$wd" "$seed" "$OUT_ROOT/$cell/study1/seed$seed"
        done
    done
    exit 0
fi

cd "$REPO_DIR"
mkdir -p "$LOG_DIR" "$OUT_ROOT"
STATUS_DIR="$(mktemp -d)"
trap 'rm -rf "$STATUS_DIR"' EXIT
THREADS=$(( $(nproc) / 2 )); [ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

if [ -z "${POD_IMAGE:-}" ]; then
    echo "WARNING: POD_IMAGE is empty; environment.json will record pod_image as null" | tee -a "$LOG_DIR/runner.log"
fi

run_one() {
    # $1=index $2=cell $3=encoding $4=lr $5=wd $6=seed
    local idx="$1" cell="$2" enc="$3" lr="$4" wd="$5" seed="$6"
    local run_dir="$OUT_ROOT/$cell/study1/seed$seed" log="$LOG_DIR/${cell}_seed${seed}.log"
    local t0 t1 code
    t0=$(date +%s)
    $(train_cmd "$enc" "$lr" "$wd" "$seed" "$run_dir") > "$log" 2>&1
    code=$?
    if [ $code -eq 0 ]; then
        PYTHONPATH="$REPO_DIR" "$PYTHON" -u scripts/eval/eval_test_splits.py --run-dir "$run_dir" --study study1 >> "$log" 2>&1
        code=$?
    fi
    t1=$(date +%s)
    printf "%s\tseed%s\texit=%d\twall_seconds=%d\n" "$cell" "$seed" "$code" "$((t1 - t0))" > "$STATUS_DIR/$(printf %02d "$idx")"
}

idx=0
for spec in "${CELLS[@]}"; do
    IFS=: read -r cell enc lr wd <<< "$spec"
    for seed in "${SEEDS[@]}"; do
        while [ "$(jobs -rp | wc -l)" -ge 2 ]; do sleep 5; done
        idx=$((idx + 1))
        run_one "$idx" "$cell" "$enc" "$lr" "$wd" "$seed" &
    done
done
wait
cat "$STATUS_DIR"/* > "$SUMMARY"
echo "=== TUNING CONTROL COMPLETE $(date); summary $SUMMARY ===" | tee -a "$LOG_DIR/runner.log"
