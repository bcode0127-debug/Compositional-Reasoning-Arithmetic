#!/usr/bin/env bash
# Positional-encoding sweep runner (design: docs/design/pos-encoding-sweep.md, pre-registration 9365283).
# Arms learned / rope / alibi x seeds 0,1,2 on Study 1, plus the NoPE floor check (seed 0). The
# sinusoidal arm is the paper's checkpoints and is not retrained here.
#
# Training flags are run_matrix.sh's, unchanged (--epochs 200 --patience 50; learning-rate schedule
# and batch size are train_v2.py's TRAINING_CONFIG_V2 defaults); the only additions are
# --positional_encoding and, through it, the output root results_v2/pos_sweep/<arm>/.
# experiments/tests/test_pos_encoding_sweep.py asserts this against run_matrix.sh.
#
# Sequential and resumable, like run_matrix.sh. Usage:
#   scripts/training/run_pos_sweep.sh                 # every run
#   scripts/training/run_pos_sweep.sh rope:0          # one run (time the first run before the rest)
#   PYTHON=/path/to/python scripts/training/run_pos_sweep.sh learned:1 alibi:2
#
# POD_IMAGE must be exported (e.g. export POD_IMAGE=<container image tag>): train_v2.py records it in
# each run's environment.json. An empty POD_IMAGE logs one WARNING line to the sweep log.
set -uo pipefail  # deliberately NOT -e: one run's failure must not abort the sweep

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
SWEEP_LOG="$REPO_DIR/logs/pos_sweep.log"
SUMMARY="$REPO_DIR/results_v2/pos_sweep/sweep_summary.txt"

cd "$REPO_DIR"
mkdir -p logs results_v2/pos_sweep
touch "$SUMMARY" "$SWEEP_LOG"

is_complete_checkpoint() {
    # $1 = path to best_model.pt. Exit 0 if it loads and has every fingerprint field.
    "$PYTHON" - "$1" <<'PYEOF'
import sys
import torch
path = sys.argv[1]
try:
    ckpt = torch.load(path, map_location="cpu")
except Exception:
    sys.exit(1)
required = ["epoch", "val_loss", "val_accuracy", "seed", "dataset_version", "generator_commit", "positional_encoding"]
for k in required:
    if k not in ckpt or ckpt[k] is None:
        sys.exit(1)
sys.exit(0)
PYEOF
}

write_summary_line() {
    # $1=run_dir $2=tag $3=wall seconds (train + test eval) -> appends one line to $SUMMARY
    local run_dir="$1" tag="$2" wall="$3"
    "$PYTHON" - "$run_dir" "$tag" "$wall" <<'PYEOF' >> "$SUMMARY"
import sys
import json
import torch
run_dir, tag, wall = sys.argv[1], sys.argv[2], sys.argv[3]
ckpt = torch.load(f"{run_dir}/best_model.pt", map_location="cpu")
try:
    hist = json.load(open(f"{run_dir}/history.json"))
    epochs_run = len(hist.get("val_accuracies", []))
    train_s = hist.get("wall_time_seconds", "unknown")
except Exception:
    epochs_run, train_s = "unknown", "unknown"
try:
    test = json.load(open(f"{run_dir}/test_results.json"))["test_accuracy"]
    test = f"{test:.2f}%"
except Exception:
    test = "missing"
print(f"{tag}\tbest_val_accuracy={ckpt['val_accuracy']:.2f}%\tbest_epoch={ckpt['epoch']}\tepochs_run={epochs_run}"
      f"\ttest_accuracy={test}\ttrain_seconds={train_s}\twall_seconds={wall}")
PYEOF
}

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=(learned:0 learned:1 learned:2 rope:0 rope:1 rope:2 alibi:0 alibi:1 alibi:2 nope:0)
fi

model=transformer
study=1

echo "=== POS SWEEP START $(date) runs=${RUNS[*]} ===" >> "$SWEEP_LOG"
if [ -z "${POD_IMAGE:-}" ]; then
  echo "WARNING: POD_IMAGE is empty; environment.json will record pod_image as null" >> "$SWEEP_LOG"
fi

for run in "${RUNS[@]}"; do
  arm="${run%%:*}"
  seed="${run##*:}"
  tag="${arm}_study${study}_seed${seed}"
  run_dir="results_v2/pos_sweep/${arm}/study${study}/seed${seed}"
  ckpt_path="${run_dir}/best_model.pt"

  if [ -f "$ckpt_path" ] && is_complete_checkpoint "$ckpt_path"; then
    echo "SKIP $tag (complete checkpoint already exists)" | tee -a "$SWEEP_LOG"
    continue
  fi

  echo "=== START $tag $(date) ===" >> "$SWEEP_LOG"
  t0=$(date +%s)
  "$PYTHON" -u train_v2.py --model "$model" --study "$study" --seed "$seed" --epochs 200 --patience 50 --positional_encoding "$arm" >> "$SWEEP_LOG" 2>&1
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    printf "%s\tFAILED training (exit code %d)\n" "$tag" "$exit_code" >> "$SUMMARY"
    echo "=== FAILED $tag $(date) exit=$exit_code ===" >> "$SWEEP_LOG"
    continue
  fi

  # ID test accuracy: autoregressive, written beside ood_results.json (never the teacher-forced val metric)
  PYTHONPATH="$REPO_DIR" "$PYTHON" -u scripts/eval/eval_test_splits.py --run-dir "$run_dir" --study "study${study}" >> "$SWEEP_LOG" 2>&1
  eval_code=$?
  t1=$(date +%s)
  if [ $eval_code -ne 0 ]; then
    echo "=== TEST EVAL FAILED $tag exit=$eval_code ===" >> "$SWEEP_LOG"
  fi
  write_summary_line "$run_dir" "$tag" "$((t1 - t0))"
  echo "=== DONE $tag $(date) wall=$((t1 - t0))s ===" >> "$SWEEP_LOG"
done

echo "=== POS SWEEP COMPLETE $(date) ===" >> "$SWEEP_LOG"
