#!/usr/bin/env bash
# Overnight 12-run seed matrix: {transformer,lstm} x {study1,study2} x {seed0,1,2}
# Sequential, resumable - safe to kill and relaunch at any point.
set -uo pipefail  # deliberately NOT -e: one run's failure must not abort the matrix

REPO_DIR="/Users/zsy/Desktop/Compositional-Reasoning-Arithmetic"
PYTHON="/private/tmp/claude-501/-Users-zsy-Desktop-Compositional-Reasoning-Arithmetic/72e84453-7e54-4e36-bf18-b297a07ea1ee/scratchpad/repo_venv/bin/python3"
BACKUP_DIR="$HOME/Desktop/matrix_backup"
MATRIX_LOG="$REPO_DIR/matrix.log"
SUMMARY="$REPO_DIR/matrix_summary.txt"

cd "$REPO_DIR"
mkdir -p "$BACKUP_DIR"
touch "$SUMMARY" "$MATRIX_LOG"

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
required = ["epoch", "val_loss", "val_accuracy", "seed", "dataset_version", "generator_commit"]
for k in required:
    if k not in ckpt or ckpt[k] is None:
        sys.exit(1)
sys.exit(0)
PYEOF
}

write_summary_line() {
    # $1=run_dir $2=tag  -> appends one line to $SUMMARY
    local run_dir="$1" tag="$2"
    "$PYTHON" - "$run_dir" "$tag" <<'PYEOF' >> "$SUMMARY"
import sys
import json
import torch
run_dir, tag = sys.argv[1], sys.argv[2]
ckpt = torch.load(f"{run_dir}/best_model.pt", map_location="cpu")
try:
    hist = json.load(open(f"{run_dir}/history.json"))
    epochs_run = len(hist.get("val_accuracies", []))
except Exception:
    epochs_run = "unknown"
print(f"{tag}\tbest_val_accuracy={ckpt['val_accuracy']:.2f}%\tbest_epoch={ckpt['epoch']}\tepochs_run={epochs_run}")
PYEOF
}

MODELS=(transformer lstm)
STUDIES=(1 2)
SEEDS=(0 1 2)

echo "=== MATRIX START $(date) ===" >> "$MATRIX_LOG"

for model in "${MODELS[@]}"; do
  for study in "${STUDIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      tag="${model}_study${study}_seed${seed}"
      run_dir="results_v2/${model}/study${study}/seed${seed}"
      ckpt_path="${run_dir}/best_model.pt"
      zip_path="${BACKUP_DIR}/${tag}.zip"

      if [ -f "$ckpt_path" ] && is_complete_checkpoint "$ckpt_path"; then
        echo "SKIP $tag (complete checkpoint already exists)" | tee -a "$MATRIX_LOG"
      else
        echo "=== START $tag $(date) ===" >> "$MATRIX_LOG"
        "$PYTHON" -u train_v2.py --model "$model" --study "$study" --seed "$seed" --epochs 200 --patience 50 >> "$MATRIX_LOG" 2>&1
        exit_code=$?

        if [ $exit_code -ne 0 ]; then
          printf "%s\tFAILED (exit code %d)\n" "$tag" "$exit_code" >> "$SUMMARY"
          echo "=== FAILED $tag $(date) exit=$exit_code ===" >> "$MATRIX_LOG"
          continue
        fi
        echo "=== DONE $tag $(date) ===" >> "$MATRIX_LOG"
      fi

      # Idempotent: covers both a fresh completion and a resumed/skipped run
      # that never got summarized/backed up before an earlier interruption.
      if [ -f "$ckpt_path" ]; then
        if ! grep -q "^${tag}[[:space:]]" "$SUMMARY" 2>/dev/null; then
          write_summary_line "$run_dir" "$tag"
        fi
        if [ ! -f "$zip_path" ]; then
          zip -rq "$zip_path" "$run_dir"
          echo "Backed up $run_dir -> $zip_path" >> "$MATRIX_LOG"
        fi
      fi
    done
  done
done

echo "=== MATRIX COMPLETE $(date) ===" >> "$MATRIX_LOG"
