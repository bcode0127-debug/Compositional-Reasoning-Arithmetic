# Legacy (pre-v2) files

Everything in this folder belongs to the retired pre-v2 pipeline, whose evaluation path silently truncated encoder input to 20 characters. The full account is in [docs/LEGACY_RESULTS.md](../docs/LEGACY_RESULTS.md). No number produced here should be cited.

Each item keeps its original path, with `legacy/` in front:

| Now | Before | What it is |
|---|---|---|
| `legacy/main.py` | `main.py` | Pre-v2 entry point (generate, train, eval) |
| `legacy/datasets/` | `datasets/` | Pre-v2 splits |
| `legacy/results/` | `results/` | Pre-v2 training histories and evaluation results |
| `legacy/figures/` | `figures/` | Pre-v2 figures and `results_summary.csv` (teacher-forcing numbers) |
| `legacy/experiments/results/` | `experiments/results/` | Pre-v2 attention, cross-attention, head-ablation, operation-breakdown and failure-trace outputs |

Three pre-v2 code files stay where they are: `experiments/analysis.py`, `experiments/config.py` and `experiments/notebooks/analysis_plots.ipynb`. `analysis.py` imports `experiments.config`, and no current code imports either, so they are kept in place as records of the retired analysis rather than moved.

The files in this folder are unchanged; only their location moved, on 2026-11-16. Paths written inside them still refer to the old locations, so they are not runnable from here without edits. The current pipeline is `train_v2.py` with `datasets_v2/`; see the [README](../README.md).
