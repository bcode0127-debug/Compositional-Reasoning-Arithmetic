# Checkpoint Registry

One row per trained checkpoint saved in this repo's history. `.pt` files themselves are gitignored (`*.pt`) and not committed — this table is the durable record of what each one is, alongside the Drive backup as the actual binary backup.

**Drive location: pending** for all rows below — Drive upload (STEP 3 of the commit/backup task) hasn't happened yet; this column will be updated in a follow-up commit once that's done.

| Path | Epoch | Val Accuracy | Test Accuracy | Val Loss | Seed | Dataset Version | Generator Commit | Drive Location |
|---|---|---|---|---|---|---|---|---|
| `results/transformer/study1/train_best_model.pt` | 25 | 7.0% | n/a (legacy, pre-v2) | 1.3723918236792088 | n/a (pre-dates seed tracking) | `datasets/` (legacy, pre-v2) | n/a (pre-dates generator_commit metadata) | pending |
| `results_v2/transformer/study1/seed0/best_model.pt` | 114 | 7.5% | 7.9% | 3.2949230819940567 | 0 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/transformer/study1/seed1/best_model.pt` | 160 | 6.8% | 7.7% | 4.223975971341133 | 1 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/transformer/study1/seed2/best_model.pt` | 160 | 7.5% | 7.9% | 3.9785922914743423 | 2 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/transformer/study2/seed0/best_model.pt` | 34 | 3.7% | 1.9% | 1.471701756119728 | 0 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/transformer/study2/seed1/best_model.pt` | 162 | 4.2% | 2.8% | 3.8162454068660736 | 1 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/transformer/study2/seed2/best_model.pt` | 51 | 3.8% | 2.1% | 1.5684214234352112 | 2 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/lstm/study1/seed0/best_model.pt` | 43 | 35.1% | 33.6% | 0.9785438999533653 | 0 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/lstm/study1/seed1/best_model.pt` | 46 | 35.9% | 35.5% | 0.9940367080271244 | 1 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/lstm/study1/seed2/best_model.pt` | 51 | 36.1% | 34.6% | 1.1237263418734074 | 2 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/lstm/study2/seed0/best_model.pt` | 57 | 32.5% | 31.8% | 1.1826078407466412 | 0 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/lstm/study2/seed1/best_model.pt` | 195 | 30.1% | 31.6% | 2.1283585280179977 | 1 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |
| `results_v2/lstm/study2/seed2/best_model.pt` | 66 | 32.0% | 28.0% | 1.3190548121929169 | 2 | `datasets_v2` | `f5044574212d2a41e5091456096e0841cf5ae6f8` | pending |

## Evaluation environment (test-set numbers)

The independent test-set accuracies above were produced by `eval_test_splits.py`
under a **freshly built venv**: `torch 2.13.0`, `numpy 2.4.6`, CPU device. Both
satisfy `requirements.txt` (`torch>=2.0`, `numpy>=2.0`), though those pins are
lower bounds rather than exact versions.

**Known gap in the artifact trail:** the venv the twelve checkpoints were *trained*
under lived in a temp directory that has since been reaped, and its exact torch /
numpy versions were never recorded - not in `matrix.log`, not in the canary logs,
and not in the checkpoint fingerprint fields (`epoch`, `val_loss`, `val_accuracy`,
`seed`, `dataset_version`, `generator_commit`). Training-time and evaluation-time
library versions therefore cannot be confirmed identical. The test-set evaluation
loads each checkpoint's saved weights and re-decodes from scratch, so it does not
depend on the training environment; but a strict bitwise reproduction of training
is not currently possible from this repo alone. Recording library versions in the
checkpoint dict would close this for future runs.

## Notes

- The legacy row (`results/transformer/study1/train_best_model.pt`) is the original Study 1 Transformer checkpoint recovered earlier in this branch's history from a separate backup source, verified against `results/transformer/study1/train_history.json`'s recorded best epoch (epoch 25, val_loss 1.3723918236792088 matches exactly). It predates the `seed`/`dataset_version`/`generator_commit` fingerprint fields added to `utils/trainer.py`'s checkpoint dict, and was trained on the original `datasets/` pipeline, not `datasets_v2`.
- All 12 `results_v2/` rows come from the 12-run seed matrix (`matrix_summary.txt`, `run_matrix.sh`), trained on the corrected pipeline (sequence-length hard asserts, padding masks, cosine LR decay, val-accuracy-based checkpoint selection/early stopping) against `datasets_v2/`.
- Every `.pt` file is gitignored; this table plus each run's `history.json`/`ood_results.json` (committed under `results_v2/`) is the reproducibility record until the Drive backup lands.
