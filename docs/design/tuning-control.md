# Tuning control: learning rate and weight decay (pre-registration)

Written 2026-11-01, before any run. Companion to `docs/design/pos-encoding-sweep.md`.

## Question

Is the in-distribution gap between the LSTM (34.57% test) and the sinusoidal Transformer (7.83%), and RoPE's gain over sinusoidal, robust to the Transformer's learning rate and weight decay?

## Why, with sources

- ACL Rolling Review's Responsible NLP checklist, item C2, asks whether a paper discusses "hyperparameter search and best-found hyperparameter values". The paper's Transformer ran at one learning rate.
- Csordás, Irie and Schmidhuber (EMNLP 2021, arXiv 2108.12284): default Transformer configurations understate systematic generalization; configuration changes alone move COGS from about 35% to 81%.
- Delétang et al. (ICLR 2023, arXiv 2207.02098) and Ruoss et al. (ACL 2023, arXiv 2305.16843): one learning-rate grid, {1e-4, 3e-4, 5e-4}, for every architecture, with the maximum over seeds and rates reported.
- Zhang et al. (arXiv 2501.08537, cited in the paper): initialization scale and weight decay decide between reasoning-based and memorized solutions. The paper's Transformer trains with plain Adam and no weight decay (`utils/trainer.py`, line 138).
- Jelassi et al. (arXiv 2306.15400): arithmetic Transformers trained with AdamW, learning rate 1e-5 to 1e-4, weight decay chosen from {1e-5, 1e-4, 1e-3, 1e-2}.
- The two architectures do not share a learning rate: LSTM 1e-3, Transformer 3e-4 (`train_v2.py`, `TRAINING_CONFIG_V2`).

## Held fixed

Everything in the sweep's protocol: Study 1 `datasets_v2`; 3 encoder and 3 decoder layers, d_model 256, 8 heads, feedforward 1024; batch 64; warmup 400 steps; cosine schedule to 1e-5; up to 200 epochs; patience 50; checkpoint selection and early stopping on validation accuracy; seeds 0, 1 and 2; one RunPod pod for every run in this control, an RTX PRO 6000 MIG 24GB slice on the same PyTorch 2.8.0 template as the sweep (no RTX 4000 Ada was available on 2026-11-01; the exact GPU is recorded in each run's environment.json); test accuracy from `scripts/eval/eval_test_splits.py --run-dir`. The LSTM is not retrained (1e-3, its paper checkpoints).

## Cells

24 new runs, 3 seeds per cell.

| Cell | Encoding | Optimizer | Learning rate | Weight decay | Priority |
|---|---|---|---|---|---|
| R1 | sinusoidal | Adam | 3e-4 | 0 | required (reference) |
| E1a | sinusoidal | Adam | 1e-4 | 0 | required |
| E1b | sinusoidal | Adam | 1e-3 (the LSTM's rate) | 0 | required |
| E3a | sinusoidal | AdamW | 3e-4 | 0.01 | required |
| R2 | RoPE | Adam | 3e-4 | 0 | extra (reference for E2a, E2b) |
| E2a | RoPE | Adam | 1e-4 | 0 | extra |
| E2b | RoPE | Adam | 1e-3 | 0 | extra |
| E3b | sinusoidal | AdamW | 3e-4 | 0.1 | extra |

R1 and R2 are the references. The existing 3e-4 runs were trained on an RTX 4000 Ada, and the 2026-10-29 hardware control measured a change of GPU and training trajectory moving test accuracy by up to 0.3 points, which is within the resolution at which reading 1 decides. Retraining the references on this pod makes every comparison same-GPU. The hardware-control retrain (`results_v2/pos_sweep/sinusoidal_retrain`, commit 2030fd5) and the sweep's RoPE arm (commit 01b5bb7) are reported beside R1 and R2, descriptively. 1e-3 sits above the published arithmetic range (Jelassi et al.) and is in the grid because it is the LSTM's rate. 0.01 is the top of Jelassi et al.'s weight-decay grid and Zhang et al.'s main setting. 0.1 is outside published weight-decay grids and follows Zhang et al.'s direction that larger weight decay helps; no source sets that value.

The 2026-10-29 retrain keeps its label "not used by any rule".

AdamW with weight decay 0 performs the same update as Adam, so E3a and E3b share the Adam reference R1. A unit test confirms that one optimizer step is identical.

## Per-run flags, fixed before any reading

- Fits: the training accuracy logged in `history.json` at the selected epoch is at least 80%. The paper's Study 1 Transformer checkpoints sit at 87–96%; the Study 2 checkpoints the paper describes as never fitting were selected at 5.85% and 14.51% and ended at 61.39% and 67.40%.
- Diverged: a non-finite loss, or validation accuracy never above the constant-output baseline (1.50% on Study 1 validation).
- Epoch cap: the run reached epoch 200 without early stopping.

Any run whose training accuracy at the selected epoch falls between 70% and 90% is flagged, and every reading that uses it is reported with and without it.

A cell is uninformative if fewer than 2 of its 3 seeds fit. It is reported as uninformative, not as a negative result.

## Readings

Test accuracy is the endpoint. Paired bootstraps resample test expressions (N_BOOT 1000, BOOT_SEED), seed-matched.

1. Tuning effect per cell: the difference from its reference cell (R1 for sinusoidal cells, R2 for RoPE cells), per seed. The cell "moves" accuracy if the 95% interval excludes 0 on every fitting seed (at least 2) in the same direction, with the count stated.
2. Best-over-grid, the Delétang and Ruoss convention: for each encoding, the maximum test accuracy over its learning rates and seeds, reported beside the mean over seeds at each learning rate.
3. The architecture gap under tuning: at the sinusoidal learning rate with the highest mean validation accuracy, LSTM minus Transformer per seed, paired by expression. "The gap holds" if the interval excludes 0 on every fitting seed (at least 2), with the count stated. Its size is reported either way.
4. RoPE's gain under tuning: at each encoding's learning rate with the highest mean validation accuracy, RoPE minus sinusoidal per seed. "The gain holds" if the interval excludes 0 on every fitting seed (at least 2), with the count stated.
5. Weight decay: reading 1 applied to E3a and E3b, with uninformative cells excluded.

Learning rates are selected on validation, never on test. Out-of-distribution accuracy (operations 4–7) is reported descriptively, as in the sweep.

## Reporting

Every test number appears beside its validation number. Best-over-grid values are labelled as maxima, and per-rate rows as means over seeds 0, 1 and 2. The best configuration per encoding, by mean validation accuracy, is named. Expected validation performance as a function of the number of configurations tried (Dodge et al., EMNLP 2019, arXiv 1909.03004) is reported per encoding, descriptively. Total GPU hours are reported for all runs in this control and for the reference runs.

## Scope and stop

No probe battery on these runs. No cells beyond these six, whatever the outcome; experiments freeze on 15 November 2026. The LSTM is not tuned: tuning only the trailing architecture is the conservative direction for the claim that the LSTM leads. Initialization scale, the other half of Zhang et al.'s complexity control, and model size go in Limitations.

## Compute

24 runs, two at a time with OMP_NUM_THREADS capped, at $0.70 per pod-hour; per-run time on this GPU is measured from the first runs. The required cells (R1, E1a, E1b, E3a; 12 runs) go first.

## Deviations

### 2026-11-01: The fit rule reads the maximum training accuracy over the run, not the value at the selected epoch

1. **What changes.** "Fits" becomes: the maximum training accuracy logged in `history.json` over the run is at least 80%. The 70–90% sensitivity band and the 2-of-3 cell rule apply to that measure unchanged. Each run's final-epoch training accuracy is reported beside its maximum, and a run whose maximum exceeds its final value by more than 10 points is flagged, because a maximum can be a transient spike. Over the 25 committed runs the largest such gap is 5.39 points (Transformer Study 2 seed 2, 72.79% maximum against 67.40% final), so the flag fires on none of them.
2. **Why.** The selected epoch is where validation accuracy peaks, and validation often peaks before training accuracy saturates. On runs committed before this control started, training accuracy at the selected epoch reads 78.85% and 70.44% for RoPE seeds 1 and 2 (test 18.0% and 21.5%), 55.02% for ALiBi seed 0 and 59.85% for learned seed 2, while the maximum training accuracy of those runs is 89.34–99.56%. The original rule would call the sweep's RoPE arm uninformative. Over the run, every committed Study 1 run reaches at least 84.86% (NoPE seed 0), and the two Study 2 runs the paper describes as never fitting reach 61.39% and 72.79%, so 80% still separates them.
3. **When.** Written after R1 seeds 0 and 1 finished, with training accuracy at the selected epoch of 77.83% and 76.56% (maxima 91.60% and 90.44%), and before any reading was computed. The evidence in item 2 comes from runs committed before this control started; the original rule was set without checking it against them. Its failure is a design error of the rule, not of the runs: it read fit at a point fixed by the validation peak, which on these runs comes before training accuracy saturates.
4. **Reporting.** Every cell is reported under both rules: the maximum-accuracy rule as primary, the selected-epoch rule as pre-registered. For each cell the results give its fit count and its informative or uninformative classification under each rule, and every reading is computed under each. Where a cell's classification or a reading's verdict differs between the rules, the text says so and names the cell.
5. **Indexing.** The selected epoch is the first index of the maximum of `val_accuracies` in `history.json`, because the trainer saves a checkpoint only on strict improvement (`utils/trainer.py`, line 208: `is_best = val_acc > best_val_accuracy`), so on a tie the first maximum is the one saved. Training accuracy at the selected epoch is read at that same index. The readings script asserts that this index equals the epoch stored in the checkpoint and that validation accuracy there matches the checkpoint's fingerprint. An earlier progress report read the hardware-control retrain's seed 1 one index early (78.79% at index 109; the selected index is 110, at 80.78%); the readings script was not affected.
6. **Correction.** The Per-run flags section says the paper's Study 1 Transformer checkpoints sit at 87–96% at the selected epoch. They sit at 92.25–96.25%; 87.45% is the LSTM's seed 0. The paper's own text already gives the right ranges (92–96% for the Transformer, 87–92% for the LSTM, in the Introduction, Results and Discussion), so the error is confined to this document.
