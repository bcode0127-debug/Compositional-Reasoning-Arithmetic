# Partial Competence and Limited Transfer
### Comparing LSTM and Transformer Generalization on Arithmetic Expressions

Code, data and results for the paper *Partial Competence and Limited Transfer: Comparing LSTM and Transformer Generalization on Arithmetic Expressions* (Jamanjyothi and Kasztelnik; manuscript, arXiv ID pending). The [Quickstart](#quickstart) checks the headline test-set numbers on a CPU without training, and [Where each number comes from](#where-each-number-comes-from) traces every reported figure to its file and the script that produces it.

---

## Abstract

We train character-level LSTM and Transformer encoder-decoders, three seeds each, to evaluate fully parenthesized integer arithmetic expressions. Study 1 tests generalization to longer expressions (2–3 → 4–7 operations). Study 2, reported as exploratory, shifts depth and tree shape together (balanced depth-2 → chain-shaped depth-3 trees at three operations). Under greedy autoregressive decoding, the LSTM shows partial competence: it reaches 34.6% exact match on an independently generated in-distribution test set (Study 1) and 30.5% (Study 2). Its accuracy falls steadily as expressions grow, from 14.8% at 4 operations to 2.1% at 7, and reaches 7.0% on the Study 2 shift. The Transformer stays near the floor throughout: 7.8% and 2.3% on the test sets, and a mean of at most 2.7% on any out-of-distribution split. Neither architecture transfers what it learns to substantially longer expressions at this data scale. The gap survives a positional-encoding sweep and a learning-rate and weight-decay control (see Further results). All checkpoints and datasets are released on the Hugging Face Hub.

---

## Results

*Values below are the canonical v2 numbers, read directly from `results_v2/matrix_summary.txt`,
each run's `ood_results.json`, and `results_v2/test_set_results.json`. Val and OOD are
mean [min-max] across seeds 0/1/2; Test is mean ± SD. All figures are autoregressive
exact-match (verified teacher-forced-equivalent under greedy decoding by `scripts/eval/cold_verify.py`).*

**Validation selected the checkpoints; the test split is independently held out and was
generated after training, so the Test column is the primary in-distribution result.**

**Study 1 (Operation-Count Generalization)**

| Model | Val | Test | OOD ops4 | OOD ops5 | OOD ops6 | OOD ops7 |
|---|---|---|---|---|---|---|
| LSTM | 35.7% [35.1-36.1] | 34.57% ± 0.95 | 14.8% [13.7-16.4] | 8.2% [7.5-9.5] | 2.7% [2.5-2.8] | 2.1% [1.8-2.5] |
| Transformer | 7.3% [6.8-7.5] | 7.83% ± 0.12 | 0.7% [0.4-1.2] | 0.6% [0.5-0.7] | 0.3% [0.1-0.5] | 0.4% [0.2-0.6] |

**Study 2 (Structural Generalization, exploratory)**

| Model | Val | Test | OOD |
|---|---|---|---|
| LSTM | 31.5% [30.1-32.5] | 30.47% ± 2.14 | 7.0% [6.3-7.8] |
| Transformer | 3.9% [3.7-4.2] | 2.27% ± 0.47 | 2.7% [2.0-3.1] |

Two of three Transformer runs did not fit the training data (maximum training accuracy over the run 61.39% and 72.79%, final epoch 61.39% and 67.40%, against the 80% fit threshold), so Study 2 does not support an architecture comparison.

## Further results

The sweep and the tuning control below are committed in this repository with their design docs; the manuscript draft predates them.

- **Probes** ([summary](experiments/results/analysis_v2/nonlinear_probe/summary.md)). On a stratified in-distribution split, a linear probe decodes running depth from the LSTM's cell state at 0.896–0.911 against a 0.527 majority. No probe decodes a running value in either model (R² 0.02 to 0.11, below the pre-set threshold). The Transformer's encoder loses linearly decodable position, from 0.995 at the input to 0.18 at layer 3. These results are correlational.
- **Positional-encoding sweep** ([design](docs/design/pos-encoding-sweep.md), [summary](experiments/results/analysis_v2/pos_sweep/summary.md)). RoPE raises the Transformer's in-distribution test accuracy over sinusoidal by 13.5, 10.3 and 13.6 points across seeds, and ALiBi by 2.5 to 5.2; learned absolute encoding does not move it. The LSTM still leads every arm.
- **Tuning control** ([design](docs/design/tuning-control.md), [readings](experiments/results/analysis_v2/tuning_control/readings.md)). At the validation-selected learning rate, the LSTM-minus-Transformer test gap is 25.8, 28.4 and 27.6 points, with 95% intervals above zero on all three seeds; RoPE's gain over sinusoidal holds at 11.8 to 12.6 points; weight decay 0.01 and 0.1 move nothing. Sinusoidal at 1e-4 does not fit within 200 epochs, and 1e-3 stops near the floor for both encodings, so those rates are not covered. Under the pre-registered selected-epoch fit rule these two comparisons are not evaluable; the design doc's 2026-11-01 Deviations entry explains the change.

---

## Quickstart

Checks the in-distribution test-set numbers from the released checkpoints, without training. On a CPU the whole sequence takes about 5 minutes (measured on a Mac mini).

```bash
git clone https://github.com/bcode0127-debug/Compositional-Reasoning-Arithmetic.git
cd Compositional-Reasoning-Arithmetic
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf download Ajay0728/Compositional-reasoning-arithmetic --include "*/best_model.pt" --local-dir results_v2
python -m scripts.eval.eval_test_splits --device cpu
git diff --stat results_v2/test_set_results.json
```

The evaluation prints each run's test accuracy and rewrites `results_v2/test_set_results.json`. An empty `git diff` means every per-example result matched the committed file.

To also reproduce the per-example results on the validation and out-of-distribution splits, run `python -m scripts.eval.eval_all_per_example`, then `git diff --stat results_v2/all_splits_per_example.json`. It decodes every split for all twelve checkpoints, about 15 more minutes on the same machine, and an empty diff again means every result matched.

Requirements: Python 3.9 or later. On Python 3.10 and later, `requirements.txt` installs scikit-learn 1.9.1, the version the probe grid ran on; on Python 3.9 it installs 1.6.1, the version the final-position addenda ran on.

## Where each number comes from

Run every command from the repository root. Each training run's directory is `results_v2/<model>/study<n>/seed<s>/`.

| Result | File | Produced by |
|---|---|---|
| Validation accuracy (Results tables) | `results_v2/matrix_summary.txt` and each run's `history.json` | `python train_v2.py --model lstm --study 1 --seed 0`, one run per model, study and seed |
| Out-of-distribution accuracy (Results tables) | each run's `ood_results.json` | `train_v2.py`, at the end of each run |
| Test accuracy (Results tables) | `results_v2/test_set_results.json` | `python -m scripts.eval.eval_test_splits --device cpu` |
| Per-example results on every split | `results_v2/all_splits_per_example.json` | `python -m scripts.eval.eval_all_per_example` |
| Per-seed results and paired tests | `results_v2/per_seed_and_paired_stats.json` | `python -m scripts.analysis.per_seed_and_paired_stats` |
| Comparison with the constant-output baseline | `results_v2/baseline_paired_stats.json` | `python -m scripts.analysis.baseline_paired_stats` |
| Chance baselines | `results_v2/chance_baselines.json` | `python -m scripts.analysis.chance_baselines` |
| Per-operation accuracy | `results_v2/item_8_9_10_stats.json` | `python -m scripts.analysis.item_8_9_10_stats` |
| Divisor distribution | `results_v2/divisor_distribution.json` | `python -m scripts.analysis.divisor_check` |
| Probes (Further results) | `experiments/results/analysis_v2/nonlinear_probe/summary.md` | `python experiments/scripts/nonlinear_probe.py --stage all` |
| Positional-encoding sweep (Further results) | `experiments/results/analysis_v2/pos_sweep/summary.md` | runs from `scripts/training/run_pos_sweep.sh` (GPU), then `python experiments/scripts/pos_sweep_battery.py --stage all` |
| Tuning control (Further results) | `experiments/results/analysis_v2/tuning_control/readings.md` | runs from `scripts/training/run_tuning_control.sh` (GPU), then `python experiments/scripts/tuning_control_readings.py` |

The scripts rewrite their own output files in place, so `git diff` shows whether a rerun matched the committed result.

## Full reproduction

### Data

The splits used in the paper are committed in `datasets_v2/`, pinned by SHA-256 in `datasets_v2/CHECKSUMS.txt` and mirrored on the [Hugging Face Hub](https://huggingface.co/datasets/Ajay0728/Compositional-reasoning-arithmetic-datasets). Verify them from the repository root with `shasum -a 256 -c datasets_v2/CHECKSUMS.txt` (macOS) or `sha256sum -c datasets_v2/CHECKSUMS.txt` (Linux). The train, validation and out-of-distribution splits come from `generate_study_datasets_v2()` in `data/generate_controlled.py`, and the test splits from `scripts/datagen/make_test_splits.py`.

### Training

Each of the twelve checkpoints is one run of `train_v2.py`, for each model (`lstm`, `transformer`), study (`1`, `2`) and seed (`0`, `1`, `2`). A run writes `best_model.pt`, `history.json` and `ood_results.json` into its directory under `results_v2/`, replacing the committed files there; pass `--output-dir` to write somewhere else. The released checkpoints are on the [Hugging Face Hub](https://huggingface.co/Ajay0728/Compositional-reasoning-arithmetic) and listed with their fingerprints in [CHECKPOINTS.md](CHECKPOINTS.md). `scripts/training/run_matrix.sh` is the record of how the original twelve runs were launched; it is kept unchanged, so the paths inside it point at the machine they ran on. The library versions of that training environment were not recorded, as the paper's Reproducibility paragraph states.

## Repository map

```
Compositional-Reasoning-Arithmetic/
├── train_v2.py          # Training entry point: one model, study and seed per run
├── data/                # Code: dataset generator, tokenizer, data loader, expression trees
├── datasets_v2/         # The splits used in the paper, with CHECKSUMS.txt
├── models/              # LSTM and Transformer encoder-decoders, positional encodings
├── utils/               # Training loop and early stopping
├── scripts/             # Pipeline scripts: evaluation, statistics, test-split generation, training runners
├── experiments/
│   ├── scripts/         # Probes, the positional-encoding sweep and the tuning control
│   ├── results/         # Their outputs, under analysis_v2/
│   ├── notebooks/       # analysis_v2.ipynb (current); analysis_plots.ipynb (pre-v2 record)
│   └── tests/           # Test suite (pytest)
├── results_v2/          # Per-run histories and out-of-distribution results, test results, statistics
├── docs/                # Design docs, decision log, retired results
├── legacy/              # Pre-v2 files kept as a record (see legacy/README.md)
├── CHECKPOINTS.md       # Registry of every trained checkpoint
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

Both script folders are current. `scripts/` runs the v2 pipeline, and `experiments/scripts/` holds the analyses built on it.

## Design docs and decisions

- [docs/design/pos-encoding-sweep.md](docs/design/pos-encoding-sweep.md) and [docs/design/tuning-control.md](docs/design/tuning-control.md): pre-registered designs, each with a dated Deviations section recording every change made after registration.
- [docs/DECISIONS.md](docs/DECISIONS.md): repository-level decisions.
- [CHECKPOINTS.md](CHECKPOINTS.md): one row per trained checkpoint, with its fingerprint.

---

## Experimental design

### Task

Models receive a fully parenthesized arithmetic expression as input and
must produce the correct integer result as output. The task is formulated
as a sequence-to-sequence problem with character-level tokenization.

```
Input:  ( ( 1 3 - 7 ) + ( 8 - 8 ) )
Output: 6
```
### Controlled studies

Study 1 varies operation count; Study 2 shifts depth and tree shape together:

**Study 1 - Operation-Count Generalization**
Train on expressions with 2–3 operations. Test on expressions with 4–7
operations. Tests whether learned computation rules extend to longer
expression chains.

**Study 2 - Structural Generalization (exploratory)**
Train on balanced depth-2 trees with three operations. Test on chain-shaped depth-3 trees with three operations. Depth and shape change together, so this study cannot isolate depth.
### Dataset specification

- Operand range: 1-20
- Division: D1 integer-only (no remainder, no division by zero). In `datasets_v2` every division divides by +1 or -1 ([divisor check](results_v2/divisor_distribution.json)), so division acts as identity or negation; the paper discloses this.
- Magnitude cap: |result| ≤ 10,000
- Format: Fully parenthesized infix notation
- Operator distribution: equal 25% per operator type
- Vocabulary: 20 character-level tokens
- Splits per study: 8,000 train, 1,000 validation and 1,000 test. Out-of-distribution: 1,000 per bucket, four buckets in Study 1 (4, 5, 6 and 7 operations) and one in Study 2.

### Model architectures

**LSTM encoder-decoder.** Bidirectional encoder, unidirectional decoder. Hidden dim 256, embedding dim 128, about 2.1M parameters.

**Transformer encoder-decoder.** 3 encoder and 3 decoder layers, 8 attention heads, model dim 256, sinusoidal positional encoding, about 5.5M parameters.

**Training, both models.** Adam with a peak learning rate of 0.001 (LSTM) or 3e-4 (Transformer), a 400-step linear warmup and cosine decay to 1e-5; teacher forcing; batch size 64; at most 200 epochs, with early stopping at a patience of 50 epochs on validation accuracy, keeping the checkpoint with the best validation accuracy. Three seeds per model and study. Input and output budgets are 64 and 12 tokens, above the longest sequence in the data; an over-length example raises an error instead of being truncated.

---

## Related work

- Lake and Baroni (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. Proceedings of the 35th International Conference on Machine Learning
- Stolfo et al. (2023). A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing
- Zhao et al. (2026). Shattered Compositionality: Counterintuitive Learning Dynamics of Transformers for Arithmetic. arXiv preprint arXiv:2601.22510
- Nanda et al. (2023). Progress measures for grokking via mechanistic interpretability. International Conference on Learning Representations
- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread
- Zhang and Nanda (2024). Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. International Conference on Learning Representations
- Makelov et al. (2023). Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching. Advances in Neural Information Processing Systems
- Lan et al. (2026). Make Mechanistic Interpretability Auditable: A Call to Develop Guidelines via Continuous Collaborative Reviewing. arXiv preprint arXiv:2606.00033
- Kazemnejad et al. (2023). The Impact of Positional Encoding on Length Generalization in Transformers. Advances in Neural Information Processing Systems
- Delétang et al. (2023). Neural Networks and the Chomsky Hierarchy. International Conference on Learning Representations

---

## License

Code: Apache-2.0 ([LICENSE](LICENSE)). The checkpoints and datasets on the Hugging Face Hub carry the same license.

## Citation
```bibtex
@misc{jamanjyothi2026partial,
  title   = {Partial Competence and Limited Transfer: Comparing LSTM and
             Transformer Generalization on Arithmetic Expressions},
  author  = {Jamanjyothi, Ajay and Kasztelnik, Karina},
  year    = {2026},
  note    = {Manuscript, arXiv ID pending}
}
```

Machine-readable citation: [CITATION.cff](CITATION.cff).

## Retired results

Results produced before the v2 pipeline were affected by an input-truncation bug and are retired, not corrected. They are kept, with the bug description, in [docs/LEGACY_RESULTS.md](docs/LEGACY_RESULTS.md); do not cite them. The pre-v2 files themselves are in [legacy/](legacy/README.md).
