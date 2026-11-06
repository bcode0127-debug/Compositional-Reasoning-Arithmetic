# Partial Competence and Limited Transfer
### Comparing LSTM and Transformer Generalization on Arithmetic Expressions

---

## Abstract

We train character-level LSTM and Transformer encoder-decoders, three seeds each, to evaluate fully parenthesized integer arithmetic expressions. Study 1 tests generalization to longer expressions (2–3 → 4–7 operations). Study 2, reported as exploratory, shifts depth and tree shape together (balanced depth-2 → chain-shaped depth-3 trees at three operations). Under greedy autoregressive decoding, the LSTM shows partial competence: it reaches 34.6% exact match on an independently generated in-distribution test set (Study 1) and 30.5% (Study 2). Its accuracy falls steadily as expressions grow, from 14.8% at 4 operations to 2.1% at 7, and reaches 7.0% on the Study 2 shift. The Transformer stays near the floor throughout: 7.8% and 2.3% on the test sets, and a mean of at most 2.7% on any out-of-distribution split. Neither architecture transfers what it learns to substantially longer expressions at this data scale. The gap survives a positional-encoding sweep and a learning-rate and weight-decay control (see Further results). All checkpoints and datasets are released on the Hugging Face Hub.

---

## Results (v2 pipeline, current)

*Values below are the canonical v2 numbers, read directly from `matrix_summary.txt`,
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

Paper: *Partial Competence and Limited Transfer: Comparing LSTM and Transformer
Generalization on Arithmetic Expressions* (arXiv ID pending).

## Further results

The sweep and the tuning control below are committed in this repository with their design docs; the manuscript draft predates them.

- **Probes** ([summary](experiments/results/analysis_v2/nonlinear_probe/summary.md)). On a stratified in-distribution split, a linear probe decodes running depth from the LSTM's cell state at 0.896–0.911 against a 0.527 majority. No probe decodes a running value in either model (R² 0.02 to 0.11, below the pre-set threshold). The Transformer's encoder loses linearly decodable position, from 0.995 at the input to 0.18 at layer 3. These results are correlational.
- **Positional-encoding sweep** ([design](docs/design/pos-encoding-sweep.md), [summary](experiments/results/analysis_v2/pos_sweep/summary.md)). RoPE raises the Transformer's in-distribution test accuracy over sinusoidal by 13.5, 10.3 and 13.6 points across seeds, and ALiBi by 2.5 to 5.2; learned absolute encoding does not move it. The LSTM still leads every arm.
- **Tuning control** ([design](docs/design/tuning-control.md), [readings](experiments/results/analysis_v2/tuning_control/readings.md)). At the validation-selected learning rate, the LSTM-minus-Transformer test gap is 25.8, 28.4 and 27.6 points, with 95% intervals above zero on all three seeds; RoPE's gain over sinusoidal holds at 11.8 to 12.6 points; weight decay 0.01 and 0.1 move nothing. Sinusoidal at 1e-4 does not fit within 200 epochs, and 1e-3 stops near the floor for both encodings, so those rates are not covered. Under the pre-registered selected-epoch fit rule these two comparisons are not evaluable; the design doc's 2026-11-01 Deviations entry explains the change.

## Retired results

Results produced before the v2 pipeline were affected by an input-truncation bug and are retired, not corrected. They are kept, with the bug description, in [docs/LEGACY_RESULTS.md](docs/LEGACY_RESULTS.md); do not cite them.

---

## Experimental Design

### Task

Models receive a fully parenthesized arithmetic expression as input and
must produce the correct integer result as output. The task is formulated
as a sequence-to-sequence problem with character-level tokenization.

```
Input:  ( ( 1 3 - 7 ) + ( 8 - 8 ) )
Output: 6
```
### Controlled Studies

Study 1 varies operation count; Study 2 shifts depth and tree shape together:

**Study 1 - Operation-Count Generalization**
Train on expressions with 2–3 operations. Test on expressions with 4–7
operations. Tests whether learned computation rules extend to longer
expression chains.

**Study 2 - Structural Generalization (exploratory)**
Train on balanced depth-2 trees with three operations. Test on chain-shaped depth-3 trees with three operations. Depth and shape change together, so this study cannot isolate depth.

### Dataset Specification

- Operand range: 1-20
- Division: D1 integer-only (no remainder, no division by zero)
- Magnitude cap: |result| ≤ 10,000
- Format: Fully parenthesized infix notation
- Operator distribution: equal 25% per operator type
- Vocabulary: 20 character-level tokens
- Split: 8,000 train / 1,000 val / 1,000 OOD per study

### Model Architectures

**LSTM Encoder-Decoder**
Bidirectional encoder, unidirectional decoder. Hidden dim 256, embedding
dim 128, ~2.1M parameters. Trained with teacher forcing, Adam (lr=0.001),
early stopping (patience=25).

**Transformer Encoder-Decoder**
3 encoder + 3 decoder layers, 8 attention heads, model dim 256, sinusoidal
positional encoding, ~5.5M parameters. Trained with teacher forcing, Adam
(lr=0.0001), early stopping (patience=25).

---

## Repository Structure

```
Compositional-Reasoning-Arithmetic/
├── train_v2.py                  # v2 training entry point (one model/study/seed per run)
├── matrix_summary.txt           # v2 seed-matrix summary (selected epoch, val accuracy)
├── CHECKPOINTS.md               # Registry of all trained checkpoints
├── main.py                      # Legacy entry point (eval path retired, see notice above)
├── scripts/                     # Run from repo root: python -m scripts.<folder>.<name>
│   ├── eval/                    # cold_verify, eval_test_splits, eval_all_per_example
│   ├── analysis/                # Reviewer stats: baselines, paired tests, divisor check, items 8-10
│   ├── datagen/                 # make_test_splits (independent test splits)
│   └── training/                # run_matrix.sh (12-run seed matrix), overfit_test
├── data/
│   ├── generate_controlled.py   # Controlled dataset generation
│   ├── tokenizer.py             # Character-level math tokenizer
│   ├── dataloader.py            # PyTorch DataLoader pipeline
│   └── tree.py                  # Binary expression tree data structure
├── models/
│   ├── lstm.py                  # LSTM encoder-decoder
│   └── transformer.py           # Transformer encoder-decoder
├── utils/
│   └── trainer.py               # Training loop + early stopping
├── experiments/
│   ├── config.py                # Checkpoint and path configuration
│   ├── analysis.py              # CP4/CP5 programmatic analysis
│   └── notebooks/
│       ├── analysis_plots.ipynb # CP1–CP5 attention visualizations (legacy)
│       └── analysis_v2.ipynb    # Mechanistic analysis v2
├── datasets/                    # Legacy splits (pre-v2)
├── datasets_v2/                 # Current splits: study1 (train/val/test/ood_ops4-7), study2 (train/val/test/ood)
├── results/                     # Legacy results (pre-v2)
├── results_v2/                  # Per-run history/OOD results + analysis outputs (*.json)
│   └── {lstm,transformer}/study{1,2}/seed{0,1,2}/
├── figures/
├── requirements.txt
└── README.md
```

---

## Reproducing Results

### Pretrained checkpoints and datasets (Hugging Face Hub)

The twelve v2 checkpoints and the `datasets_v2` splits are published on the
Hugging Face Hub, so results can be checked without retraining:

- **Models:** [Ajay0728/Compositional-reasoning-arithmetic](https://huggingface.co/Ajay0728/Compositional-reasoning-arithmetic)
  - `{lstm,transformer}/study{1,2}/seed{0,1,2}/best_model.pt`, the same files listed in [`CHECKPOINTS.md`](CHECKPOINTS.md)
- **Datasets:** [Ajay0728/Compositional-reasoning-arithmetic-datasets](https://huggingface.co/datasets/Ajay0728/Compositional-reasoning-arithmetic-datasets)
  - the same files as `datasets_v2/`, including `CHECKSUMS.txt`

To place the checkpoints where the evaluation scripts expect them
(`results_v2/...`):
```bash
pip install huggingface_hub
hf download Ajay0728/Compositional-reasoning-arithmetic --include "*/best_model.pt" --local-dir results_v2
```

### Setup
```bash
git clone https://github.com/bcode0127-debug/Compositional-Reasoning-Arithmetic.git
cd Compositional-Reasoning-Arithmetic
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 2.0+

### Generate datasets
```bash
python main.py --mode generate
```

Generates 20,000 controlled samples across both studies.

### Train
```bash
# LSTM
python main.py --mode train --model lstm --num-epochs 100

# Transformer
python main.py --mode train --model transformer --num-epochs 100
```

### Evaluate

*Retired - see the input-truncation notice above. `main.py --mode eval`
silently truncates input to 20 characters and its output should not be
treated as a valid measurement. Kept runnable as historical record only.*
```bash
python main.py --mode eval --model lstm
python main.py --mode eval --model transformer
```

For a valid evaluation, use `train_v2.py` / `scripts/eval/cold_verify.py`
against `datasets_v2/` instead. Run scripts from the repo root as modules so
their root-relative paths resolve:
```bash
python -m scripts.eval.eval_test_splits --device cpu
python -m scripts.analysis.per_seed_and_paired_stats
```

### Attention analysis

*Retired - see the input-truncation notice above.* Open
`experiments/notebooks/analysis_plots.ipynb` and run all cells to
reproduce the historical (truncation-affected) CP1–CP5 results, saved to
`experiments/results/`. For the corrected analysis, use
`experiments/notebooks/analysis_v2.ipynb`.

---

## Related Work

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

## Citation
```bibtex
@misc{jamanjyothi2026partial,
  title   = {Partial Competence and Limited Transfer: Comparing LSTM and
             Transformer Generalization on Arithmetic Expressions},
  author  = {Jamanjyothi, Ajay and Kasztelnik, Karina},
  year    = {2026},
  note    = {arXiv preprint (cs.LG)}
}
```
