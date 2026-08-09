# Do Neural Sequence Models Learn Compositional Reasoning?
### Mechanistic Evidence from Arithmetic Generalization

---

## Abstract

Neural sequence models trained on arithmetic expressions fit the training
distribution well under teacher forcing, yet their autoregressive generation
degrades sharply on held-out in-distribution inputs and collapses on
out-of-distribution (OOD) expressions. This work provides empirical and
mechanistic evidence that neither LSTM nor Transformer encoder-decoder
architectures learn compositional reasoning - both rely on surface-level
pattern memorization. All accuracies are reported under autoregressive
decoding. Attention analysis of the Transformer shows the OOD collapse is
not error accumulation during decoding, but an immediate breakdown in
encoder representations at the first decoding step.

---

## ⚠️ Pre-v2 Results Invalidated (Input-Truncation Bug)

Every number below through the "Attention Analysis" table — the 1.6%/0.3%
OOD headline and CP1–CP5 — was produced by `main.py`'s legacy eval path
(`main.py::evaluate_model()`), which silently truncates encoder input to
`max_input_len=20` characters (`src_ids[:max_input_len]`, no error, no
warning). Measured directly against the datasets these numbers were
computed on: 99.0% of `datasets/study1/ood.json` (max length 50) and 70.0%
of `datasets/study2/ood.json` exceed 20 characters. That means the model
was evaluated on a chopped prefix of nearly every OOD example - in most
cases missing the operators and operands that determine the correct
answer. The "catastrophic OOD collapse" these numbers report is
substantially confounded by this truncation artifact, not a clean
measurement of compositional failure. CP1–CP5's attention maps,
cross-attention, head ablation, and failure traces all inherit the same
truncated inputs, so their interpretation is compromised the same way.

**These results are retired, not corrected.** They are left in place below
as a historical record of what this repo previously reported, not as
findings to cite. The legacy code path (`main.py`, `experiments/analysis.py`,
`experiments/notebooks/analysis_plots.ipynb`) and the legacy `datasets/`
directory are unchanged and still runnable, but no valid claim about
compositional reasoning should be drawn from their output.

The corrected, superseding results are the v2 pipeline (`train_v2.py` +
`datasets_v2/`, `max_input_len=64` with hard-fail encode_checked instead of
silent truncation) and mechanistic analysis v2:
- Seed-matrix training results: [`matrix_summary.txt`](matrix_summary.txt),
  [`CHECKPOINTS.md`](CHECKPOINTS.md)
- Mechanistic v2 (behavioral + attention + probing, on verified checkpoints,
  truncation-safe throughout): [`experiments/notebooks/analysis_v2.ipynb`](experiments/notebooks/analysis_v2.ipynb),
  [`experiments/results/analysis_v2/`](experiments/results/analysis_v2/)

---

## Results (v2 pipeline, current)

*Values below are the canonical v2 numbers, read directly from `matrix_summary.txt`,
each run's `ood_results.json`, and `results_v2/test_set_results.json`. Val and OOD are
mean [min-max] across seeds 0/1/2; Test is mean ± SD. All figures are autoregressive
exact-match (verified teacher-forced-equivalent under greedy decoding by `cold_verify.py`).*

**Validation selected the checkpoints; the test split is independently held out and was
generated after training, so the Test column is the primary in-distribution result.**

**Study 1 (Length Generalization)**

| Model | Val | Test | OOD ops4 | OOD ops5 | OOD ops6 | OOD ops7 |
|---|---|---|---|---|---|---|
| LSTM | 35.7% [35.1-36.1] | 34.57% ± 0.95 | 14.8% [13.7-16.4] | 8.2% [7.5-9.5] | 2.7% [2.5-2.8] | 2.1% [1.8-2.5] |
| Transformer | 7.3% [6.8-7.5] | 7.83% ± 0.12 | 0.7% [0.4-1.2] | 0.6% [0.5-0.7] | 0.3% [0.1-0.5] | 0.4% [0.2-0.6] |

**Study 2 (Depth Generalization)**

| Model | Val | Test | OOD |
|---|---|---|---|
| LSTM | 31.5% [30.1-32.5] | 30.47% ± 2.14 | 7.0% [6.3-7.8] |
| Transformer | 3.9% [3.7-4.2] | 2.27% ± 0.47 | 2.7% [2.0-3.1] |

Paper: *Partial Competence and Memorization: Comparing LSTM and Transformer
Generalization on Arithmetic Expressions* (arXiv ID pending).
---

## Key Findings

*The findings below predate the v2 pipeline and are affected by the
input-truncation bug described above. Retained as historical record only -
see the notice above before citing any number in this section or the two
tables that follow.*

- Under autoregressive decoding, OOD accuracy collapses to 1.6% (LSTM) and
  0.3% (Transformer) on length generalization; multiplication and division
  reach 0.0% OOD
- Length generalization shows a sharp in-distribution → OOD drop (LSTM
  32.6% → 1.6%); depth accuracy is low both in- and out-of-distribution,
  indicating the models never learned depth composition in the first place
- Transformer Layer 3 encoder attention collapses to near-uniform
  distributions on OOD inputs while Layer 1 retains structure
- No attention head shows meaningful OOD-specific sensitivity when ablated (top head's OOD-accuracy drop ≈ 0.000); ablation deltas are driven by in-distribution drops, not OOD-specific compositional work - failure is architecturally distributed, not localized in a single circuit
- All four operations collapse to near-total OOD failure (0.00%-0.58%), with no meaningful ranking between them
- 4 of 5 traced OOD failures occur at decoding Step 0, ruling out 
  error accumulation as the cause

---

## Results

*Historical / retired - see the input-truncation notice above.*

### Baseline Accuracy

*Val and OOD accuracies are reported under autoregressive decoding at inference
time. Train accuracy is teacher-forced (training-time) and is shown only for
reference; the Gap column is computed as Val − OOD so all compared numbers use
the same decoding mode.*

| Model | Study | Train (TF) | Val | OOD | Gap (Val−OOD) |
|---|---|---|---|---|---|
| LSTM | Study 1 - Length (2-3 → 4-7 ops) | 95.1% | 32.6% | **1.6%** | 31.0 pp |
| LSTM | Study 2 - Depth (d=2 → d=3) | 89.2% | 11.5% | **11.9%** | −0.4 pp |
| Transformer | Study 1 - Length | 56.7% | 7.0% | **0.3%** | 6.7 pp |
| Transformer | Study 2 - Depth | 51.3% | 3.9% | **4.0%** | −0.1 pp |

### Attention Analysis - Transformer (Study 1)

| Checkpoint | What was measured | Finding |
|---|---|---|
| CP1 | Encoder self-attention heatmaps (ID vs OOD) | Layer 1 heads structured on ID; Layer 3 collapses to uniform on OOD |
| CP2 | Cross-attention per decoding step (ID vs OOD) | ID attention shifts selectively per step; OOD spreads flat across all tokens |
| CP3 | Head ablation - ID drop vs OOD drop delta (48 heads) | Top head (`EL3H2`) OOD drop ≈ 0.000; ablation deltas are driven by ID drops, not OOD-specific effects - no head performs OOD-specific compositional work, failure is distributed across the architecture |
| CP4 | OOD accuracy by dominant operation type | `+`: 1/441 = 0.23%, `-`: 2/347 = 0.58%, `*`: 1/201 = 0.50%, `/`: 0/11 = 0.00%* |
| CP5 | Failure onset traces - first wrong decoding step | 4/5 failures at Step 0; immediate breakdown, not error accumulation |

*CP4's `/` bucket has only 11 OOD examples; its exact 0.00% is not statistically reliable in isolation. CP5 traces 5 failures for illustration, not to establish a population failure rate.*

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

Two generalization axes are tested independently:

**Study 1 - Length Generalization**
Train on expressions with 2–3 operations. Test on expressions with 4–7
operations. Tests whether learned computation rules extend to longer
expression chains.

**Study 2 - Depth Generalization**
Train on expression trees of depth 2. Test on depth 3. Tests whether
learned subexpression evaluation extends to deeper nesting.

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
├── main.py                      # Entry point — generate / train / eval
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
│       └── analysis_plots.ipynb # CP1–CP5 attention visualizations
├── datasets/
│   ├── study1/                  # Length generalization splits
│   └── study2/                  # Depth generalization splits
├── results/
│   ├── lstm_baseline/           # LSTM checkpoints + training histories
│   └── transformer/             # Transformer checkpoints + training histories
├── requirements.txt
└── README.md
```

---

## Reproducing Results

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

For a valid evaluation, use `train_v2.py` / `cold_verify.py` against
`datasets_v2/` instead.

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
@misc{jamanjyothi2026compositional,
  title   = {Do Neural Sequence Models Learn Compositional Reasoning?
             Mechanistic Evidence from Arithmetic Generalization},
  author  = {Jamanjyothi, Ajay},
  year    = {2026},
  note    = {arXiv preprint (cs.LG)}
}
```
