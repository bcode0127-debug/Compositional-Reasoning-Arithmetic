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

## Key Findings

- Under autoregressive decoding, OOD accuracy collapses to 1.6% (LSTM) and
  0.3% (Transformer) on length generalization; multiplication and division
  reach 0.0% OOD
- Length generalization shows a sharp in-distribution → OOD drop (LSTM
  32.6% → 1.6%); depth accuracy is low both in- and out-of-distribution,
  indicating the models never learned depth composition in the first place
- Transformer Layer 3 encoder attention collapses to near-uniform
  distributions on OOD inputs while Layer 1 retains structure
- No attention head crosses the 0.01 compositional specialization threshold -
  failure is architecturally distributed, not localized
- 4 of 5 traced OOD failures occur at decoding Step 0, ruling out error
  accumulation as the cause

---

## Results

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
| CP3 | Head ablation - ID drop vs OOD drop delta (48 heads) | No head crosses 0.01 threshold; failure is distributed across architecture |
| CP4 | OOD accuracy by dominant operation type | `+`: 0.2%, `-`: 2.4%, `*`: 0.0%, `/`: 0.0% |
| CP5 | Failure onset traces - first wrong decoding step | 4/5 failures at Step 0; immediate breakdown, not error accumulation |

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
```bash
python main.py --mode eval --model lstm
python main.py --mode eval --model transformer
```

### Attention analysis

Open `experiments/notebooks/analysis_plots.ipynb` and run all cells.
Saves all CP1–CP5 results to `experiments/results/`.

---

## Related Work

- Dziri et al. (2023) - Transformers solve compositional tasks via
  linearized pattern matching, not systematic reasoning
- Stolfo et al. (2023) - Causal mediation analysis of arithmetic
  reasoning in language models
- Elhage et al. (2021) - Mathematical framework for Transformer circuits
- Zhang et al. (2025) - Complexity control and OOD generalization in
  Transformers
- Hahn et al. (2026) - Shattered compositionality; arithmetic subskills
  acquired independently and subject to interference (arXiv:2601.22510)

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
