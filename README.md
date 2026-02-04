# Compositional Generalization in Mathematical Reasoning

Investigating whether neural networks learn algorithmic reasoning or memorize patterns in arithmetic expression evaluation.

**Status:** In Progress | **Current Phase:** LSTM Baseline Complete

---

## Table of Contents
- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Experimental Design](#experimental-design)

---

## Overview

This project systematically evaluates compositional generalization in neural sequence models through controlled arithmetic reasoning tasks. We test whether models learn underlying computational algorithms or surface-level pattern matching.

---

## Research Objectives

This work investigates compositional generalization in sequence-to-sequence models through two controlled studies:

1. **Length Generalization:** Evaluate whether models trained on expressions with 2-3 operations can generalize to expressions with 4-7 operations.

2. **Depth Generalization:** Evaluate whether models trained on expression trees of depth 2 can generalize to depth 3.

Compared LSTM and Transformer architectures to identify which architectural properties support compositional reasoning in arithmetic evaluation tasks.

---

## Results

### LSTM Baseline COMPLETE

| Study | Train Acc | Val (Held-Out) | OOD Acc | Generalization Gap |
|-------|-----------|----------------|---------|-------------------|
| **Study 1** (Length: 2-3 → 4-7 ops) | 99.41% | 39.70% | **0.80%** | -98.61% |
| **Study 2** (Depth: d=2 → d=3) | 99.45% | 14.40% | **13.80%** | -85.65% |

**Key Finding:** LSTM achieves near-perfect accuracy on training distribution but catastrophic failure on OOD examples.

### Transformer Baseline IN PROGRESS

*Results pending*

---

## Project Structure
```
Compositional-Reasoning-Arithmetic/
├── main.py                     # Entry point
├── models/
│   ├── lstm.py                 # LSTM encoder-decoder
│   └── transformer.py          # [In Progress]
├── utils/
│   ├── tokenizer.py            # Expression tokenization
│   └── trainer.py              # Training logic
├── data/
│   └── dataset.py              # Data loading
├── datasets/
│   ├── study1/                 # Length generalization (10K samples)
│   ├── study2/                 # Depth generalization (10K samples)
│   └── verification/           # 40 controlled validation samples
├── results/
│   ├── lstm_baseline/          # Training histories + evaluation
│   └── transformer_baseline/   # [Pending]
├── figures/                    # Publication-quality plots
├── notebooks/
│   └── lstm_training.ipynb     # Reproducible training workflow
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements
- Python 3.8+
- PyTorch 2.0+ with CUDA
- 16GB+ GPU RAM (A100)

### Installation
```bash
git clone https://github.com/bcode0127-debug/Compositional-Reasoning-Arithmetic.git
cd Math-reasoning-research
pip install -r requirements.txt
```

---

## Usage

### 1. Generate Datasets
```bash
python main.py --mode generate
```
Creates 20,000 controlled samples across two generalization studies.

### 2. Run Sanity Check
```bash
python main.py --mode sanity
```
Verifies model can memorize 30 samples (expected: 100% accuracy).

### 3. Train Model
```bash
# LSTM (complete)
python main.py --mode train --model lstm --num-epochs 100 --batch-size 32 --lr 0.001

# Transformer (in progress)
python main.py --mode train --model transformer --num-epochs 100 --batch-size 32 --lr 0.001
```

### 4. Evaluate
```bash
python main.py --mode eval --model lstm
```

---

## Experimental Design

### Dataset Constraints
All expressions maintain:
- **Operand range:** 1-20
- **Result magnitude:** ≤1000
- **Operations:** +, -, *, / (balanced)
- **Format:** Fully parenthesized prefix notation
- **Verification:** 40 hand-crafted samples for validation

### Study 1: Length Generalization

**Train:** 2-3 operations → **Test:** 4-7 operations

| Split | Operations | Samples | Example |
|-------|-----------|---------|---------|
| Train | 2-3 | 8,000 | `((5 + 3) * 2)` → `16` |
| Val | 2-3 | 1,000 | `((7 - 2) + 4)` → `9` |
| OOD | 4-7 | 1,000 | `(((8 + 13) - 17) + ((5 - 19) * 15))` → `-206` |

### Study 2: Depth Generalization

**Train:** depth=2 → **Test:** depth=3

| Split | Depth | Samples | Example |
|-------|-------|---------|---------|
| Train | 2 | 8,000 | `((15 - 6) + (1 + 17))` → `27` |
| Val | 2 | 1,000 | `((10 + 5) * (3 - 1))` → `30` |
| OOD | 3 | 1,000 | `((18 - 8) * (20 + 19))` → `390` |

### Model Architecture

**LSTM Encoder-Decoder** 
- Embedding: 128-dim
- Hidden: 256-dim  
- Vocabulary: 21 tokens
- Optimizer: Adam (lr=0.001)
- Training: Teacher forcing + early stopping (patience=5)

**Transformer** 
- [Details pending]

---

**Last Updated:** February 4, 2026
