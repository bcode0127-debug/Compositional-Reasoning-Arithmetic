# Compositional Generalization in Mathematical Reasoning

Investigating whether neural networks learn algorithmic reasoning or memorize patterns in arithmetic expression evaluation.

**Status:** In Progress | **Current Phase:** LSTM + Transformer Baselines Complete

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

This project systematically evaluates compositional generalization in neural sequence models through controlled arithmetic reasoning tasks. tested whether models learn underlying computational algorithms or surface-level pattern matching.

---

## Research Objectives

This work investigates compositional generalization in sequence-to-sequence models through two controlled studies:

1. **Length Generalization:** Evaluate whether models trained on expressions with 2-3 operations can generalize to expressions with 4-7 operations.

2. **Depth Generalization:** Evaluate whether models trained on expression trees of depth 2 can generalize to depth 3.

Compared LSTM and Transformer architectures to identify which architectural properties support compositional reasoning in arithmetic evaluation tasks.

---

## Results

### LSTM Baseline 

| Study | Train Acc | Val Acc | OOD Acc | Gen Gap |
|-------|-----------|---------|---------|---------|
| **Study 1** (Length: 2-3 → 4-7 ops) | 95.1% | 38.2% | **1.8%** | 93.3% |
| **Study 2** (Depth: d=2 → d=3) | 89.2% | 15.9% | **10.4%** | 78.8% |

**Key Finding:** LSTM achieves high accuracy on training distribution but catastrophic failure on OOD examples, confirming surface-level pattern memorization over algorithmic reasoning.

---

### Transformer  

| Study | Train Acc | Val Acc | OOD Acc | Gen Gap |
|-------|-----------|---------|---------|---------|
| **Study 1** (Length: 2-3 → 4-7 ops) | 56.7% | 12.1% | **0.4%** | 56.3% |
| **Study 2** (Depth: d=2 → d=3) | 51.3% | 5.7% | **2.3%** | 49.0% |

**Key Finding:** Transformer underperforms LSTM on both train and OOD accuracy. Additionally shows autoregressive generation weakness - val accuracy (6-12%) is significantly lower than teacher-forced train accuracy (51-57%), indicating compounding error accumulation during inference.

---

### Overall Comparison

| Model | Study | Train Acc | Val Acc | OOD Acc | Gen Gap |
|-------|-------|-----------|---------|---------|---------|
| LSTM | Study 1 (Length) | 95.1% | 38.2% | 1.8% | 93.3% |
| LSTM | Study 2 (Depth) | 89.2% | 15.9% | 10.4% | 78.8% |
| Transformer | Study 1 (Length) | 56.7% | 12.1% | 0.4% | 56.3% |
| Transformer | Study 2 (Depth) | 51.3% | 5.7% | 2.3% | 49.0% |

**Both architectures fail compositional generalization. Neither learns algorithmic reasoning.**

---

## Project Structure
```
Compositional-Reasoning-Arithmetic/
├── main.py                     # Entry point
├── models/
│   ├── lstm.py                 # LSTM encoder-decoder
│   └── transformer.py          # Transformer encoder-decoder
├── utils/
│   └── trainer.py              # Training logic
├── data/
│   ├── generate_controlled.py  # Dataset generation
│   ├── tokenizer.py            # Expression tokenization
│   └── dataloader.py           # Data loading
├── datasets/
│   ├── study1/                 # Length generalization (10K samples)
│   └── study2/                 # Depth generalization (10K samples)
├── results/
│   ├── lstm_baseline/          # LSTM training histories + checkpoints
│   └── transformer/            # Transformer training histories + checkpoints
├── figures/                    # Publication-quality plots
├── notebooks/                  # Analysis and visualization
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements
- Python 3.8+
- PyTorch 2.0+ with CUDA

### Installation
```bash
git clone https://github.com/bcode0127-debug/Compositional-Reasoning-Arithmetic.git
cd Compositional-Reasoning-Arithmetic
pip install -r requirements.txt
```

---

## Usage

### 1. Generate Datasets
```bash
python main.py --mode generate
```
Creates 20,000 controlled samples across two generalization studies.

### 2. Train Model
```bash
# LSTM (auto lr=0.001)
python main.py --mode train --model lstm --num-epochs 100

# Transformer (auto lr=0.0001)
python main.py --mode train --model transformer --num-epochs 100
```

### 3. Evaluate
```bash
python main.py --mode eval --model lstm
python main.py --mode eval --model transformer
```

---

## Experimental Design

### Dataset Constraints
All expressions maintain:
- **Operand range:** 1-20
- **Result magnitude:** ≤1000
- **Operations:** +, -, *, / (balanced)
- **Format:** Fully parenthesized infix notation

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

### Model Architectures

**LSTM Encoder-Decoder**
- Embedding: 128-dim
- Hidden: 256-dim
- Parameters: 2,120,724
- Optimizer: Adam (lr=0.001)
- Training: Teacher forcing + early stopping (patience=25)

**Transformer Encoder-Decoder**
- Model dim: 256
- Attention heads: 8
- Encoder/Decoder layers: 3
- Parameters: 5,546,004
- Optimizer: Adam (lr=0.0001)
- Training: Teacher forcing + early stopping (patience=25)

---

**Last Updated:** February 17, 2026
