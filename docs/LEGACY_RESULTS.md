# Retired pre-v2 results

## ⚠️ Pre-v2 Results Invalidated (Input-Truncation Bug)

Every number below through the "Attention Analysis" table, the 1.6%/0.3%
OOD headline and CP1–CP5, was produced by `main.py`'s legacy eval path
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
- Seed-matrix training results: [`matrix_summary.txt`](../matrix_summary.txt),
  [`CHECKPOINTS.md`](../CHECKPOINTS.md)
- Mechanistic v2 (behavioral + attention + probing, on verified checkpoints,
  truncation-safe throughout): [`experiments/notebooks/analysis_v2.ipynb`](../experiments/notebooks/analysis_v2.ipynb),
  [`experiments/results/analysis_v2/`](../experiments/results/analysis_v2/)

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
