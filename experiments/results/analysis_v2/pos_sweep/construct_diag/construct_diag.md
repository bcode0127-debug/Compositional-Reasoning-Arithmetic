This is a construct diagnostic for the positional-encoding sweep (hypotheses H1 to H3). It sits beside the pre-registered verdicts: summary.md, decision_rules.json and every battery output are unchanged by it.

The hypotheses came from a Mac smoke run of the battery code (scikit-learn 1.6.1, scratch output directory, anchor check skipped), whose layer-1 injection result for RoPE and ALiBi was reported before the pod battery ran; the pod battery then gave the same result. The tests and readings below were fixed before this script was written.

Run: 2026-10-29T11:59:41.255935+00:00 on NVIDIA RTX 4000 Ada Generation pod image `RunPod PyTorch template 1.5.0-2.8.0 (PyTorch 2.8.0, CUDA 12.8.1, Ubuntu 24.04)`, torch 2.8.0+cu128, scikit-learn 1.9.1, probes on cpu, code commit `bf65116e4b6fcd282dad43826090e3e962aa70ec`.

## H1: ALiBi, reversal equivariance (20 stratified test expressions)

| Checkpoint | encoder_input | layer1 | layer2 | layer3 | memory | Max (layers 1-3, memory) | Pass (<= 1e-5) |
|---|---|---|---|---|---|---|---|
| alibi seed0 | 0.00e+00 | 9.54e-07 | 1.43e-06 | 2.03e-06 | 1.82e-06 | 2.03e-06 | True |
| alibi seed1 | 0.00e+00 | 9.54e-07 | 1.19e-06 | 1.91e-06 | 1.67e-06 | 1.91e-06 | True |
| alibi seed2 | 0.00e+00 | 1.19e-06 | 9.54e-07 | 1.67e-06 | 1.61e-06 | 1.67e-06 | True |
| control nope_seed0 (expected pass) | 0.00e+00 | 3.58e-07 | 9.54e-07 | 1.51e-06 | 1.46e-06 | 1.51e-06 | True |
| control sinusoidal_seed0 (expected fail) | 2.00e+00 | 2.55e+00 | 3.92e+00 | 3.62e+00 | 3.58e+00 | 3.92e+00 | False |

## H3: RoPE, reversal and random permutation

Expected (amended reading): encoder_input passes, since RoPE adds no position at the input; layers 1-3 and memory fail. "Fails by orders of magnitude" means the largest of layers 1-3 and memory is at least 100 times the tolerance.

| Checkpoint | Test | encoder_input | layer1 | layer2 | layer3 | memory | Fails by orders of magnitude |
|---|---|---|---|---|---|---|---|
| rope seed0 | reversal | 0.00e+00 | 5.75e-01 | 4.42e+00 | 5.42e+00 | 5.08e+00 | True |
| rope seed0 | permutation | 0.00e+00 | 5.83e-01 | 4.67e+00 | 5.51e+00 | 5.05e+00 | True |
| rope seed1 | reversal | 0.00e+00 | 7.09e-01 | 4.93e+00 | 5.03e+00 | 4.91e+00 | True |
| rope seed1 | permutation | 0.00e+00 | 8.34e-01 | 4.91e+00 | 5.39e+00 | 5.14e+00 | True |
| rope seed2 | reversal | 0.00e+00 | 6.13e-01 | 4.88e+00 | 5.99e+00 | 5.37e+00 | True |
| rope seed2 | permutation | 0.00e+00 | 6.34e-01 | 4.86e+00 | 6.42e+00 | 5.75e+00 | True |

## H2: relative-offset probe (pairs: {'train': 12000, 'stratified_test': 6000})

Split: pairs never cross expressions: training pairs come from the 300 training expressions of the stratified split, test pairs from its 150 test expressions; the bootstrap resamples test expressions.

Reference (amended, applied before the readings were computed): the highest point accuracy among NoPE at the same layer and the order-blind pair baselines B1p (one-hot char_i + one-hot char_j), B2p (B1p + one-hot expression length) and B3p (B2p + character counts), chosen mechanically. The probe run itself used NoPE and B2p only; its checkpoint statistics are reused unchanged and the readings were recomputed. Before the amendment was received, H2 point accuracies for sinusoidal seeds 0-2 and learned seed 0 had been printed in the pod run log, and a Mac test run (scikit-learn 1.6.1) had printed ALiBi seed 0's table.

A reading is "carries relative position" if the lower bound of (arm - chosen reference), paired by expression, exceeds +0.1 on all three seeds.

| Arm | Layer | Target | Accuracy by seed | NoPE | B1p | B2p | B3p | Chosen | Lower bound by seed | Carries |
|---|---|---|---|---|---|---|---|---|---|---|
| sinusoidal | layer1 | magnitude | 0.155, 0.145, 0.145 | 0.139 | 0.158 | 0.148 | 0.147 | B1p | -0.014, -0.022, -0.024 | False |
| sinusoidal | layer1 | sign | 0.815, 0.847, 0.844 | 0.615 | 0.608 | 0.607 | 0.618 | B3p | +0.182, +0.214, +0.209 | True |
| sinusoidal | layer2 | magnitude | 0.134, 0.146, 0.139 | 0.135 | 0.158 | 0.148 | 0.147 | B1p | -0.036, -0.024, -0.029 | False |
| sinusoidal | layer2 | sign | 0.636, 0.644, 0.647 | 0.617 | 0.608 | 0.607 | 0.618 | B3p | +0.005, +0.010, +0.014 | False |
| sinusoidal | layer3 | magnitude | 0.138, 0.138, 0.131 | 0.135 | 0.158 | 0.148 | 0.147 | B1p | -0.030, -0.032, -0.037 | False |
| sinusoidal | layer3 | sign | 0.624, 0.612, 0.616 | 0.608 | 0.608 | 0.607 | 0.618 | B3p | -0.008, -0.021, -0.014 | False |
| learned | layer1 | magnitude | 0.159, 0.148, 0.150 | 0.139 | 0.158 | 0.148 | 0.147 | B1p | -0.011, -0.022, -0.019 | False |
| learned | layer1 | sign | 0.778, 0.779, 0.771 | 0.615 | 0.608 | 0.607 | 0.618 | B3p | +0.144, +0.147, +0.139 | True |
| learned | layer2 | magnitude | 0.137, 0.131, 0.141 | 0.135 | 0.158 | 0.148 | 0.147 | B1p | -0.032, -0.039, -0.027 | False |
| learned | layer2 | sign | 0.631, 0.638, 0.628 | 0.617 | 0.608 | 0.607 | 0.618 | B3p | -0.000, +0.008, -0.003 | False |
| learned | layer3 | magnitude | 0.141, 0.134, 0.139 | 0.135 | 0.158 | 0.148 | 0.147 | B1p | -0.028, -0.034, -0.030 | False |
| learned | layer3 | sign | 0.611, 0.607, 0.609 | 0.608 | 0.608 | 0.607 | 0.618 | B3p | -0.020, -0.025, -0.021 | False |
| rope | layer1 | magnitude | 0.152, 0.156, 0.146 | 0.139 | 0.158 | 0.148 | 0.147 | B1p | -0.015, -0.012, -0.024 | False |
| rope | layer1 | sign | 0.652, 0.656, 0.677 | 0.615 | 0.608 | 0.607 | 0.618 | B3p | +0.021, +0.025, +0.043 | False |
| rope | layer2 | magnitude | 0.151, 0.148, 0.144 | 0.135 | 0.158 | 0.148 | 0.147 | B1p | -0.018, -0.021, -0.025 | False |
| rope | layer2 | sign | 0.774, 0.768, 0.768 | 0.617 | 0.608 | 0.607 | 0.618 | B3p | +0.139, +0.133, +0.134 | True |
| rope | layer3 | magnitude | 0.147, 0.148, 0.150 | 0.135 | 0.158 | 0.148 | 0.147 | B1p | -0.022, -0.020, -0.018 | False |
| rope | layer3 | sign | 0.748, 0.804, 0.817 | 0.608 | 0.608 | 0.607 | 0.618 | B3p | +0.113, +0.170, +0.181 | True |
| alibi | layer1 | magnitude | 0.142, 0.153, 0.156 | 0.139 | 0.158 | 0.148 | 0.147 | B1p | -0.028, -0.014, -0.013 | False |
| alibi | layer1 | sign | 0.611, 0.618, 0.608 | 0.615 | 0.608 | 0.607 | 0.618 | B3p | -0.020, -0.010, -0.021 | False |
| alibi | layer2 | magnitude | 0.148, 0.147, 0.152 | 0.135 | 0.158 | 0.148 | 0.147 | B1p | -0.021, -0.021, -0.017 | False |
| alibi | layer2 | sign | 0.934, 0.929, 0.925 | 0.617 | 0.608 | 0.607 | 0.618 | B3p | +0.302, +0.297, +0.293 | True |
| alibi | layer3 | magnitude | 0.149, 0.141, 0.153 | 0.135 | 0.158 | 0.148 | 0.147 | B1p | -0.020, -0.027, -0.018 | False |
| alibi | layer3 | sign | 0.923, 0.898, 0.908 | 0.608 | 0.608 | 0.607 | 0.618 | B3p | +0.291, +0.265, +0.275 | True |

Majority rate in the test pairs: magnitude 0.162, sign 0.504.

RoPE or ALiBi clears the reference at: rope layer2 sign, rope layer3 sign, alibi layer2 sign, alibi layer3 sign.

## Pre-stated predictions

A prediction holds only if it holds at every layer 1 to 3 (an aggregation fixed in the script before the run); the per-layer readings are in the H2 table. If neither RoPE nor ALiBi clears the reference, that is reported above as a finding (no linearly decodable relative offset), not as a failure.

| Prediction | Held |
|---|---|
| H1: ALiBi is reversal-equivariant (pass on all seeds) | held |
| H3: RoPE encoder_input passes; layers 1-3 and memory fail, reversal and permutation (all seeds) | held |
| H2: RoPE carries magnitude and sign (every layer 1-3) | not held |
| H2: ALiBi carries magnitude and its sign stays near the references (every layer 1-3) | not held |
