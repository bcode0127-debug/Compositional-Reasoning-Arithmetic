# Nonlinear MLP probe: 12 checkpoints, matched to the Phase 3 linear protocol

## Pre-registered verdict rule

- **Unit of inference is the expression, not the position.** Positions inside one expression are correlated, so n is 150 examples per held-out split (2,945 positions on the Study 1 ID split). Every test resamples whole expressions with all their positions.
- **Significance.** For each probe seed, a 1,000-resample example-level bootstrap of the paired held-out difference (MLP minus control). Pass iff the 95% CI lower bound exceeds 0 for all three seeds.
- **Effect size, depth.** MLP mean minus std exceeds control + 0.05. At n = 150 and p near 0.73 the example-level standard error is about 0.037, so 0.05 alone is roughly 1.4 standard errors and carries no significance; the bootstrap carries that. 0.05 is kept as the effect floor because it is triple the best linear margin over majority (+0.017, LSTM h, ID). Sanity bar: MLP mean minus std also exceeds majority.
- **Effect size, value.** MLP R² exceeds control + 0.10 and exceeds 0.10 absolute, with the same bootstrap on the paired R² difference. 0.10 is five times the best linear value (0.018).
- **Control check, not an assumption.** The shuffled-label R² is verified on every split. Expected at or below 0.02; any control R² above 0.05 is reported below before any verdict.
- **Verdict per target and split.** "yes" iff the bootstrap and both effect bars pass; else "no". Where the MLP beats linear by more than 0.05 but fails any bar: "signal, below threshold". The bootstrap CI (per probe seed) is reported beside every verdict.

## Caveats

Probes are correlational and sit beside the ablation and attention evidence. An MLP null at these widths narrows the paper's probe-scoping disjunction to "distributed, beyond an MLP of these widths, or not represented"; it does not prove absence.

Study 2 rows are **exploratory**. transformer/study2/seed0 and seed2 are **unfit** (train acc 5.85% / 14.51%): run, not interpreted.
Transformer activations are extracted **with** the source key-padding mask (as the model computes them); see the bridge table for the notebook's unmasked extraction.


## STEP 4b header notes

**Label-space cap.** Every depth probe, linear and MLP, is trained on val[:300], whose depths are {1, 2} in both studies, so its output space is those classes and any position whose true depth lies outside them cannot be predicted. Per split, the ceiling is 1 minus the share of such positions (table below), compared with the bar majority + 0.05. Where the ceiling is below the bar (Study 1 ops6, ops7) a depth verdict of "no" reads "structurally capped": the test could not have passed. Study 1 ops5 reads "no (ceiling within x of bar)". The same cap applies to the Phase 3 linear depth rows (LogisticRegression trained on the same classes).

**Why an input baseline (item 1).** Study 2 is near-templated: 9 distinct depth sequences across its 300 training expressions and 9 across its 150 ID expressions. The two unfit Study 2 Transformers (train acc 5.85% / 14.51%) still pass the rule on 4 of 8 rows. Both suggest depth can be read off surface position and length without any learned representation, so every depth row is compared with an MLP on one-hot(char) + one-hot(position) + one-hot(length) features, with no prefix and no model activations.

**Pre-registered reading for the input baseline.** If the activation MLP does not beat the input-baseline MLP with bootstrap lower bound above 0 on all three probe seeds, the Study 1 LSTM depth ID result is positional and is not a representation finding (verdict "yes, not above input baseline"). From STEP 4c, Study 1 "above input baseline" requires this for both input baselines, one-hot and ordinal.

**What the matched "ID" split is.** In the matched protocol the probe trains on 300 two-operation expressions (lengths 13 to 16) and is tested on a slice that is 117 three-operation and 33 two-operation (lengths 13 to 23), so the "ID" test is a 2-to-3-operation transfer for the probe and the early-stop set never saw a 3-operation example; this applies to the Phase 3 linear rows, which used the same split. For the same reason the one-hot input baseline's 0.743 on that split is a floor: its length bits for lengths 19 to 23 were never active in training. The stratified split (STEP 4c) answers the in-distribution question.

**Control.** Study 1 depth controls stop at epoch 1 to 5 with control_train 0.57 to 0.64 against a 0.554 training majority, so partial memorization only.

## STEP 4d header notes

**Stratified split, linear.** On the stratified split the linear probe alone decodes depth from the LSTM cell state at 0.896–0.911 against majority 0.527 and one-hot linear 0.697 (paired 95% CIs, linear(c) − linear(one-hot): [+0.172, +0.258]; [+0.167, +0.248]; [+0.160, +0.242]; linear(c) − majority: [+0.357, +0.414]; [+0.348, +0.407]; [+0.342, +0.399], seeds 0–2), so the in-distribution half of the paper's linear null does not hold for the LSTM. The Phase 3 "ID_val_heldout" row was a 2-to-3-operation transfer test (train 300 two-operation, test 117 three-operation and 33 two-operation), and its 0.717 against 0.729 is a transfer null, not evidence of no representation.

**MLP over linear, LSTM c, stratified.** The MLP adds +0.022 / +0.045 / +0.041 over linear across checkpoint seeds 0–2 (mean +0.036), as measured.

**Input baselines.** The one-hot and ordinal baselines did not converge on the stratified split (0.853 vs 0.775), so the one-hot is the binding template baseline; on the matched split the ordinal baseline scores below its own control (0.658 vs 0.681) and is not used as a bar there.

**Limit.** Study 2 OOD "above input baseline" rows were not checked against an ordinal baseline.

**Stratified verdicts (depth).** LSTM c yes above input baseline 3 of 3 seeds; LSTM h 1 of 3 (seed 1); Transformer no; on the stratified split depth decodability falls from 0.662 at the encoder input to 0.506 at layer 3 (seed 0 linear), so the encoder loses linearly decodable position (0.995 at the input to 0.18 at layer 3 and in the memory) and per-position depth (at majority by layer 2 in all three seeds) while keeping token identity (0.999 through the memory); identity probes are linear only.

**Layer-wise, linear vs MLP.** Linear depth decodability falls monotonically in all three seeds. The MLP peaks at layer 1 in seeds 1 and 2 (0.680, 0.677, below the one-hot(char, position) MLP baseline of 0.732) before falling to majority at layer 2.

**Value, in-distribution.** Value is not decodable in-distribution by linear or MLP readers in either model (R² 0.02 to 0.11; the 0.112 is Transformer seed 2 MLP, the one "signal, below threshold" row; all CIs against input baselines span zero).

**Decoder memory.** The decoder cross-attends only to the encoder output, which is layer 3 followed by the encoder's final LayerNorm; per-position depth there is at majority (linear 0.508, layer 3 before the norm 0.506, vs majority 0.527, seed 0), so the first generated token is emitted from a memory with no per-position depth code. This is consistent with the Transformer's 70.7% first-error-at-step-0 rate from the behavioral analysis (Phase 1 failure histogram, seed 0: 3,463 of 4,901 failures); it is correlational, and whether the decoder could recover structure by aggregating over tokens is bounded by the identity probes in the layer-wise section.

## Control R² check

63 value rows checked (3 control seeds each). Max control R² over all rows and seeds: 0.025. Rows with any seed > 0.02: 1. Rows with any seed > 0.05: 0.


## Transformer bridge (study1 seed0, linear probes)

| split | value R² Phase 3 stored | unmasked rerun | masked rerun | depth acc Phase 3 stored | unmasked rerun | masked rerun |
|---|---|---|---|---|---|---|
| ID_val_heldout | -0.099 | -0.099 | -0.149 | 0.628 | 0.628 | 0.628 |
| ops4 | -0.033 | -0.033 | -0.004 | 0.412 | 0.412 | 0.431 |
| ops5 | -0.061 | -0.061 | -0.009 | 0.301 | 0.301 | 0.299 |
| ops6 | -0.057 | -0.057 | -0.012 | 0.232 | 0.232 | 0.219 |
| ops7 | -0.138 | -0.138 | -0.019 | 0.170 | 0.170 | 0.158 |

## 240-trained (early-stopped) vs 300-refit

Refit minus 240-trained, over 126 rows: mean +0.0009, median -0.0002, min -0.0532, max +0.3250.

## Label-space ceilings (depth)

| study | split | share outside training classes | ceiling | majority | bar (majority + 0.05) | ceiling − bar |
|---|---|---|---|---|---|---|
| study1 | ID_val_heldout | 0.000 | 1.000 | 0.729 | 0.779 | +0.221 |
| study1 | ops4 | 0.291 | 0.709 | 0.527 | 0.577 | +0.131 |
| study1 | ops5 | 0.471 | 0.529 | 0.471 | 0.521 | +0.009 |
| study1 | ops6 | 0.594 | 0.406 | 0.594 | 0.644 | -0.238 |
| study1 | ops7 | 0.683 | 0.317 | 0.683 | 0.733 | -0.416 |
| study2 | ID_val_heldout | 0.000 | 1.000 | 0.764 | 0.814 | +0.186 |
| study2 | ood | 0.382 | 0.618 | 0.382 | 0.432 | +0.186 |

## Input baseline (depth): one-hot char + position + length, no model

| study | split | linear | MLP mean ± std | control | boot CI MLP − control (s0; s1; s2) | majority |
|---|---|---|---|---|---|---|
| study1 | ID_val_heldout | 0.743 | 0.743 ± 0.000 | 0.636 | [+0.055, +0.108]; [+0.125, +0.178]; [+0.066, +0.117] | 0.729 |
| study1 | ops4 | 0.368 | 0.368 ± 0.000 | 0.479 | [-0.138, -0.125]; [-0.109, -0.099]; [-0.104, -0.093] | 0.527 |
| study1 | ops5 | 0.253 | 0.253 ± 0.000 | 0.349 | [-0.118, -0.107]; [-0.091, -0.081]; [-0.094, -0.085] | 0.471 |
| study1 | ops6 | 0.233 | 0.233 ± 0.000 | 0.256 | [-0.038, -0.033]; [-0.013, -0.009]; [-0.024, -0.018] | 0.594 |
| study1 | ops7 | 0.169 | 0.169 ± 0.000 | 0.192 | [-0.036, -0.033]; [-0.013, -0.010]; [-0.026, -0.022] | 0.683 |
| study2 | ID_val_heldout | 0.981 | 1.000 ± 0.000 | 0.764 | [+0.234, +0.238]; [+0.234, +0.238]; [+0.234, +0.238] | 0.764 |
| study2 | ood | 0.277 | 0.278 ± 0.000 | 0.310 | [-0.036, -0.026]; [-0.036, -0.026]; [-0.036, -0.026] | 0.382 |

Activation-MLP depth fits were rerun to pair them with the input baseline; max |rerun − stored| over 18 depth files: 0.00e+00.

## Masking no-op check (Transformer study1 seed0, ID depth, linear)

Masked and unmasked linear depth predictions differ at 860 of 2945 positions (masked correct at 429, unmasked correct at 431); accuracy 0.6275 vs 0.6282. Activations differ by max 3.123, mean 0.512. Masked run read transformer_study1_seed0_pooled_train.npz, transformer_study1_seed0_pooled_ID_val_heldout.npz, transformer_study1_seed0_pooled_ops4.npz, transformer_study1_seed0_pooled_ops5.npz, transformer_study1_seed0_pooled_ops6.npz, transformer_study1_seed0_pooled_ops7.npz (transformer_encoder_states_mean_pooled_masked).

## Epoch cap (LSTM depth fits whose chosen epoch is ≥ 191)

| fit | chosen epoch (stored / rerun) | stopped by | stop loss at best | final | final-epoch Δ | mean Δ last 10 | still falling |
|---|---|---|---|---|---|---|---|
| lstm_study1_seed1_c_depth_probeseed1 | 195 / 195 | epoch cap | 0.0001 | 0.0002 | -0.00001 | -0.00000 | True |
| lstm_study1_seed2_h_depth_probeseed2 | 193 / 193 | epoch cap | 0.0002 | 0.0002 | +0.00001 | -0.00000 | False |
| lstm_study1_seed2_c_depth_probeseed0 | 197 / 197 | epoch cap | 0.0001 | 0.0001 | -0.00000 | -0.00000 | True |
| lstm_study1_seed2_c_depth_probeseed1 | 200 / 200 | epoch cap | 0.0001 | 0.0001 | -0.00000 | -0.00000 | True |
| lstm_study1_seed2_c_depth_probeseed2 | 198 / 198 | epoch cap | 0.0001 | 0.0001 | -0.00000 | +0.00000 | True |

## STEP 4c: ordinal input baseline (matched protocol, Study 1 depth)

| split | linear | MLP mean ± std | control | boot CI MLP − control (s0; s1; s2) | majority |
|---|---|---|---|---|---|
| ID_val_heldout | 0.589 | 0.658 ± 0.011 | 0.681 | [-0.056, +0.003]; [-0.011, +0.050]; [-0.089, -0.023] | 0.729 |
| ops4 | 0.296 | 0.400 ± 0.014 | 0.445 | [-0.122, -0.111]; [+0.108, +0.125]; [-0.140, -0.130] | 0.527 |
| ops5 | 0.310 | 0.341 ± 0.026 | 0.312 | [-0.055, -0.045]; [+0.188, +0.199]; [-0.065, -0.055] | 0.471 |
| ops6 | 0.197 | 0.282 ± 0.001 | 0.231 | [-0.002, -0.000]; [+0.154, +0.158]; [-0.002, -0.000] | 0.594 |
| ops7 | 0.173 | 0.211 ± 0.000 | 0.176 | [+0.000, +0.000]; [+0.105, +0.106]; [+0.000, +0.000] | 0.683 |

Activation-MLP depth fits rerun for pairing; max |rerun − stored| over 9 files: 0.00e+00.

## STEP 4c: stratified split (Study 1 depth, in-distribution)

Seeded (seed 0) random 300/150 split of study1 val.json, stratified by operation count: train {'2': 100, '3': 200} (carve-out {'2': 80, '3': 160}, early-stop {'3': 40, '2': 20}), test {'3': 100, '2': 50}; train lengths 13–23, test lengths 13–23. Training depth classes [1, 2, 3]; test ceiling 1.000, majority 0.527. Input baselines on this split: one-hot linear 0.697, MLP 0.853 ± 0.001 (control 0.525); ordinal linear 0.593, MLP 0.775 ± 0.004 (control 0.525). The matched columns repeat the Phase 3 comparison rows (ID_val_heldout) for the same checkpoint and state.

| model | seed | state | protocol | linear | mlp | control | ctrl_train | majority | ib_onehot_mlp | ib_ordinal_mlp | boot CI vs control | boot CI vs one-hot | boot CI vs ordinal | n_ex | n_pos | rule verdict | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm | 0 | h | stratified | 0.834 | 0.876 ± 0.005 | 0.497 | 0.559 | 0.527 | 0.853 | 0.775 | [+0.354, +0.405]; [+0.360, +0.407]; [+0.350, +0.395] | [-0.006, +0.050]; [+0.002, +0.057]; [-0.010, +0.047] | [+0.068, +0.125]; [+0.082, +0.134]; [+0.070, +0.128] | 150 | 2856 | yes | yes, not above input baseline |
| lstm | 0 | h | matched (ID) | 0.746 | 0.810 ± 0.002 | 0.513 | 0.604 | 0.729 | 0.743 | 0.658 | [+0.352, +0.399]; [+0.251, +0.307]; [+0.206, +0.264] | [+0.056, +0.074]; [+0.054, +0.076]; [+0.059, +0.081] | [+0.122, +0.146]; [+0.145, +0.175]; [+0.146, +0.173] | 150 | 2945 | yes | yes, above input baseline |
| lstm | 0 | c | stratified | 0.911 | 0.933 ± 0.004 | 0.513 | 0.556 | 0.527 | 0.853 | 0.775 | [+0.396, +0.455]; [+0.388, +0.445]; [+0.389, +0.448] | [+0.056, +0.107]; [+0.044, +0.102]; [+0.058, +0.110] | [+0.124, +0.191]; [+0.123, +0.184]; [+0.133, +0.200] | 150 | 2856 | yes | yes, above input baseline |
| lstm | 0 | c | matched (ID) | 0.717 | 0.801 ± 0.002 | 0.502 | 0.597 | 0.729 | 0.743 | 0.658 | [+0.211, +0.267]; [+0.315, +0.370]; [+0.288, +0.340] | [+0.047, +0.066]; [+0.046, +0.065]; [+0.052, +0.071] | [+0.115, +0.137]; [+0.138, +0.164]; [+0.140, +0.163] | 150 | 2945 | yes | yes, above input baseline |
| lstm | 1 | h | stratified | 0.865 | 0.899 ± 0.005 | 0.505 | 0.559 | 0.527 | 0.853 | 0.775 | [+0.388, +0.440]; [+0.367, +0.416]; [+0.354, +0.407] | [+0.022, +0.078]; [+0.014, +0.071]; [+0.020, +0.072] | [+0.096, +0.157]; [+0.094, +0.147]; [+0.095, +0.158] | 150 | 2856 | yes | yes, above input baseline |
| lstm | 1 | h | matched (ID) | 0.758 | 0.817 ± 0.016 | 0.575 | 0.619 | 0.729 | 0.743 | 0.658 | [+0.204, +0.255]; [+0.258, +0.309]; [+0.184, +0.245] | [+0.065, +0.087]; [+0.079, +0.106]; [+0.043, +0.065] | [+0.132, +0.157]; [+0.171, +0.205]; [+0.132, +0.157] | 150 | 2945 | yes | yes, above input baseline |
| lstm | 1 | c | stratified | 0.904 | 0.949 ± 0.000 | 0.505 | 0.557 | 0.527 | 0.853 | 0.775 | [+0.427, +0.481]; [+0.415, +0.471]; [+0.408, +0.465] | [+0.069, +0.120]; [+0.071, +0.122]; [+0.074, +0.124] | [+0.138, +0.200]; [+0.145, +0.204]; [+0.145, +0.212] | 150 | 2856 | yes | yes, above input baseline |
| lstm | 1 | c | matched (ID) | 0.709 | 0.785 ± 0.013 | 0.607 | 0.605 | 0.729 | 0.743 | 0.658 | [+0.198, +0.246]; [+0.101, +0.162]; [+0.153, +0.206] | [+0.041, +0.061]; [+0.011, +0.035]; [+0.043, +0.061] | [+0.108, +0.131]; [+0.105, +0.134]; [+0.131, +0.153] | 150 | 2945 | yes | yes, above input baseline |
| lstm | 2 | h | stratified | 0.837 | 0.877 ± 0.005 | 0.498 | 0.555 | 0.527 | 0.853 | 0.775 | [+0.355, +0.410]; [+0.359, +0.411]; [+0.345, +0.396] | [-0.012, +0.047]; [+0.002, +0.057]; [-0.001, +0.055] | [+0.061, +0.122]; [+0.082, +0.137]; [+0.077, +0.140] | 150 | 2856 | yes | yes, not above input baseline |
| lstm | 2 | h | matched (ID) | 0.728 | 0.804 ± 0.010 | 0.485 | 0.611 | 0.729 | 0.743 | 0.658 | [+0.311, +0.360]; [+0.318, +0.371]; [+0.250, +0.307] | [+0.057, +0.080]; [+0.057, +0.080]; [+0.035, +0.059] | [+0.124, +0.151]; [+0.150, +0.178]; [+0.125, +0.150] | 150 | 2945 | yes | yes, above input baseline |
| lstm | 2 | c | stratified | 0.896 | 0.937 ± 0.002 | 0.498 | 0.554 | 0.527 | 0.853 | 0.775 | [+0.442, +0.498]; [+0.396, +0.453]; [+0.396, +0.450] | [+0.060, +0.113]; [+0.052, +0.106]; [+0.060, +0.111] | [+0.130, +0.192]; [+0.130, +0.191]; [+0.134, +0.200] | 150 | 2856 | yes | yes, above input baseline |
| lstm | 2 | c | matched (ID) | 0.730 | 0.783 ± 0.009 | 0.518 | 0.603 | 0.729 | 0.743 | 0.658 | [+0.308, +0.361]; [+0.253, +0.310]; [+0.152, +0.212] | [+0.024, +0.047]; [+0.038, +0.065]; [+0.022, +0.042] | [+0.093, +0.117]; [+0.131, +0.163]; [+0.110, +0.134] | 150 | 2945 | yes | yes, above input baseline |
| transformer | 0 | pooled | stratified | 0.522 | 0.531 ± 0.004 | 0.516 | 0.552 | 0.527 | 0.853 | 0.775 | [+0.006, +0.020]; [+0.015, +0.045]; [-0.011, +0.013] | [-0.351, -0.283]; [-0.354, -0.288]; [-0.356, -0.296] | [-0.262, -0.224]; [-0.262, -0.225]; [-0.266, -0.225] | 150 | 2856 | no | no |
| transformer | 0 | pooled | matched (ID) | 0.628 | 0.587 ± 0.008 | 0.584 | 0.574 | 0.729 | 0.743 | 0.658 | [-0.018, +0.033]; [-0.026, +0.013]; [-0.015, +0.034] | [-0.167, -0.124]; [-0.184, -0.148]; [-0.177, -0.133] | [-0.101, -0.052]; [-0.093, -0.049]; [-0.091, -0.039] | 150 | 2945 | no | no |
| transformer | 1 | pooled | stratified | 0.523 | 0.529 ± 0.003 | 0.514 | 0.556 | 0.527 | 0.853 | 0.775 | [+0.007, +0.024]; [+0.009, +0.025]; [+0.006, +0.024] | [-0.354, -0.286]; [-0.361, -0.293]; [-0.356, -0.294] | [-0.267, -0.226]; [-0.268, -0.228]; [-0.265, -0.225] | 150 | 2856 | no | no |
| transformer | 1 | pooled | matched (ID) | 0.546 | 0.499 ± 0.007 | 0.631 | 0.576 | 0.729 | 0.743 | 0.658 | [-0.107, -0.051]; [-0.213, -0.154]; [-0.159, -0.106] | [-0.251, -0.221]; [-0.258, -0.229]; [-0.269, -0.238] | [-0.183, -0.151]; [-0.166, -0.131]; [-0.181, -0.146] | 150 | 2945 | no | no |
| transformer | 2 | pooled | stratified | 0.529 | 0.526 ± 0.001 | 0.517 | 0.558 | 0.527 | 0.853 | 0.775 | [-0.007, +0.012]; [+0.004, +0.032]; [-0.006, +0.016] | [-0.359, -0.297]; [-0.360, -0.295]; [-0.356, -0.297] | [-0.274, -0.235]; [-0.268, -0.229]; [-0.266, -0.225] | 150 | 2856 | no | no |
| transformer | 2 | pooled | matched (ID) | 0.499 | 0.484 ± 0.007 | 0.638 | 0.582 | 0.729 | 0.743 | 0.658 | [-0.150, -0.090]; [-0.249, -0.196]; [-0.144, -0.093] | [-0.275, -0.244]; [-0.283, -0.252]; [-0.266, -0.235] | [-0.207, -0.174]; [-0.191, -0.153]; [-0.179, -0.143] | 150 | 2945 | no | no |

## STEP 4d: stratified split, value (Study 1, in-distribution)

Same strat0 split and cached activations as the stratified depth table above. Input baselines with regression heads: one-hot Ridge R² 0.049, MLP 0.059 ± 0.000 (control -0.007); ordinal Ridge 0.033, MLP 0.038 ± 0.003 (control -0.009). Bars (STEP 4d): MLP mean − std > control + 0.10 and > 0.10, with the bootstrap vs control; the pre-registered rule (MLP mean, not mean − std) is shown alongside.

| model | seed | state | linear (Ridge) | mlp | control | ctrl_train | ib_onehot_mlp | ib_ordinal_mlp | boot CI vs control | boot CI vs one-hot | boot CI vs ordinal | n_ex | n_pos | pre-registered rule | 4d rule | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm | 0 | h | 0.048 | 0.046 ± 0.004 | -0.023 | 0.069 | 0.059 | 0.038 | [+0.003, +0.117]; [-0.011, +0.115]; [+0.018, +0.154] | [-0.076, +0.025]; [-0.089, +0.042]; [-0.078, +0.040] | [-0.024, +0.038]; [-0.041, +0.048]; [-0.036, +0.051] | 150 | 2856 | no | no | no |
| lstm | 0 | c | 0.052 | 0.040 ± 0.003 | -0.030 | 0.072 | 0.059 | 0.038 | [+0.024, +0.123]; [-0.027, +0.132]; [+0.039, +0.181] | [-0.069, +0.030]; [-0.098, +0.037]; [-0.065, +0.028] | [-0.013, +0.031]; [-0.050, +0.036]; [-0.020, +0.035] | 150 | 2856 | no | no | no |
| lstm | 1 | h | 0.056 | 0.056 ± 0.004 | -0.020 | 0.071 | 0.059 | 0.038 | [+0.016, +0.120]; [-0.010, +0.136]; [+0.032, +0.121] | [-0.076, +0.036]; [-0.101, +0.064]; [-0.058, +0.040] | [-0.026, +0.045]; [-0.057, +0.059]; [-0.009, +0.039] | 150 | 2856 | no | no | no |
| lstm | 1 | c | 0.063 | 0.051 ± 0.010 | -0.011 | 0.049 | 0.059 | 0.038 | [-0.025, +0.086]; [-0.011, +0.125]; [+0.034, +0.104] | [-0.108, +0.031]; [-0.087, +0.054]; [-0.059, +0.041] | [-0.059, +0.035]; [-0.037, +0.048]; [-0.009, +0.039] | 150 | 2856 | no | no | no |
| lstm | 2 | h | 0.068 | 0.072 ± 0.003 | -0.018 | 0.060 | 0.059 | 0.038 | [+0.003, +0.145]; [-0.027, +0.153]; [+0.009, +0.138] | [-0.078, +0.084]; [-0.098, +0.074]; [-0.072, +0.079] | [-0.040, +0.084]; [-0.055, +0.068]; [-0.037, +0.073] | 150 | 2856 | no | no | no |
| lstm | 2 | c | 0.065 | 0.075 ± 0.009 | -0.028 | 0.055 | 0.059 | 0.038 | [-0.012, +0.168]; [-0.006, +0.182]; [+0.005, +0.145] | [-0.102, +0.079]; [-0.092, +0.117]; [-0.080, +0.066] | [-0.064, +0.083]; [-0.052, +0.106]; [-0.042, +0.064] | 150 | 2856 | no | no | no |
| transformer | 0 | pooled | 0.072 | 0.081 ± 0.008 | -0.013 | 0.031 | 0.059 | 0.038 | [-0.064, +0.191]; [-0.049, +0.164]; [-0.126, +0.172] | [-0.150, +0.131]; [-0.153, +0.115]; [-0.209, +0.113] | [-0.102, +0.142]; [-0.107, +0.121]; [-0.174, +0.118] | 150 | 2856 | no | no | no |
| transformer | 1 | pooled | 0.030 | 0.019 ± 0.008 | -0.024 | 0.033 | 0.059 | 0.038 | [-0.176, +0.119]; [-0.117, +0.130]; [-0.193, +0.129] | [-0.242, +0.059]; [-0.217, +0.056]; [-0.306, +0.054] | [-0.220, +0.069]; [-0.178, +0.050]; [-0.259, +0.054] | 150 | 2856 | no | no | no |
| transformer | 2 | pooled | 0.059 | 0.112 ± 0.011 | -0.021 | 0.038 | 0.059 | 0.038 | [+0.008, +0.250]; [+0.007, +0.206]; [-0.038, +0.229] | [-0.070, +0.160]; [-0.089, +0.134]; [-0.148, +0.137] | [-0.041, +0.181]; [-0.049, +0.144]; [-0.110, +0.153] | 150 | 2856 | signal, below threshold | signal, below threshold | signal, below threshold |

## STEP 4d: Transformer layer-wise depth sanity (study1 seed0, stratified split, report only)

**Pre-registered reading.** If (a) the encoder input decodes near the one-hot(char, position) baseline and (b) decoding falls layer by layer, position information is lost inside the encoder and that is a finding; if (a) is at majority, the extraction or scaler is suspect.

Majority 0.527. One-hot(char, position) baseline, no length: linear 0.664, MLP 0.732 ± 0.001 (control 0.523). Mean pool recomputed here vs the cached stratified activations: max |diff| 0.00e+00.

| features | dim | linear | mlp | control | ctrl_train | boot CI vs control | boot CI vs one-hot(char, position) |
|---|---|---|---|---|---|---|---|
| encoder_input | 256 | 0.662 | 0.659 ± 0.006 | 0.527 | 0.546 | [+0.115, +0.167]; [+0.108, +0.151]; [+0.102, +0.150] | [-0.088, -0.043]; [-0.094, -0.054]; [-0.103, -0.054] |
| layer1 | 256 | 0.602 | 0.576 ± 0.026 | 0.522 | 0.544 | [+0.056, +0.098]; [+0.047, +0.089]; [+0.008, +0.027] | [-0.155, -0.105]; [-0.167, -0.121]; [-0.219, -0.167] |
| layer2 | 256 | 0.524 | 0.534 ± 0.002 | 0.514 | 0.554 | [+0.006, +0.022]; [+0.022, +0.048]; [+0.008, +0.021] | [-0.224, -0.168]; [-0.222, -0.169]; [-0.227, -0.175] |
| layer3 | 256 | 0.506 | 0.529 ± 0.003 | 0.518 | 0.566 | [-0.000, +0.017]; [+0.005, +0.023]; [+0.001, +0.015] | [-0.235, -0.180]; [-0.226, -0.173]; [-0.230, -0.177] |
| mean_pool | 256 | 0.522 | 0.531 ± 0.004 | 0.516 | 0.552 | [+0.006, +0.020]; [+0.015, +0.045]; [-0.011, +0.013] | [-0.221, -0.170]; [-0.227, -0.172]; [-0.234, -0.180] |

## STEP 4e: linear paired bootstraps (stratified split, depth)

One-hot linear 0.697; majority 0.527. One CI per row (the linear probe has no seed); 1,000 example-level resamples.

| row | linear | CI linear − one-hot linear | CI linear − majority |
|---|---|---|---|
| lstm_seed0_h | 0.834 | [+0.100, +0.173] | [+0.283, +0.332] |
| lstm_seed0_c | 0.911 | [+0.172, +0.258] | [+0.357, +0.414] |
| lstm_seed1_h | 0.865 | [+0.133, +0.203] | [+0.314, +0.360] |
| lstm_seed1_c | 0.904 | [+0.167, +0.248] | [+0.348, +0.407] |
| lstm_seed2_h | 0.837 | [+0.106, +0.177] | [+0.287, +0.335] |
| lstm_seed2_c | 0.896 | [+0.160, +0.242] | [+0.342, +0.399] |
| transformer_seed0_pooled | 0.522 | [-0.199, -0.150] | [-0.020, +0.009] |
| transformer_seed1_pooled | 0.523 | [-0.198, -0.150] | [-0.017, +0.009] |
| transformer_seed2_pooled | 0.529 | [-0.191, -0.144] | [-0.009, +0.013] |

## STEP 4e: Transformer layer-wise depth, all seeds (stratified split)

Majority 0.527; one-hot(char, position) baseline linear 0.664, MLP 0.732. Seed 0 rows other than memory are from STEP 4d. memory = layer 3 after the encoder's final LayerNorm (what the decoder reads). Cells: linear / MLP (control).

| features | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| encoder_input | 0.662 / 0.659 (0.527) | 0.663 / 0.619 (0.527) | 0.662 / 0.660 (0.527) |
| layer1 | 0.602 / 0.576 (0.522) | 0.605 / 0.680 (0.522) | 0.648 / 0.677 (0.522) |
| layer2 | 0.524 / 0.534 (0.514) | 0.516 / 0.529 (0.509) | 0.522 / 0.528 (0.511) |
| layer3 | 0.506 / 0.529 (0.518) | 0.513 / 0.523 (0.512) | 0.502 / 0.519 (0.511) |
| mean_pool | 0.522 / 0.531 (0.516) | 0.523 / 0.529 (0.514) | 0.529 / 0.526 (0.517) |
| memory | 0.508 / 0.529 (0.518) | 0.512 / 0.523 (0.513) | 0.503 / 0.520 (0.511) |

**Pre-registered reading (identity probes).** If character identity stays near 1.0 through layer 3 while depth falls to majority, the encoder keeps the tokens but does not compute structure; if character identity also decays, the decoder's memory loses the inputs themselves.

## STEP 4e: per-layer identity probes, Transformer (stratified split, linear, report only)

Cells: character-identity accuracy / position-index accuracy.

| features | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| encoder_input | 1.000 / 0.995 | 1.000 / 0.995 | 1.000 / 0.995 |
| layer1 | 1.000 / 0.558 | 1.000 / 0.608 | 1.000 / 0.631 |
| layer2 | 1.000 / 0.238 | 1.000 / 0.238 | 1.000 / 0.263 |
| layer3 | 1.000 / 0.194 | 1.000 / 0.178 | 1.000 / 0.181 |
| mean_pool | 1.000 / 0.249 | 1.000 / 0.244 | 1.000 / 0.252 |
| memory | 0.999 / 0.195 | 1.000 / 0.177 | 1.000 / 0.181 |

Majority: character 0.280, position 0.053.

## Results

Metric: R² for value, accuracy for depth. mlp = 300-refit mean ± std over probe seeds 0–2; mlp240 = early-stopped 240-trained mean; control = permuted-label MLP held-out mean; ctrl_train = control on its own (permuted) training labels; boot CI vs control = 95% CI of paired MLP − control per probe seed (the verdict test); boot CI vs majority = 95% CI of paired MLP − constant predictor of the eval split's own majority class (depth only; reported, not part of the verdict). STEP 4b (depth only): ib_mlp / ib_linear = input baseline (one-hot char, position, length); boot CI vs input = 95% CI of paired MLP(activations) − MLP(input baseline) per probe seed; ib_ordinal_mlp / boot CI vs ordinal = STEP 4c ordinal input baseline (one-hot char, scalar position, length, position/length; Study 1 depth only); ceiling = label-space ceiling. verdict = 4c-qualified verdict for Study 1 depth rows, 4b-qualified elsewhere; rule verdict = the pre-registered rule's output before the qualifiers.

| model | study | seed | state | target | split | linear | mlp | mlp240 | control | ctrl_train | majority | selectivity | boot CI vs control (s0; s1; s2) | boot CI vs majority (s0; s1; s2) | ib_mlp | ib_linear | boot CI vs input (s0; s1; s2) | ib_ordinal_mlp | boot CI vs ordinal (s0; s1; s2) | ceiling | n_ex | n_pos | rule verdict | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lstm | study1 | 0 | h | value | ID_val_heldout | 0.018 | -0.028 ± 0.022 | -0.012 | -0.044 | 0.057 | – | 0.016 | [-0.095, +0.089]; [-0.337, +0.061]; [-0.044, +0.058] | – | – | – | – | – | – | – | 150 | 2945 | no | no |
| lstm | study1 | 0 | h | value | ops4 | -0.008 | -0.016 ± 0.008 | -0.015 | -0.021 | 0.057 | – | 0.006 | [-0.010, +0.032]; [-0.018, +0.034]; [-0.023, +0.002] | – | – | – | – | – | – | – | 150 | 4138 | no | no |
| lstm | study1 | 0 | h | value | ops5 | -0.023 | -0.029 ± 0.004 | -0.028 | -0.026 | 0.057 | – | -0.003 | [-0.007, +0.015]; [-0.010, +0.013]; [-0.018, -0.007] | – | – | – | – | – | – | – | 150 | 5086 | no | no |
| lstm | study1 | 0 | h | value | ops6 | -0.018 | -0.023 ± 0.005 | -0.022 | -0.015 | 0.057 | – | -0.008 | [-0.016, +0.008]; [-0.019, +0.011]; [-0.022, -0.006] | – | – | – | – | – | – | – | 150 | 6099 | no | no |
| lstm | study1 | 0 | h | value | ops7 | -0.018 | -0.019 ± 0.004 | -0.018 | -0.011 | 0.057 | – | -0.007 | [-0.014, +0.003]; [-0.019, +0.001]; [-0.017, -0.008] | – | – | – | – | – | – | – | 150 | 7101 | no | no |
| lstm | study1 | 0 | h | depth | ID_val_heldout | 0.746 | 0.810 ± 0.002 | 0.809 | 0.513 | 0.604 | 0.729 | 0.297 | [+0.352, +0.399]; [+0.251, +0.307]; [+0.206, +0.264] | [+0.056, +0.103]; [+0.056, +0.104]; [+0.063, +0.107] | 0.743 | 0.743 | [+0.056, +0.074]; [+0.054, +0.076]; [+0.059, +0.081] | 0.658 | [+0.122, +0.146]; [+0.145, +0.175]; [+0.146, +0.173] | 1.000 | 150 | 2945 | yes | yes, above input baseline |
| lstm | study1 | 0 | h | depth | ops4 | 0.417 | 0.438 ± 0.008 | 0.427 | 0.391 | 0.604 | 0.527 | 0.047 | [+0.073, +0.108]; [+0.032, +0.066]; [-0.018, +0.019] | [-0.109, -0.089]; [-0.091, -0.067]; [-0.102, -0.078] | 0.368 | 0.368 | [+0.050, +0.070]; [+0.068, +0.092]; [+0.057, +0.082] | 0.400 | [+0.029, +0.049]; [+0.015, +0.042]; [+0.033, +0.057] | 0.709 | 150 | 4138 | no | no |
| lstm | study1 | 0 | h | depth | ops5 | 0.273 | 0.246 ± 0.009 | 0.243 | 0.281 | 0.604 | 0.471 | -0.035 | [-0.031, -0.007]; [-0.030, -0.008]; [-0.078, -0.053] | [-0.235, -0.218]; [-0.221, -0.205]; [-0.243, -0.225] | 0.253 | 0.253 | [-0.017, -0.001]; [-0.003, +0.012]; [-0.025, -0.008] | 0.341 | [-0.088, -0.068]; [-0.127, -0.112]; [-0.097, -0.075] | 0.529 | 150 | 5086 | no | no (ceiling within 0.009 of bar) |
| lstm | study1 | 0 | h | depth | ops6 | 0.146 | 0.141 ± 0.005 | 0.147 | 0.212 | 0.604 | 0.594 | -0.071 | [-0.076, -0.063]; [-0.075, -0.059]; [-0.084, -0.068] | [-0.449, -0.442]; [-0.461, -0.453]; [-0.459, -0.451] | 0.233 | 0.233 | [-0.088, -0.082]; [-0.100, -0.093]; [-0.098, -0.092] | 0.282 | [-0.137, -0.130]; [-0.150, -0.142]; [-0.146, -0.140] | 0.406 | 150 | 6099 | no | structurally capped |
| lstm | study1 | 0 | h | depth | ops7 | 0.129 | 0.123 ± 0.004 | 0.127 | 0.155 | 0.604 | 0.683 | -0.032 | [-0.051, -0.040]; [-0.034, -0.021]; [-0.028, -0.017] | [-0.558, -0.551]; [-0.567, -0.559]; [-0.566, -0.560] | 0.169 | 0.169 | [-0.043, -0.037]; [-0.052, -0.046]; [-0.051, -0.046] | 0.211 | [-0.085, -0.079]; [-0.094, -0.088]; [-0.093, -0.089] | 0.317 | 150 | 7101 | no | structurally capped |
| lstm | study1 | 0 | c | value | ID_val_heldout | -0.020 | 0.002 ± 0.011 | -0.000 | -0.046 | 0.031 | – | 0.048 | [-0.049, +0.097]; [-0.070, +0.078]; [+0.028, +0.100] | – | – | – | – | – | – | – | 150 | 2945 | no | no |
| lstm | study1 | 0 | c | value | ops4 | -0.020 | -0.018 ± 0.006 | -0.021 | -0.024 | 0.031 | – | 0.006 | [+0.004, +0.038]; [-0.017, +0.013]; [-0.003, +0.032] | – | – | – | – | – | – | – | 150 | 4138 | no | no |
| lstm | study1 | 0 | c | value | ops5 | -0.048 | -0.032 ± 0.009 | -0.038 | -0.031 | 0.031 | – | -0.001 | [-0.001, +0.019]; [-0.020, +0.002]; [-0.011, +0.008] | – | – | – | – | – | – | – | 150 | 5086 | no | no |
| lstm | study1 | 0 | c | value | ops6 | -0.035 | -0.025 ± 0.004 | -0.029 | -0.017 | 0.031 | – | -0.008 | [-0.010, +0.010]; [-0.024, -0.005]; [-0.008, +0.007] | – | – | – | – | – | – | – | 150 | 6099 | no | no |
| lstm | study1 | 0 | c | value | ops7 | -0.037 | -0.019 ± 0.002 | -0.020 | -0.013 | 0.031 | – | -0.005 | [-0.008, +0.007]; [-0.015, -0.005]; [-0.011, +0.004] | – | – | – | – | – | – | – | 150 | 7101 | no | no |
| lstm | study1 | 0 | c | depth | ID_val_heldout | 0.717 | 0.801 ± 0.002 | 0.804 | 0.502 | 0.597 | 0.729 | 0.299 | [+0.211, +0.267]; [+0.315, +0.370]; [+0.288, +0.340] | [+0.048, +0.096]; [+0.045, +0.096]; [+0.053, +0.100] | 0.743 | 0.743 | [+0.047, +0.066]; [+0.046, +0.065]; [+0.052, +0.071] | 0.658 | [+0.115, +0.137]; [+0.138, +0.164]; [+0.140, +0.163] | 1.000 | 150 | 2945 | yes | yes, above input baseline |
| lstm | study1 | 0 | c | depth | ops4 | 0.425 | 0.449 ± 0.010 | 0.449 | 0.362 | 0.597 | 0.527 | 0.087 | [-0.009, +0.025]; [+0.122, +0.158]; [+0.093, +0.129] | [-0.097, -0.074]; [-0.076, -0.054]; [-0.097, -0.074] | 0.368 | 0.368 | [+0.062, +0.086]; [+0.083, +0.106]; [+0.062, +0.085] | 0.400 | [+0.041, +0.065]; [+0.031, +0.056]; [+0.038, +0.060] | 0.709 | 150 | 4138 | no | no |
| lstm | study1 | 0 | c | depth | ops5 | 0.260 | 0.244 ± 0.002 | 0.247 | 0.261 | 0.597 | 0.471 | -0.017 | [-0.076, -0.049]; [-0.001, +0.023]; [-0.014, +0.014] | [-0.236, -0.217]; [-0.236, -0.213]; [-0.238, -0.219] | 0.253 | 0.253 | [-0.018, +0.000]; [-0.018, +0.006]; [-0.021, -0.002] | 0.341 | [-0.089, -0.067]; [-0.142, -0.119]; [-0.092, -0.069] | 0.529 | 150 | 5086 | no | no (ceiling within 0.009 of bar) |
| lstm | study1 | 0 | c | depth | ops6 | 0.149 | 0.150 ± 0.002 | 0.152 | 0.197 | 0.597 | 0.594 | -0.047 | [-0.082, -0.066]; [-0.037, -0.021]; [-0.045, -0.031] | [-0.446, -0.439]; [-0.451, -0.442]; [-0.447, -0.439] | 0.233 | 0.233 | [-0.084, -0.079]; [-0.089, -0.082]; [-0.085, -0.080] | 0.282 | [-0.133, -0.127]; [-0.139, -0.132]; [-0.134, -0.128] | 0.406 | 150 | 6099 | no | structurally capped |
| lstm | study1 | 0 | c | depth | ops7 | 0.113 | 0.126 ± 0.002 | 0.129 | 0.150 | 0.597 | 0.683 | -0.024 | [-0.035, -0.022]; [-0.019, -0.007]; [-0.037, -0.023] | [-0.558, -0.553]; [-0.562, -0.556]; [-0.559, -0.553] | 0.169 | 0.169 | [-0.044, -0.039]; [-0.048, -0.042]; [-0.044, -0.039] | 0.211 | [-0.086, -0.082]; [-0.090, -0.084]; [-0.086, -0.082] | 0.317 | 150 | 7101 | no | structurally capped |
| lstm | study1 | 1 | h | value | ID_val_heldout | -0.005 | -0.038 ± 0.019 | -0.022 | -0.045 | 0.035 | – | 0.007 | [-0.269, +0.042]; [-0.137, +0.060]; [-0.064, +0.106] | – | – | – | – | – | – | – | 150 | 2945 | no | no |
| lstm | study1 | 1 | h | value | ops4 | -0.016 | -0.020 ± 0.002 | -0.016 | -0.024 | 0.035 | – | 0.003 | [-0.050, +0.012]; [-0.032, +0.014]; [-0.018, +0.018] | – | – | – | – | – | – | – | 150 | 4138 | no | no |
| lstm | study1 | 1 | h | value | ops5 | -0.030 | -0.028 ± 0.006 | -0.033 | -0.027 | 0.035 | – | -0.001 | [-0.026, +0.001]; [-0.000, +0.025]; [-0.014, +0.010] | – | – | – | – | – | – | – | 150 | 5086 | no | no |
| lstm | study1 | 1 | h | value | ops6 | -0.038 | -0.034 ± 0.003 | -0.036 | -0.018 | 0.035 | – | -0.016 | [-0.042, -0.007]; [-0.028, -0.001]; [-0.032, +0.001] | – | – | – | – | – | – | – | 150 | 6099 | no | no |
| lstm | study1 | 1 | h | value | ops7 | -0.026 | -0.024 ± 0.004 | -0.023 | -0.014 | 0.035 | – | -0.010 | [-0.031, -0.008]; [-0.020, +0.002]; [-0.018, -0.001] | – | – | – | – | – | – | – | 150 | 7101 | no | no |
| lstm | study1 | 1 | h | depth | ID_val_heldout | 0.758 | 0.817 ± 0.016 | 0.818 | 0.575 | 0.619 | 0.729 | 0.242 | [+0.204, +0.255]; [+0.258, +0.309]; [+0.184, +0.245] | [+0.067, +0.112]; [+0.082, +0.132]; [+0.046, +0.094] | 0.743 | 0.743 | [+0.065, +0.087]; [+0.079, +0.106]; [+0.043, +0.065] | 0.658 | [+0.132, +0.157]; [+0.171, +0.205]; [+0.132, +0.157] | 1.000 | 150 | 2945 | yes | yes, above input baseline |
| lstm | study1 | 1 | h | depth | ops4 | 0.420 | 0.440 ± 0.015 | 0.431 | 0.413 | 0.619 | 0.527 | 0.027 | [-0.008, +0.028]; [+0.062, +0.093]; [-0.026, +0.011] | [-0.105, -0.082]; [-0.080, -0.056]; [-0.114, -0.092] | 0.368 | 0.368 | [+0.054, +0.078]; [+0.079, +0.104]; [+0.044, +0.068] | 0.400 | [+0.032, +0.057]; [+0.026, +0.053]; [+0.022, +0.044] | 0.709 | 150 | 4138 | no | no |
| lstm | study1 | 1 | h | depth | ops5 | 0.277 | 0.240 ± 0.006 | 0.243 | 0.290 | 0.619 | 0.471 | -0.050 | [-0.071, -0.045]; [-0.054, -0.028]; [-0.067, -0.035] | [-0.234, -0.220]; [-0.247, -0.231]; [-0.236, -0.217] | 0.253 | 0.253 | [-0.017, -0.002]; [-0.029, -0.014]; [-0.019, +0.000] | 0.341 | [-0.087, -0.069]; [-0.153, -0.138]; [-0.090, -0.067] | 0.529 | 150 | 5086 | no | no (ceiling within 0.009 of bar) |
| lstm | study1 | 1 | h | depth | ops6 | 0.148 | 0.144 ± 0.005 | 0.151 | 0.211 | 0.619 | 0.594 | -0.067 | [-0.076, -0.062]; [-0.066, -0.050]; [-0.083, -0.066] | [-0.447, -0.439]; [-0.455, -0.448]; [-0.459, -0.453] | 0.233 | 0.233 | [-0.086, -0.080]; [-0.094, -0.088]; [-0.099, -0.093] | 0.282 | [-0.134, -0.128]; [-0.144, -0.137]; [-0.147, -0.140] | 0.406 | 150 | 6099 | no | structurally capped |
| lstm | study1 | 1 | h | depth | ops7 | 0.127 | 0.125 ± 0.004 | 0.130 | 0.160 | 0.619 | 0.683 | -0.035 | [-0.051, -0.039]; [-0.029, -0.017]; [-0.042, -0.028] | [-0.556, -0.550]; [-0.561, -0.555]; [-0.565, -0.559] | 0.169 | 0.169 | [-0.041, -0.037]; [-0.046, -0.041]; [-0.050, -0.046] | 0.211 | [-0.083, -0.079]; [-0.088, -0.084]; [-0.093, -0.088] | 0.317 | 150 | 7101 | no | structurally capped |
| lstm | study1 | 1 | c | value | ID_val_heldout | 0.002 | -0.020 ± 0.023 | 0.012 | -0.033 | 0.039 | – | 0.013 | [-0.252, +0.052]; [-0.043, +0.084]; [-0.128, +0.077] | – | – | – | – | – | – | – | 150 | 2945 | no | no |
| lstm | study1 | 1 | c | value | ops4 | -0.009 | -0.024 ± 0.007 | -0.014 | -0.022 | 0.039 | – | -0.002 | [-0.050, +0.010]; [-0.013, +0.016]; [-0.046, +0.004] | – | – | – | – | – | – | – | 150 | 4138 | no | no |
| lstm | study1 | 1 | c | value | ops5 | -0.021 | -0.044 ± 0.005 | -0.032 | -0.025 | 0.039 | – | -0.019 | [-0.044, -0.012]; [-0.038, -0.009]; [-0.034, -0.003] | – | – | – | – | – | – | – | 150 | 5086 | no | no |
| lstm | study1 | 1 | c | value | ops6 | -0.038 | -0.041 ± 0.003 | -0.034 | -0.015 | 0.039 | – | -0.025 | [-0.058, -0.016]; [-0.049, -0.010]; [-0.048, -0.009] | – | – | – | – | – | – | – | 150 | 6099 | no | no |
| lstm | study1 | 1 | c | value | ops7 | -0.032 | -0.031 ± 0.003 | -0.022 | -0.014 | 0.039 | – | -0.017 | [-0.039, -0.013]; [-0.029, -0.007]; [-0.027, -0.010] | – | – | – | – | – | – | – | 150 | 7101 | no | no |
| lstm | study1 | 1 | c | depth | ID_val_heldout | 0.709 | 0.785 ± 0.013 | 0.788 | 0.607 | 0.605 | 0.729 | 0.178 | [+0.198, +0.246]; [+0.101, +0.162]; [+0.153, +0.206] | [+0.042, +0.091]; [+0.009, +0.066]; [+0.043, +0.089] | 0.743 | 0.743 | [+0.041, +0.061]; [+0.011, +0.035]; [+0.043, +0.061] | 0.658 | [+0.108, +0.131]; [+0.105, +0.134]; [+0.131, +0.153] | 1.000 | 150 | 2945 | yes | yes, above input baseline |
| lstm | study1 | 1 | c | depth | ops4 | 0.422 | 0.446 ± 0.001 | 0.443 | 0.432 | 0.605 | 0.527 | 0.014 | [+0.034, +0.071]; [-0.033, +0.000]; [-0.009, +0.022] | [-0.093, -0.069]; [-0.096, -0.069]; [-0.092, -0.069] | 0.368 | 0.368 | [+0.066, +0.090]; [+0.064, +0.090]; [+0.066, +0.091] | 0.400 | [+0.045, +0.069]; [+0.011, +0.040]; [+0.044, +0.067] | 0.709 | 150 | 4138 | no | no |
| lstm | study1 | 1 | c | depth | ops5 | 0.266 | 0.259 ± 0.011 | 0.252 | 0.308 | 0.605 | 0.471 | -0.048 | [-0.055, -0.026]; [-0.072, -0.046]; [-0.058, -0.032] | [-0.232, -0.211]; [-0.208, -0.184]; [-0.228, -0.207] | 0.253 | 0.253 | [-0.014, +0.006]; [+0.010, +0.033]; [-0.010, +0.011] | 0.341 | [-0.085, -0.060]; [-0.114, -0.091]; [-0.081, -0.056] | 0.529 | 150 | 5086 | no | no (ceiling within 0.009 of bar) |
| lstm | study1 | 1 | c | depth | ops6 | 0.141 | 0.152 ± 0.005 | 0.147 | 0.222 | 0.605 | 0.594 | -0.070 | [-0.065, -0.047]; [-0.096, -0.084]; [-0.069, -0.056] | [-0.450, -0.442]; [-0.439, -0.431]; [-0.448, -0.440] | 0.233 | 0.233 | [-0.089, -0.082]; [-0.078, -0.072]; [-0.087, -0.081] | 0.282 | [-0.137, -0.131]; [-0.128, -0.121]; [-0.135, -0.129] | 0.406 | 150 | 6099 | no | structurally capped |
| lstm | study1 | 1 | c | depth | ops7 | 0.117 | 0.129 ± 0.006 | 0.124 | 0.170 | 0.605 | 0.683 | -0.041 | [-0.055, -0.041]; [-0.058, -0.047]; [-0.028, -0.017] | [-0.561, -0.555]; [-0.549, -0.543]; [-0.560, -0.554] | 0.169 | 0.169 | [-0.046, -0.042]; [-0.034, -0.029]; [-0.045, -0.041] | 0.211 | [-0.089, -0.084]; [-0.077, -0.071]; [-0.087, -0.083] | 0.317 | 150 | 7101 | no | structurally capped |
| lstm | study1 | 2 | h | value | ID_val_heldout | 0.012 | 0.015 ± 0.024 | 0.002 | -0.034 | 0.042 | – | 0.049 | [+0.013, +0.114]; [+0.020, +0.124]; [-0.098, +0.073] | – | – | – | – | – | – | – | 150 | 2945 | no | no |
| lstm | study1 | 2 | h | value | ops4 | -0.018 | -0.017 ± 0.004 | -0.023 | -0.018 | 0.042 | – | 0.001 | [-0.013, +0.031]; [-0.016, +0.026]; [-0.039, +0.014] | – | – | – | – | – | – | – | 150 | 4138 | no | no |
| lstm | study1 | 2 | h | value | ops5 | -0.028 | -0.022 ± 0.003 | -0.038 | -0.018 | 0.042 | – | -0.004 | [-0.014, +0.015]; [-0.008, +0.021]; [-0.026, +0.009] | – | – | – | – | – | – | – | 150 | 5086 | no | no |
| lstm | study1 | 2 | h | value | ops6 | -0.025 | -0.022 ± 0.003 | -0.028 | -0.013 | 0.042 | – | -0.010 | [-0.018, +0.003]; [-0.018, +0.005]; [-0.028, -0.003] | – | – | – | – | – | – | – | 150 | 6099 | no | no |
| lstm | study1 | 2 | h | value | ops7 | -0.020 | -0.016 ± 0.002 | -0.021 | -0.013 | 0.042 | – | -0.004 | [-0.009, +0.003]; [-0.013, +0.002]; [-0.023, -0.001] | – | – | – | – | – | – | – | 150 | 7101 | no | no |
| lstm | study1 | 2 | h | depth | ID_val_heldout | 0.728 | 0.804 ± 0.010 | 0.806 | 0.485 | 0.611 | 0.729 | 0.319 | [+0.311, +0.360]; [+0.318, +0.371]; [+0.250, +0.307] | [+0.060, +0.107]; [+0.058, +0.108]; [+0.037, +0.085] | 0.743 | 0.743 | [+0.057, +0.080]; [+0.057, +0.080]; [+0.035, +0.059] | 0.658 | [+0.124, +0.151]; [+0.150, +0.178]; [+0.125, +0.150] | 1.000 | 150 | 2945 | yes | yes, above input baseline |
| lstm | study1 | 2 | h | depth | ops4 | 0.421 | 0.462 ± 0.009 | 0.454 | 0.360 | 0.611 | 0.527 | 0.101 | [+0.082, +0.121]; [+0.125, +0.164]; [+0.039, +0.079] | [-0.083, -0.056]; [-0.067, -0.041]; [-0.087, -0.062] | 0.368 | 0.368 | [+0.075, +0.103]; [+0.092, +0.118]; [+0.073, +0.098] | 0.400 | [+0.054, +0.083]; [+0.038, +0.069]; [+0.049, +0.073] | 0.709 | 150 | 4138 | no | no |
| lstm | study1 | 2 | h | depth | ops5 | 0.286 | 0.276 ± 0.004 | 0.277 | 0.261 | 0.611 | 0.471 | 0.015 | [-0.002, +0.023]; [+0.026, +0.053]; [-0.019, +0.010] | [-0.203, -0.185]; [-0.200, -0.181]; [-0.210, -0.191] | 0.253 | 0.253 | [+0.015, +0.033]; [+0.018, +0.037]; [+0.008, +0.027] | 0.341 | [-0.055, -0.035]; [-0.107, -0.087]; [-0.064, -0.040] | 0.529 | 150 | 5086 | no | no (ceiling within 0.009 of bar) |
| lstm | study1 | 2 | h | depth | ops6 | 0.157 | 0.152 ± 0.003 | 0.155 | 0.195 | 0.611 | 0.594 | -0.043 | [-0.049, -0.033]; [-0.042, -0.024]; [-0.064, -0.047] | [-0.442, -0.434]; [-0.447, -0.438]; [-0.449, -0.440] | 0.233 | 0.233 | [-0.080, -0.074]; [-0.086, -0.078]; [-0.088, -0.080] | 0.282 | [-0.129, -0.122]; [-0.135, -0.127]; [-0.136, -0.128] | 0.406 | 150 | 6099 | no | structurally capped |
| lstm | study1 | 2 | h | depth | ops7 | 0.135 | 0.131 ± 0.003 | 0.133 | 0.153 | 0.611 | 0.683 | -0.021 | [-0.034, -0.022]; [-0.022, -0.009]; [-0.027, -0.015] | [-0.552, -0.545]; [-0.557, -0.549]; [-0.558, -0.551] | 0.169 | 0.169 | [-0.037, -0.032]; [-0.042, -0.035]; [-0.043, -0.038] | 0.211 | [-0.079, -0.074]; [-0.084, -0.077]; [-0.085, -0.080] | 0.317 | 150 | 7101 | no | structurally capped |
| lstm | study1 | 2 | c | value | ID_val_heldout | -0.065 | -0.012 ± 0.010 | 0.000 | -0.035 | 0.030 | – | 0.023 | [-0.008, +0.036]; [-0.020, +0.096]; [-0.079, +0.057] | – | – | – | – | – | – | – | 150 | 2945 | signal, below threshold | signal, below threshold |
| lstm | study1 | 2 | c | value | ops4 | -0.053 | -0.032 ± 0.002 | -0.031 | -0.022 | 0.030 | – | -0.010 | [-0.033, +0.007]; [-0.051, -0.001]; [-0.047, +0.003] | – | – | – | – | – | – | – | 150 | 4138 | no | no |
| lstm | study1 | 2 | c | value | ops5 | -0.100 | -0.044 ± 0.011 | -0.054 | -0.027 | 0.030 | – | -0.018 | [-0.033, -0.012]; [-0.049, -0.012]; [-0.022, +0.005] | – | – | – | – | – | – | – | 150 | 5086 | signal, below threshold | signal, below threshold |
| lstm | study1 | 2 | c | value | ops6 | -0.080 | -0.037 ± 0.004 | -0.039 | -0.014 | 0.030 | – | -0.023 | [-0.031, -0.008]; [-0.063, -0.017]; [-0.034, -0.009] | – | – | – | – | – | – | – | 150 | 6099 | no | no |
| lstm | study1 | 2 | c | value | ops7 | -0.076 | -0.029 ± 0.003 | -0.031 | -0.015 | 0.030 | – | -0.014 | [-0.020, -0.006]; [-0.039, -0.013]; [-0.019, -0.004] | – | – | – | – | – | – | – | 150 | 7101 | no | no |
| lstm | study1 | 2 | c | depth | ID_val_heldout | 0.730 | 0.783 ± 0.009 | 0.766 | 0.518 | 0.603 | 0.729 | 0.265 | [+0.308, +0.361]; [+0.253, +0.310]; [+0.152, +0.212] | [+0.024, +0.079]; [+0.040, +0.094]; [+0.022, +0.071] | 0.743 | 0.743 | [+0.024, +0.047]; [+0.038, +0.065]; [+0.022, +0.042] | 0.658 | [+0.093, +0.117]; [+0.131, +0.163]; [+0.110, +0.134] | 1.000 | 150 | 2945 | yes | yes, above input baseline |
| lstm | study1 | 2 | c | depth | ops4 | 0.428 | 0.466 ± 0.010 | 0.443 | 0.385 | 0.603 | 0.527 | 0.081 | [+0.098, +0.133]; [+0.093, +0.129]; [-0.001, +0.035] | [-0.071, -0.046]; [-0.065, -0.039]; [-0.087, -0.062] | 0.368 | 0.368 | [+0.088, +0.114]; [+0.094, +0.120]; [+0.071, +0.097] | 0.400 | [+0.068, +0.092]; [+0.041, +0.070]; [+0.049, +0.073] | 0.709 | 150 | 4138 | no | no |
| lstm | study1 | 2 | c | depth | ops5 | 0.283 | 0.279 ± 0.003 | 0.272 | 0.278 | 0.603 | 0.471 | 0.001 | [-0.012, +0.016]; [+0.008, +0.038]; [-0.037, -0.007] | [-0.205, -0.180]; [-0.200, -0.177]; [-0.207, -0.184] | 0.253 | 0.253 | [+0.013, +0.037]; [+0.018, +0.040]; [+0.010, +0.032] | 0.341 | [-0.057, -0.031]; [-0.105, -0.084]; [-0.061, -0.035] | 0.529 | 150 | 5086 | no | no (ceiling within 0.009 of bar) |
| lstm | study1 | 2 | c | depth | ops6 | 0.150 | 0.150 ± 0.000 | 0.149 | 0.204 | 0.603 | 0.594 | -0.054 | [-0.062, -0.045]; [-0.041, -0.023]; [-0.085, -0.069] | [-0.449, -0.439]; [-0.449, -0.439]; [-0.449, -0.440] | 0.233 | 0.233 | [-0.088, -0.079]; [-0.088, -0.079]; [-0.088, -0.080] | 0.282 | [-0.136, -0.127]; [-0.137, -0.128]; [-0.136, -0.128] | 0.406 | 150 | 6099 | no | structurally capped |
| lstm | study1 | 2 | c | depth | ops7 | 0.127 | 0.124 ± 0.001 | 0.125 | 0.152 | 0.603 | 0.683 | -0.028 | [-0.042, -0.029]; [-0.026, -0.013]; [-0.036, -0.023] | [-0.563, -0.554]; [-0.565, -0.557]; [-0.562, -0.555] | 0.169 | 0.169 | [-0.048, -0.040]; [-0.050, -0.043]; [-0.047, -0.041] | 0.211 | [-0.091, -0.082]; [-0.092, -0.086]; [-0.089, -0.083] | 0.317 | 150 | 7101 | no | structurally capped |
| lstm | study2 (exploratory) | 0 | h | value | ID_val_heldout | 0.021 | 0.022 ± 0.003 | 0.032 | -0.069 | 0.103 | – | 0.091 | [+0.066, +0.301]; [-0.010, +0.114]; [+0.035, +0.179] | – | – | – | – | – | – | – | 150 | 3174 | no | no |
| lstm | study2 (exploratory) | 0 | h | value | ood | 0.041 | 0.027 ± 0.005 | 0.036 | -0.053 | 0.103 | – | 0.079 | [+0.071, +0.219]; [+0.044, +0.106]; [+0.036, +0.132] | – | – | – | – | – | – | – | 150 | 3171 | no | no |
| lstm | study2 (exploratory) | 0 | h | depth | ID_val_heldout | 0.995 | 1.000 ± 0.000 | 1.000 | 0.756 | 0.765 | 0.764 | 0.244 | [+0.237, +0.242]; [+0.241, +0.248]; [+0.245, +0.253] | [+0.234, +0.238]; [+0.234, +0.238]; [+0.234, +0.238] | 1.000 | 0.981 | [+0.000, +0.000]; [-0.001, +0.000]; [-0.001, +0.000] | – | – | 1.000 | 150 | 3174 | yes | yes, not above input baseline |
| lstm | study2 (exploratory) | 0 | h | depth | ood | 0.366 | 0.391 ± 0.000 | 0.390 | 0.311 | 0.765 | 0.382 | 0.080 | [+0.075, +0.089]; [+0.074, +0.088]; [+0.069, +0.085] | [-0.002, +0.021]; [-0.001, +0.020]; [-0.003, +0.020] | 0.278 | 0.277 | [+0.104, +0.121]; [+0.105, +0.121]; [+0.103, +0.120] | – | – | 0.618 | 150 | 3171 | yes | yes, above input baseline |
| lstm | study2 (exploratory) | 0 | c | value | ID_val_heldout | 0.025 | 0.015 ± 0.020 | 0.042 | -0.057 | 0.055 | – | 0.072 | [-0.028, +0.073]; [-0.034, +0.125]; [+0.043, +0.211] | – | – | – | – | – | – | – | 150 | 3174 | no | no |
| lstm | study2 (exploratory) | 0 | c | value | ood | 0.018 | 0.016 ± 0.027 | 0.038 | -0.044 | 0.055 | – | 0.059 | [-0.021, +0.048]; [+0.027, +0.115]; [+0.047, +0.169] | – | – | – | – | – | – | – | 150 | 3171 | no | no |
| lstm | study2 (exploratory) | 0 | c | depth | ID_val_heldout | 0.997 | 1.000 ± 0.000 | 1.000 | 0.752 | 0.765 | 0.764 | 0.248 | [+0.235, +0.243]; [+0.246, +0.262]; [+0.245, +0.257] | [+0.234, +0.238]; [+0.234, +0.238]; [+0.234, +0.238] | 1.000 | 0.981 | [+0.000, +0.000]; [+0.000, +0.000]; [+0.000, +0.000] | – | – | 1.000 | 150 | 3174 | yes | yes, not above input baseline |
| lstm | study2 (exploratory) | 0 | c | depth | ood | 0.385 | 0.370 ± 0.003 | 0.368 | 0.310 | 0.765 | 0.382 | 0.061 | [+0.057, +0.074]; [+0.048, +0.065]; [+0.051, +0.067] | [-0.019, +0.004]; [-0.026, -0.005]; [-0.024, -0.001] | 0.278 | 0.277 | [+0.088, +0.105]; [+0.080, +0.097]; [+0.082, +0.100] | – | – | 0.618 | 150 | 3171 | no | no |
| lstm | study2 (exploratory) | 1 | h | value | ID_val_heldout | -0.019 | 0.019 ± 0.007 | 0.034 | -0.078 | 0.113 | – | 0.096 | [-0.070, +0.159]; [-0.009, +0.162]; [+0.004, +0.164] | – | – | – | – | – | – | – | 150 | 3174 | no | no |
| lstm | study2 (exploratory) | 1 | h | value | ood | -0.002 | 0.043 ± 0.004 | 0.045 | -0.066 | 0.113 | – | 0.109 | [+0.072, +0.181]; [+0.076, +0.191]; [+0.051, +0.162] | – | – | – | – | – | – | – | 150 | 3171 | no | no |
| lstm | study2 (exploratory) | 1 | h | depth | ID_val_heldout | 0.998 | 0.998 ± 0.003 | 1.000 | 0.760 | 0.766 | 0.764 | 0.238 | [+0.237, +0.242]; [+0.238, +0.246]; [+0.228, +0.236] | [+0.234, +0.238]; [+0.234, +0.238]; [+0.226, +0.233] | 1.000 | 0.981 | [+0.000, +0.000]; [-0.002, +0.000]; [-0.009, -0.004] | – | – | 1.000 | 150 | 3174 | yes | yes, not above input baseline |
| lstm | study2 (exploratory) | 1 | h | depth | ood | 0.386 | 0.391 ± 0.000 | 0.391 | 0.310 | 0.766 | 0.382 | 0.081 | [+0.073, +0.087]; [+0.073, +0.089]; [+0.071, +0.093] | [-0.002, +0.019]; [-0.001, +0.019]; [-0.004, +0.021] | 0.278 | 0.277 | [+0.103, +0.120]; [+0.104, +0.121]; [+0.101, +0.123] | – | – | 0.618 | 150 | 3171 | yes | yes, above input baseline |
| lstm | study2 (exploratory) | 1 | c | value | ID_val_heldout | 0.002 | 0.025 ± 0.018 | 0.042 | -0.077 | 0.047 | – | 0.102 | [+0.057, +0.249]; [-0.122, +0.125]; [-0.111, +0.187] | – | – | – | – | – | – | – | 150 | 3174 | no | no |
| lstm | study2 (exploratory) | 1 | c | value | ood | 0.049 | 0.058 ± 0.019 | 0.066 | -0.055 | 0.047 | – | 0.113 | [+0.158, +0.283]; [+0.032, +0.113]; [-0.030, +0.105] | – | – | – | – | – | – | – | 150 | 3171 | no | no |
| lstm | study2 (exploratory) | 1 | c | depth | ID_val_heldout | 0.997 | 1.000 ± 0.000 | 1.000 | 0.760 | 0.766 | 0.764 | 0.240 | [+0.234, +0.238]; [+0.238, +0.248]; [+0.237, +0.243] | [+0.234, +0.238]; [+0.234, +0.238]; [+0.234, +0.238] | 1.000 | 0.981 | [-0.001, +0.000]; [-0.001, +0.000]; [+0.000, +0.000] | – | – | 1.000 | 150 | 3174 | yes | yes, not above input baseline |
| lstm | study2 (exploratory) | 1 | c | depth | ood | 0.409 | 0.423 ± 0.000 | 0.424 | 0.310 | 0.766 | 0.382 | 0.113 | [+0.104, +0.122]; [+0.106, +0.122]; [+0.106, +0.121] | [+0.030, +0.053]; [+0.031, +0.053]; [+0.031, +0.053] | 0.278 | 0.277 | [+0.134, +0.155]; [+0.134, +0.155]; [+0.136, +0.154] | – | – | 0.618 | 150 | 3171 | yes | yes, above input baseline |
| lstm | study2 (exploratory) | 2 | h | value | ID_val_heldout | 0.002 | -0.001 ± 0.006 | 0.020 | -0.071 | 0.125 | – | 0.071 | [-0.023, +0.140]; [+0.001, +0.158]; [-0.068, +0.142] | – | – | – | – | – | – | – | 150 | 3174 | no | no |
| lstm | study2 (exploratory) | 2 | h | value | ood | 0.038 | 0.064 ± 0.004 | 0.066 | -0.059 | 0.125 | – | 0.123 | [+0.081, +0.188]; [+0.070, +0.215]; [+0.062, +0.215] | – | – | – | – | – | – | – | 150 | 3171 | no | no |
| lstm | study2 (exploratory) | 2 | h | depth | ID_val_heldout | 0.995 | 1.000 ± 0.000 | 0.999 | 0.753 | 0.770 | 0.764 | 0.247 | [+0.240, +0.247]; [+0.245, +0.255]; [+0.242, +0.252] | [+0.234, +0.238]; [+0.234, +0.238]; [+0.234, +0.238] | 1.000 | 0.981 | [-0.001, +0.000]; [-0.002, +0.000]; [-0.001, +0.000] | – | – | 1.000 | 150 | 3174 | yes | yes, not above input baseline |
| lstm | study2 (exploratory) | 2 | h | depth | ood | 0.368 | 0.380 ± 0.001 | 0.379 | 0.310 | 0.770 | 0.382 | 0.071 | [+0.065, +0.080]; [+0.064, +0.079]; [+0.060, +0.076] | [-0.011, +0.010]; [-0.012, +0.007]; [-0.012, +0.009] | 0.278 | 0.277 | [+0.094, +0.112]; [+0.092, +0.110]; [+0.093, +0.110] | – | – | 0.618 | 150 | 3171 | no | no |
| lstm | study2 (exploratory) | 2 | c | value | ID_val_heldout | 0.016 | -0.006 ± 0.011 | 0.023 | -0.044 | 0.054 | – | 0.038 | [-0.013, +0.129]; [-0.144, +0.109]; [-0.122, +0.099] | – | – | – | – | – | – | – | 150 | 3174 | no | no |
| lstm | study2 (exploratory) | 2 | c | value | ood | 0.031 | 0.063 ± 0.009 | 0.078 | -0.030 | 0.054 | – | 0.093 | [+0.054, +0.149]; [+0.034, +0.186]; [+0.005, +0.156] | – | – | – | – | – | – | – | 150 | 3171 | no | no |
| lstm | study2 (exploratory) | 2 | c | depth | ID_val_heldout | 0.996 | 1.000 ± 0.000 | 1.000 | 0.759 | 0.767 | 0.764 | 0.241 | [+0.236, +0.245]; [+0.239, +0.246]; [+0.235, +0.247] | [+0.234, +0.238]; [+0.234, +0.238]; [+0.234, +0.238] | 1.000 | 0.981 | [+0.000, +0.000]; [-0.001, +0.000]; [-0.002, +0.000] | – | – | 1.000 | 150 | 3174 | yes | yes, not above input baseline |
| lstm | study2 (exploratory) | 2 | c | depth | ood | 0.385 | 0.401 ± 0.004 | 0.397 | 0.310 | 0.767 | 0.382 | 0.092 | [+0.084, +0.101]; [+0.089, +0.106]; [+0.077, +0.096] | [+0.009, +0.031]; [+0.015, +0.034]; [+0.002, +0.026] | 0.278 | 0.277 | [+0.114, +0.133]; [+0.119, +0.137]; [+0.107, +0.128] | – | – | 0.618 | 150 | 3171 | yes | yes, above input baseline |
| transformer | study1 | 0 | pooled | value | ID_val_heldout | -0.149 | -0.236 ± 0.070 | -0.183 | -0.000 | 0.054 | – | -0.236 | [-1.107, -0.033]; [-0.817, +0.007]; [-1.279, -0.068] | – | – | – | – | – | – | – | 150 | 2945 | no | no |
| transformer | study1 | 0 | pooled | value | ops4 | -0.004 | 0.001 ± 0.007 | 0.012 | -0.018 | 0.054 | – | 0.019 | [-0.075, +0.042]; [-0.026, +0.041]; [-0.126, +0.029] | – | – | – | – | – | – | – | 150 | 4138 | no | no |
| transformer | study1 | 0 | pooled | value | ops5 | -0.009 | -0.005 ± 0.006 | 0.001 | -0.030 | 0.054 | – | 0.025 | [-0.005, +0.048]; [+0.011, +0.057]; [-0.037, +0.034] | – | – | – | – | – | – | – | 150 | 5086 | no | no |
| transformer | study1 | 0 | pooled | value | ops6 | -0.012 | -0.017 ± 0.003 | -0.012 | -0.019 | 0.054 | – | 0.002 | [-0.032, +0.012]; [-0.018, +0.012]; [-0.056, +0.008] | – | – | – | – | – | – | – | 150 | 6099 | no | no |
| transformer | study1 | 0 | pooled | value | ops7 | -0.019 | -0.021 ± 0.005 | -0.015 | -0.014 | 0.054 | – | -0.007 | [-0.042, +0.008]; [-0.031, +0.005]; [-0.068, +0.000] | – | – | – | – | – | – | – | 150 | 7101 | no | no |
| transformer | study1 | 0 | pooled | depth | ID_val_heldout | 0.628 | 0.587 ± 0.008 | 0.592 | 0.584 | 0.574 | 0.729 | 0.003 | [-0.018, +0.033]; [-0.026, +0.013]; [-0.015, +0.034] | [-0.156, -0.109]; [-0.174, -0.129]; [-0.165, -0.120] | 0.743 | 0.743 | [-0.167, -0.124]; [-0.184, -0.148]; [-0.177, -0.133] | 0.658 | [-0.101, -0.052]; [-0.093, -0.049]; [-0.091, -0.039] | 1.000 | 150 | 2945 | no | no |
| transformer | study1 | 0 | pooled | depth | ops4 | 0.431 | 0.372 ± 0.003 | 0.375 | 0.404 | 0.574 | 0.527 | -0.032 | [-0.046, -0.015]; [-0.044, -0.013]; [-0.058, -0.020] | [-0.164, -0.140]; [-0.171, -0.146]; [-0.167, -0.144] | 0.368 | 0.368 | [-0.005, +0.019]; [-0.012, +0.013]; [-0.009, +0.016] | 0.400 | [-0.027, -0.001]; [-0.064, -0.039]; [-0.033, -0.007] | 0.709 | 150 | 4138 | no | no |
| transformer | study1 | 0 | pooled | depth | ops5 | 0.299 | 0.269 ± 0.002 | 0.268 | 0.297 | 0.574 | 0.471 | -0.028 | [-0.036, -0.013]; [-0.035, -0.015]; [-0.047, -0.022] | [-0.212, -0.193]; [-0.208, -0.190]; [-0.215, -0.196] | 0.253 | 0.253 | [+0.007, +0.024]; [+0.010, +0.027]; [+0.004, +0.020] | 0.341 | [-0.063, -0.044]; [-0.114, -0.098]; [-0.066, -0.048] | 0.529 | 150 | 5086 | no | no (ceiling within 0.009 of bar) |
| transformer | study1 | 0 | pooled | depth | ops6 | 0.219 | 0.205 ± 0.001 | 0.202 | 0.217 | 0.574 | 0.594 | -0.012 | [-0.021, -0.003]; [-0.015, +0.002]; [-0.027, -0.009] | [-0.398, -0.384]; [-0.395, -0.381]; [-0.396, -0.382] | 0.233 | 0.233 | [-0.036, -0.025]; [-0.033, -0.022]; [-0.034, -0.023] | 0.282 | [-0.084, -0.073]; [-0.083, -0.071]; [-0.082, -0.071] | 0.406 | 150 | 6099 | no | structurally capped |
| transformer | study1 | 0 | pooled | depth | ops7 | 0.158 | 0.151 ± 0.002 | 0.146 | 0.164 | 0.574 | 0.683 | -0.013 | [-0.015, -0.002]; [-0.018, -0.006]; [-0.025, -0.014] | [-0.537, -0.529]; [-0.534, -0.525]; [-0.539, -0.530] | 0.169 | 0.169 | [-0.023, -0.015]; [-0.020, -0.011]; [-0.024, -0.016] | 0.211 | [-0.066, -0.057]; [-0.062, -0.053]; [-0.066, -0.058] | 0.317 | 150 | 7101 | no | structurally capped |
| transformer | study1 | 1 | pooled | value | ID_val_heldout | -0.161 | -0.167 ± 0.144 | -0.147 | 0.013 | 0.062 | – | -0.180 | [-0.512, +0.023]; [-0.654, +0.040]; [-1.619, -0.059] | – | – | – | – | – | – | – | 150 | 2945 | no | no |
| transformer | study1 | 1 | pooled | value | ops4 | 0.007 | 0.019 ± 0.005 | 0.014 | -0.005 | 0.062 | – | 0.024 | [-0.014, +0.042]; [-0.037, +0.044]; [-0.131, +0.049] | – | – | – | – | – | – | – | 150 | 4138 | no | no |
| transformer | study1 | 1 | pooled | value | ops5 | 0.009 | 0.017 ± 0.002 | 0.011 | -0.015 | 0.062 | – | 0.032 | [+0.023, +0.053]; [+0.015, +0.051]; [-0.008, +0.055] | – | – | – | – | – | – | – | 150 | 5086 | no | no |
| transformer | study1 | 1 | pooled | value | ops6 | -0.008 | -0.003 ± 0.001 | -0.006 | -0.011 | 0.062 | – | 0.008 | [+0.003, +0.019]; [-0.004, +0.017]; [-0.026, +0.019] | – | – | – | – | – | – | – | 150 | 6099 | no | no |
| transformer | study1 | 1 | pooled | value | ops7 | -0.013 | -0.007 ± 0.002 | -0.009 | -0.011 | 0.062 | – | 0.003 | [+0.001, +0.017]; [-0.008, +0.011]; [-0.031, +0.010] | – | – | – | – | – | – | – | 150 | 7101 | no | no |
| transformer | study1 | 1 | pooled | depth | ID_val_heldout | 0.546 | 0.499 ± 0.007 | 0.490 | 0.631 | 0.576 | 0.729 | -0.133 | [-0.107, -0.051]; [-0.213, -0.154]; [-0.159, -0.106] | [-0.248, -0.194]; [-0.258, -0.200]; [-0.265, -0.214] | 0.743 | 0.743 | [-0.251, -0.221]; [-0.258, -0.229]; [-0.269, -0.238] | 0.658 | [-0.183, -0.151]; [-0.166, -0.131]; [-0.181, -0.146] | 1.000 | 150 | 2945 | no | no |
| transformer | study1 | 1 | pooled | depth | ops4 | 0.375 | 0.315 ± 0.011 | 0.310 | 0.442 | 0.576 | 0.527 | -0.127 | [-0.100, -0.067]; [-0.178, -0.152]; [-0.148, -0.120] | [-0.209, -0.185]; [-0.229, -0.208]; [-0.234, -0.212] | 0.368 | 0.368 | [-0.050, -0.026]; [-0.069, -0.048]; [-0.074, -0.053] | 0.400 | [-0.071, -0.046]; [-0.122, -0.098]; [-0.100, -0.076] | 0.709 | 150 | 4138 | no | no |
| transformer | study1 | 1 | pooled | depth | ops5 | 0.279 | 0.237 ± 0.007 | 0.234 | 0.318 | 0.576 | 0.471 | -0.080 | [-0.062, -0.041]; [-0.114, -0.093]; [-0.097, -0.076] | [-0.235, -0.215]; [-0.243, -0.224]; [-0.251, -0.231] | 0.253 | 0.253 | [-0.016, +0.001]; [-0.026, -0.006]; [-0.033, -0.014] | 0.341 | [-0.086, -0.067]; [-0.149, -0.132]; [-0.103, -0.082] | 0.529 | 150 | 5086 | no | no (ceiling within 0.009 of bar) |
| transformer | study1 | 1 | pooled | depth | ops6 | 0.223 | 0.185 ± 0.001 | 0.183 | 0.239 | 0.576 | 0.594 | -0.053 | [-0.047, -0.034]; [-0.074, -0.061]; [-0.059, -0.045] | [-0.416, -0.402]; [-0.414, -0.401]; [-0.416, -0.404] | 0.233 | 0.233 | [-0.054, -0.042]; [-0.052, -0.042]; [-0.055, -0.044] | 0.282 | [-0.102, -0.090]; [-0.102, -0.091]; [-0.104, -0.092] | 0.406 | 150 | 6099 | no | structurally capped |
| transformer | study1 | 1 | pooled | depth | ops7 | 0.177 | 0.146 ± 0.004 | 0.142 | 0.182 | 0.576 | 0.683 | -0.036 | [-0.037, -0.026]; [-0.046, -0.033]; [-0.042, -0.031] | [-0.547, -0.536]; [-0.537, -0.527]; [-0.545, -0.534] | 0.169 | 0.169 | [-0.032, -0.023]; [-0.023, -0.013]; [-0.030, -0.021] | 0.211 | [-0.074, -0.065]; [-0.065, -0.056]; [-0.072, -0.063] | 0.317 | 150 | 7101 | no | structurally capped |
| transformer | study1 | 2 | pooled | value | ID_val_heldout | -0.184 | -0.474 ± 0.208 | -0.799 | 0.006 | 0.074 | – | -0.481 | [-1.292, +0.007]; [-1.886, -0.062]; [-3.052, -0.199] | – | – | – | – | – | – | – | 150 | 2945 | no | no |
| transformer | study1 | 2 | pooled | value | ops4 | 0.041 | 0.053 ± 0.002 | 0.040 | -0.016 | 0.074 | – | 0.069 | [-0.019, +0.098]; [-0.046, +0.099]; [-0.099, +0.103] | – | – | – | – | – | – | – | 150 | 4138 | no | no |
| transformer | study1 | 2 | pooled | value | ops5 | 0.039 | 0.046 ± 0.005 | 0.036 | -0.024 | 0.074 | – | 0.071 | [+0.041, +0.101]; [+0.020, +0.102]; [-0.010, +0.092] | – | – | – | – | – | – | – | 150 | 5086 | no | no |
| transformer | study1 | 2 | pooled | value | ops6 | -0.002 | 0.001 ± 0.005 | -0.005 | -0.014 | 0.074 | – | 0.015 | [+0.001, +0.037]; [-0.022, +0.031]; [-0.050, +0.025] | – | – | – | – | – | – | – | 150 | 6099 | no | no |
| transformer | study1 | 2 | pooled | value | ops7 | -0.008 | -0.002 ± 0.005 | -0.012 | -0.012 | 0.074 | – | 0.009 | [-0.014, +0.035]; [-0.035, +0.026]; [-0.051, +0.025] | – | – | – | – | – | – | – | 150 | 7101 | no | no |
| transformer | study1 | 2 | pooled | depth | ID_val_heldout | 0.499 | 0.484 ± 0.007 | 0.487 | 0.638 | 0.582 | 0.729 | -0.154 | [-0.150, -0.090]; [-0.249, -0.196]; [-0.144, -0.093] | [-0.274, -0.216]; [-0.279, -0.226]; [-0.262, -0.209] | 0.743 | 0.743 | [-0.275, -0.244]; [-0.283, -0.252]; [-0.266, -0.235] | 0.658 | [-0.207, -0.174]; [-0.191, -0.153]; [-0.179, -0.143] | 1.000 | 150 | 2945 | no | no |
| transformer | study1 | 2 | pooled | depth | ops4 | 0.287 | 0.273 ± 0.006 | 0.272 | 0.464 | 0.582 | 0.527 | -0.191 | [-0.183, -0.158]; [-0.246, -0.226]; [-0.177, -0.154] | [-0.266, -0.248]; [-0.269, -0.252]; [-0.255, -0.237] | 0.368 | 0.368 | [-0.107, -0.089]; [-0.110, -0.092]; [-0.096, -0.078] | 0.400 | [-0.129, -0.109]; [-0.162, -0.143]; [-0.120, -0.101] | 0.709 | 150 | 4138 | no | no |
| transformer | study1 | 2 | pooled | depth | ops5 | 0.208 | 0.198 ± 0.003 | 0.196 | 0.337 | 0.582 | 0.471 | -0.139 | [-0.134, -0.114]; [-0.178, -0.162]; [-0.134, -0.114] | [-0.283, -0.268]; [-0.282, -0.268]; [-0.277, -0.261] | 0.253 | 0.253 | [-0.065, -0.051]; [-0.064, -0.051]; [-0.058, -0.044] | 0.341 | [-0.135, -0.119]; [-0.189, -0.175]; [-0.128, -0.112] | 0.529 | 150 | 5086 | no | no (ceiling within 0.009 of bar) |
| transformer | study1 | 2 | pooled | depth | ops6 | 0.164 | 0.154 ± 0.003 | 0.153 | 0.253 | 0.582 | 0.594 | -0.100 | [-0.097, -0.084]; [-0.127, -0.116]; [-0.093, -0.081] | [-0.445, -0.435]; [-0.449, -0.439]; [-0.443, -0.432] | 0.233 | 0.233 | [-0.084, -0.075]; [-0.088, -0.080]; [-0.081, -0.072] | 0.282 | [-0.132, -0.123]; [-0.137, -0.129]; [-0.129, -0.120] | 0.406 | 150 | 6099 | no | structurally capped |
| transformer | study1 | 2 | pooled | depth | ops7 | 0.124 | 0.116 ± 0.003 | 0.115 | 0.191 | 0.582 | 0.683 | -0.075 | [-0.071, -0.062]; [-0.096, -0.090]; [-0.069, -0.060] | [-0.571, -0.565]; [-0.573, -0.567]; [-0.566, -0.558] | 0.169 | 0.169 | [-0.056, -0.052]; [-0.058, -0.053]; [-0.051, -0.045] | 0.211 | [-0.099, -0.094]; [-0.100, -0.096]; [-0.093, -0.087] | 0.317 | 150 | 7101 | no | structurally capped |
| transformer | study2 (exploratory) | 0 | pooled | value | ID_val_heldout | 0.091 | 0.184 ± 0.004 | 0.182 | -0.025 | 0.037 | – | 0.209 | [+0.083, +0.293]; [+0.039, +0.276]; [+0.080, +0.313] | – | – | – | – | – | – | – | 150 | 3174 | yes | unfit (train acc 5.85%); rule: yes |
| transformer | study2 (exploratory) | 0 | pooled | value | ood | 0.112 | 0.117 ± 0.003 | 0.124 | -0.025 | 0.037 | – | 0.142 | [+0.080, +0.237]; [+0.060, +0.227]; [+0.091, +0.235] | – | – | – | – | – | – | – | 150 | 3171 | yes | unfit (train acc 5.85%); rule: yes |
| transformer | study2 (exploratory) | 0 | pooled | depth | ID_val_heldout | 0.825 | 0.832 ± 0.007 | 0.829 | 0.763 | 0.764 | 0.764 | 0.069 | [+0.070, +0.088]; [+0.054, +0.077]; [+0.051, +0.074] | [+0.070, +0.088]; [+0.052, +0.076]; [+0.051, +0.074] | 1.000 | 0.981 | [-0.167, -0.148]; [-0.184, -0.161]; [-0.185, -0.162] | – | – | 1.000 | 150 | 3174 | yes | unfit (train acc 5.85%); rule: yes, not above input baseline |
| transformer | study2 (exploratory) | 0 | pooled | depth | ood | 0.255 | 0.240 ± 0.003 | 0.245 | 0.310 | 0.764 | 0.382 | -0.070 | [-0.079, -0.063]; [-0.075, -0.059]; [-0.081, -0.065] | [-0.152, -0.135]; [-0.146, -0.130]; [-0.153, -0.137] | 0.278 | 0.277 | [-0.048, -0.033]; [-0.043, -0.027]; [-0.049, -0.034] | – | – | 0.618 | 150 | 3171 | no | unfit (train acc 5.85%); rule: no |
| transformer | study2 (exploratory) | 1 | pooled | value | ID_val_heldout | 0.044 | 0.069 ± 0.010 | 0.072 | -0.025 | 0.049 | – | 0.094 | [-0.142, +0.195]; [-0.038, +0.134]; [-0.039, +0.184] | – | – | – | – | – | – | – | 150 | 3174 | no | no |
| transformer | study2 (exploratory) | 1 | pooled | value | ood | 0.086 | 0.087 ± 0.013 | 0.102 | -0.023 | 0.049 | – | 0.110 | [+0.074, +0.203]; [+0.035, +0.139]; [+0.077, +0.201] | – | – | – | – | – | – | – | 150 | 3171 | no | no |
| transformer | study2 (exploratory) | 1 | pooled | depth | ID_val_heldout | 0.767 | 0.798 ± 0.003 | 0.794 | 0.763 | 0.765 | 0.764 | 0.035 | [+0.021, +0.040]; [+0.030, +0.048]; [+0.029, +0.044] | [+0.021, +0.040]; [+0.028, +0.045]; [+0.029, +0.044] | 1.000 | 0.981 | [-0.216, -0.195]; [-0.209, -0.191]; [-0.208, -0.192] | – | – | 1.000 | 150 | 3174 | no | no |
| transformer | study2 (exploratory) | 1 | pooled | depth | ood | 0.282 | 0.260 ± 0.005 | 0.262 | 0.309 | 0.765 | 0.382 | -0.049 | [-0.065, -0.048]; [-0.054, -0.039]; [-0.052, -0.038] | [-0.139, -0.118]; [-0.128, -0.111]; [-0.126, -0.108] | 0.278 | 0.277 | [-0.034, -0.016]; [-0.024, -0.008]; [-0.022, -0.007] | – | – | 0.618 | 150 | 3171 | no | no |
| transformer | study2 (exploratory) | 2 | pooled | value | ID_val_heldout | 0.080 | 0.150 ± 0.009 | 0.143 | -0.036 | 0.050 | – | 0.186 | [+0.036, +0.268]; [-0.022, +0.249]; [+0.052, +0.264] | – | – | – | – | – | – | – | 150 | 3174 | signal, below threshold | unfit (train acc 14.51%); rule: signal, below threshold |
| transformer | study2 (exploratory) | 2 | pooled | value | ood | 0.108 | 0.104 ± 0.001 | 0.107 | -0.034 | 0.050 | – | 0.138 | [+0.083, +0.239]; [+0.074, +0.219]; [+0.080, +0.228] | – | – | – | – | – | – | – | 150 | 3171 | yes | unfit (train acc 14.51%); rule: yes |
| transformer | study2 (exploratory) | 2 | pooled | depth | ID_val_heldout | 0.790 | 0.814 ± 0.007 | 0.817 | 0.759 | 0.763 | 0.764 | 0.055 | [+0.048, +0.067]; [+0.031, +0.053]; [+0.053, +0.077] | [+0.048, +0.067]; [+0.030, +0.053]; [+0.041, +0.063] | 1.000 | 0.981 | [-0.188, -0.169]; [-0.206, -0.184]; [-0.196, -0.174] | – | – | 1.000 | 150 | 3174 | no | unfit (train acc 14.51%); rule: no |
| transformer | study2 (exploratory) | 2 | pooled | depth | ood | 0.281 | 0.256 ± 0.002 | 0.261 | 0.309 | 0.763 | 0.382 | -0.053 | [-0.060, -0.045]; [-0.060, -0.045]; [-0.063, -0.046] | [-0.133, -0.116]; [-0.135, -0.116]; [-0.139, -0.119] | 0.278 | 0.277 | [-0.028, -0.014]; [-0.030, -0.014]; [-0.033, -0.017] | – | – | 0.618 | 150 | 3171 | no | unfit (train acc 14.51%); rule: no |

## Capacity sweep (Study 1 seed 0)

| model | state | target | split | h=64 | h=256 | h=1024 |
|---|---|---|---|---|---|---|
| lstm | h | value | ID_val_heldout | -0.061 (no) | -0.028 (no) | -0.051 (no) |
| lstm | h | value | ops4 | -0.015 (no) | -0.016 (no) | -0.010 (no) |
| lstm | h | value | ops5 | -0.030 (no) | -0.029 (no) | -0.018 (no) |
| lstm | h | value | ops6 | -0.023 (no) | -0.023 (no) | -0.017 (no) |
| lstm | h | value | ops7 | -0.021 (no) | -0.019 (no) | -0.015 (no) |
| lstm | h | depth | ID_val_heldout | 0.813 (yes) | 0.810 (yes) | 0.801 (yes) |
| lstm | h | depth | ops4 | 0.429 (no) | 0.438 (no) | 0.426 (no) |
| lstm | h | depth | ops5 | 0.242 (no) | 0.246 (no) | 0.248 (no) |
| lstm | h | depth | ops6 | 0.149 (no) | 0.141 (no) | 0.147 (no) |
| lstm | h | depth | ops7 | 0.130 (no) | 0.123 (no) | 0.128 (no) |
| lstm | c | value | ID_val_heldout | -0.013 (no) | 0.002 (no) | -0.168 (no) |
| lstm | c | value | ops4 | -0.019 (no) | -0.018 (no) | -0.027 (no) |
| lstm | c | value | ops5 | -0.034 (no) | -0.032 (no) | -0.035 (no) |
| lstm | c | value | ops6 | -0.028 (no) | -0.025 (no) | -0.027 (no) |
| lstm | c | value | ops7 | -0.020 (no) | -0.019 (no) | -0.025 (no) |
| lstm | c | depth | ID_val_heldout | 0.807 (yes) | 0.801 (yes) | 0.801 (yes) |
| lstm | c | depth | ops4 | 0.450 (no) | 0.449 (no) | 0.459 (no) |
| lstm | c | depth | ops5 | 0.249 (no) | 0.244 (no) | 0.256 (no) |
| lstm | c | depth | ops6 | 0.153 (no) | 0.150 (no) | 0.152 (no) |
| lstm | c | depth | ops7 | 0.129 (no) | 0.126 (no) | 0.128 (no) |
| transformer | pooled | value | ID_val_heldout | -0.176 (no) | -0.236 (no) | -0.220 (no) |
| transformer | pooled | value | ops4 | 0.005 (no) | 0.001 (no) | 0.007 (no) |
| transformer | pooled | value | ops5 | -0.004 (no) | -0.005 (no) | -0.001 (no) |
| transformer | pooled | value | ops6 | -0.018 (no) | -0.017 (no) | -0.014 (no) |
| transformer | pooled | value | ops7 | -0.020 (no) | -0.021 (no) | -0.019 (no) |
| transformer | pooled | depth | ID_val_heldout | 0.596 (no) | 0.587 (no) | 0.569 (no) |
| transformer | pooled | depth | ops4 | 0.376 (no) | 0.372 (no) | 0.365 (no) |
| transformer | pooled | depth | ops5 | 0.270 (no) | 0.269 (no) | 0.258 (no) |
| transformer | pooled | depth | ops6 | 0.207 (no) | 0.205 (no) | 0.202 (no) |
| transformer | pooled | depth | ops7 | 0.151 (no) | 0.151 (no) | 0.147 (no) |

## STEP 4f: final-position final-answer probes

**Pooling.** analysis_v2.ipynb cell 32: state at the final real position (len(expr) - 1) only; the Transformer state is transformer_encoder_states_mean_pooled, i.e. the mean over the 3 encoder layers at each position (pooling over layers, not over positions), unmasked; Ridge(alpha=1.0) on raw features; target = int(output). The restatement is identical, except the Transformer layer outputs are computed with the source key-padding mask (transformer_encoder_states_mean_pooled_masked). Run on cpu (torch 2.8.0, sklearn 1.6.1).

### Matched protocol, Transformer seed 0 (Ridge R²)

Unmasked rerun vs stored: max |diff| 2.38e-06. Train R²: unmasked 0.636, masked 0.894.

| split | n | stored (unmasked) | unmasked rerun | masked (restated) |
|---|---|---|---|---|
| ID_val_heldout | 150 | 0.050 | 0.050 | -0.495 |
| ops4 | 150 | 0.012 | 0.012 | -0.066 |
| ops5 | 150 | -0.058 | -0.058 | -0.050 |
| ops6 | 150 | -0.086 | -0.086 | -0.016 |
| ops7 | 150 | -0.132 | -0.132 | -0.057 |

### Stratified split (strat0), final-position final answer (R²)

n = 300 train / 150 test single vectors. Template control, one-hot(length): Ridge 0.011 (train 0.050), MLP 0.005 ± 0.001 (train 0.048, 0.048, 0.047), control -0.009. Held-out R² below -0.5 is flagged as small-n instability.

| row | Ridge (train) | MLP mean ± std | MLP train (s0, s1, s2) | control (train) | boot CI MLP − control | boot CI MLP − length baseline | rule verdict | flags |
|---|---|---|---|---|---|---|---|---|
| lstm_study1_seed0_h_final | 0.020 (0.087) | -0.006 ± 0.000 | 0.120, 0.196, 0.181 | -0.100 (-0.027, 0.001, 0.061) | [-0.198, +0.298]; [+0.005, +0.337]; [-0.040, +0.168] | [-0.267, +0.093]; [-0.220, +0.075]; [-0.205, +0.077] | no | – |
| lstm_study1_seed0_c_final | 0.026 (0.176) | -0.008 ± 0.006 | 0.054, 0.176, 0.128 | -0.122 (-0.041, -0.069, 0.075) | [-0.014, +0.251]; [+0.026, +0.645]; [-0.058, +0.177] | [-0.136, +0.040]; [-0.147, +0.054]; [-0.121, +0.057] | no | – |
| lstm_study1_seed1_h_final | 0.004 (0.131) | 0.004 ± 0.036 | 0.157, -0.109, 0.231 | -0.162 (-0.207, 0.029, 0.033) | [+0.194, +0.641]; [-0.100, +0.232]; [-0.289, +0.264] | [-0.128, +0.130]; [-0.413, +0.075]; [-0.356, +0.113] | no | – |
| lstm_study1_seed1_c_final | 0.021 (0.242) | -0.027 ± 0.060 | 0.261, -0.269, 0.224 | -0.191 (-0.164, 0.130, 0.027) | [+0.168, +0.627]; [-0.287, +0.190]; [-0.003, +0.363] | [-0.135, +0.103]; [-0.555, +0.001]; [-0.163, +0.102] | no | – |
| lstm_study1_seed2_h_final | 0.066 (0.140) | 0.080 ± 0.023 | 0.161, 0.233, 0.167 | -0.064 (0.097, 0.040, 0.134) | [-0.048, +0.222]; [+0.009, +0.334]; [+0.030, +0.265] | [-0.123, +0.202]; [-0.321, +0.211]; [-0.069, +0.207] | no | – |
| lstm_study1_seed2_c_final | 0.086 (0.218) | 0.060 ± 0.022 | 0.133, 0.141, 0.223 | -0.199 (-0.070, -0.034, -0.177) | [-0.005, +0.441]; [-0.027, +0.441]; [-0.049, +0.581] | [-0.128, +0.182]; [-0.121, +0.135]; [-0.147, +0.143] | no | – |
| transformer_study1_seed0_pooled_final | -0.004 (0.862) | 0.132 ± 0.028 | 0.997, 0.919, 0.916 | -0.142 (0.102, 0.314, 0.292) | [-0.775, +0.537]; [-0.345, +0.597]; [-0.405, +0.564] | [-0.941, +0.464]; [-0.722, +0.458]; [-0.675, +0.396] | signal, below threshold | – |
| transformer_study1_seed1_pooled_final | -0.206 (0.800) | -0.074 ± 0.031 | 0.843, 0.876, 0.824 | -0.139 (0.110, 0.117, 0.132) | [-0.881, +0.285]; [-0.885, +0.358]; [-0.626, +0.405] | [-1.025, +0.205]; [-1.054, +0.245]; [-0.899, +0.233] | signal, below threshold | – |
| transformer_study1_seed2_pooled_final | 0.068 (0.792) | 0.211 ± 0.022 | 0.620, 0.701, 0.624 | -0.119 (0.100, 0.178, 0.119) | [-0.315, +0.643]; [-0.166, +0.581]; [-0.149, +0.606] | [-0.449, +0.441]; [-0.266, +0.474]; [-0.341, +0.429] | signal, below threshold | – |
