# Positional-encoding sweep: pre-registered design

Status: pre-registration. This file is committed before any run; the summary of results records this file's commit hash. Changes after the first run go in a dated "Deviations" section at the end, never into the sections above it.

## Question

Does the positional-encoding choice cause the Transformer's encoder to lose order, and with it the in-distribution gap to the LSTM, or is the loss architectural?

## Facts the sweep starts from

From the paper (code repo 185b127, 2c5e91d, 35412d2; stratified split strat0, Study 1):
- Linear position decodability in the Transformer encoder falls from 0.995 at the encoder input to 0.178–0.194 at layer 3 and 0.177–0.195 in the memory, against a majority of 0.053 (three seeds).
- Linear depth decodability falls from 0.662 at the input to 0.502–0.513 at layer 3, against a majority of 0.527.
- Character identity stays at or above 0.999 everywhere.
- In-distribution test accuracy: LSTM 34.57%, Transformer 7.83% (sinusoidal encoding).

Construction facts (models/transformer.py, train_v2.py):
- The encoder builds no causal mask; `generate_square_subsequent_mask` is used by the decoder only. The encoder takes a key padding mask only, so among real tokens its self-attention stack is permutation-equivariant whenever no position signal is present.
- The input path is `encoder_embedding * sqrt(d_model)` followed by `pos_encoder` (sinusoidal, added to the embedding).

Literature the design rests on:
- Position decodability decays with depth in bidirectional encoders that inject position once at the input: BERT-base, learned absolute embeddings, near-perfect at layers 1–3, sharp drop from layer 4 (Lin, Tan and Frank, 2019).
- Without positional encodings, position arises in causal models through the causal mask (variance mechanism, Chi et al., 2023; 1/t construction, Kazemnejad et al., 2023, Theorem 1; empirical, Haviv et al., 2022). In a bidirectional model the variance is uniform across positions and the signal does not arise (Chi et al., 2023).
- Transformers fail regular and counting tasks regardless of positional encoding (none, sin/cos, RoPE, ALiBi, Transformer-XL relative) where LSTMs succeed (Delétang et al., 2023).
- Among explicit schemes in causal decoders, length generalization ranks NoPE > T5 relative > ALiBi > RoPE ≈ sinusoidal (Kazemnejad et al., 2023).
- ALiBi in a bidirectional encoder uses a symmetric distance penalty on |i − j| (MosaicBERT, Portes et al., 2023; Jina Embeddings 2).

## Arms

Axis: where position is injected. Once at the input (sinusoidal, learned absolute) versus re-injected at every attention layer (RoPE, ALiBi).

| Arm | Injection | Encoder | Decoder self-attention | Parameters vs paper |
|---|---|---|---|---|
| sinusoidal | once, input | as in the paper | as in the paper | identical; the paper's existing checkpoints, not retrained |
| learned absolute | once, input | learned table, max length 64 | learned table | +64 × 256 per table |
| RoPE | every layer | rotation on q, k in each encoder layer | same, causal | no added parameters; sinusoidal buffer removed |
| ALiBi, symmetric | every layer | bias −m·|i − j| per head, slopes geometric as in Press et al. | causal ALiBi as in Press et al. | no added parameters; sinusoidal buffer removed |

Cross-attention is unchanged in every arm. Encoder and decoder use the same scheme so that the behavioral endpoint reflects one encoding, not a mixture.

Floor check, not an arm: NoPE in the encoder only (decoder keeps sinusoidal), one seed. Its layer-3 position probe must read within ±0.02 of the 0.053 majority by construction (permutation equivariance), and a unit test asserts the equivariance before training. Reported as a check; a reading outside ±0.02 stops the sweep for a bug hunt.

Conditional fifth arm: T5-style learned relative bias (re-injecting). Runs only if the trigger in "Decision rules" fires. Pre-registered here so that running it is not a choice made after seeing results.

## Fixed across arms

The paper's Transformer configuration and training protocol, unchanged: 3 encoder and 3 decoder layers, 8 heads, d_model 256, feed-forward 1024, dropout as in train_v2.py; peak learning rate 3e-4 with a 400-step linear warmup and cosine decay to 1e-5, batch size 64, up to 200 epochs with early-stopping patience 50 (scripts/training/run_matrix.sh: --epochs 200 --patience 50; train_v2.py:36-43), checkpoint selected on validation accuracy; Study 1 datasets_v2 splits; seeds 0, 1, 2. Only the positional scheme varies. Training-environment versions are recorded this time (torch, numpy, CUDA, GPU), per run, in each run's history.json.

## Endpoints

1. Primary: linear position decodability at encoder layer 3, stratified split strat0, same probe protocol as the paper's identity probes (unscaled logistic regression, position index as class, majority 0.053).
2. Secondary: in-distribution test accuracy on the independent Study 1 test set, autoregressive exact match, against the LSTM's 34.57% and the sinusoidal Transformer's 7.83%.
3. Third: per-position depth decodability at the memory (linear and MLP, the paper's protocol, majority 0.527, one-hot(char, position) baseline 0.664 / 0.732).

Rows comparable across arms: layers 1–3, mean pool, memory. The encoder-input row is a construction check reported per arm and excluded from cross-arm claims: it reads near 0.995 for once-injected arms and at majority for RoPE and ALiBi by construction. Layer 1 is the named injection check for RoPE and ALiBi. An arm counts as "injected" if, at layer 1, the 95% bootstrap lower bound on (linear position accuracy − 0.053) exceeds +0.10 on all three seeds; otherwise it is reported as "not injected", its layer-3 reading is excluded from P2 and from the fifth-arm trigger, and the failure to inject is the finding for that arm.

The OOD curve (ops4–7) is recorded but is not an endpoint: the Transformer sits at the constant-output floor on all four buckets, and a floor cannot register a change.

## Pre-registered predictions

- P1: learned absolute decays like sinusoidal. At layer 3, the paired bootstrap 95% CI on (learned − sinusoidal) position accuracy includes zero on at least two of three seeds.
- P2: re-injecting arms hold position above once-injected arms. At layer 3, the paired bootstrap 95% lower bound on (min over {RoPE, ALiBi} − max over {sinusoidal, learned}) is above zero on all three seeds.

## Decision rules (fixed before the first run)

- "Position survives" at layer 3 for an arm: the bootstrap 95% lower bound on (linear position accuracy − 0.053) exceeds +0.10 on all three seeds. Example-level bootstrap, 1,000 resamples, over the 150 stratified test expressions. The lower bound, not the point estimate, carries the +0.10, the same shape as the fifth-arm trigger.
- "ID gap moves" for an arm: the example-level paired bootstrap 95% CI on (arm − sinusoidal) test accuracy, seed-matched, excludes zero in the same direction on all three seeds, on the 1,000 independent test examples.
- "Depth code appears" at the memory: the paper's depth verdict rule (MLP mean minus std above control + 0.05 and above majority; bootstrap vs control and vs majority lower bounds above zero; above the one-hot(char, position) baseline's MLP 0.732 with lower bound above zero), all three seeds.
- Fifth-arm trigger: T5-relative runs only if, at layer 3, the paired bootstrap 95% lower bound on (min over re-injecting arms − max over once-injected arms) exceeds +0.10 absolute position accuracy on all three seeds, same probe and baseline.
- Kill gate, stated as a result: if no arm satisfies "position survives", the loss is architectural rather than encoding-specific, and that is the sweep's finding. If an arm satisfies "position survives" and "ID gap moves" does not fire for it, the mechanism is encoding-specific and the behavioral gap is not.

## Compute

9 new training runs (learned, RoPE, ALiBi × 3 seeds) plus 1 NoPE floor run; the sinusoidal arm reuses the paper's checkpoints. Each run is a 5.5M-parameter model for up to 200 epochs on Study 1; the paper's Transformer checkpoints were selected at epochs 114, 160 and 160, so most runs will use a large share of the budget. Per-run wall time is unmeasured on an L4: the first run is timed before the rest launch, and the pod budget is set from it, with 1.5 hours per run as the planning ceiling. Record actual wall time per run.

## Outputs

- results_v2/pos_sweep/<arm>/study1/seed{k}/: history.json (with environment), evaluation_results.json, best_model.pt fingerprinted as the paper's checkpoints are (epoch, val_loss, val_accuracy, seed, dataset_version, generator_commit, plus `positional_encoding`), uploaded to the Hugging Face model repo under pos_sweep/.
- experiments/results/analysis_v2/pos_sweep/: per-arm probe JSONs on the paper's schema, summary.md with the decision rules and this file's commit hash printed at the top, environment.json.
- CHECKPOINTS.md gains one row per new checkpoint.

## Tests (written before the runs)

- Encoding units, float32, all with explicit tolerances: RoPE preserves vector norms (max abs difference ≤ 1e-5) and the relative-position property (for random q, k, the rotated dot product at positions (i, j) equals the one at (i + s, j + s) for every shift s, max abs difference ≤ 1e-4); the ALiBi encoder bias matrix equals −m_h·|i − j| with slopes m_h = 2^(−8h/H) for H = 8 heads (max abs difference ≤ 1e-6), and is symmetric; the NoPE encoder is permutation-equivariant on real tokens (permute inputs, outputs permute identically, max abs difference ≤ 1e-5); the sinusoidal path with the refactored script loads the paper's existing checkpoints with identical state-dict keys and reproduces their stored validation accuracy within 0.1 points.
- Probe reproduction gate: the probe battery on the paper's sinusoidal seed-0 checkpoint reproduces the committed layer-3 position (0.194) and depth (0.506) values within ±0.02 before any new arm is probed.
- Schema test on every new JSON.

## Deviations

(empty at pre-registration)

## References

Chi, T.-C., et al. (2023). Latent positional information is in the self-attention variance of transformer language models without positional embeddings. arXiv:2305.13571.
Delétang, G., et al. (2023). Neural networks and the Chomsky hierarchy. ICLR 2023. arXiv:2207.02098.
Haviv, A., et al. (2022). Transformer language models without positional encodings still learn positional information. Findings of EMNLP 2022. arXiv:2203.16634.
Kazemnejad, A., et al. (2023). The impact of positional encoding on length generalization in Transformers. NeurIPS 2023. arXiv:2305.19466.
Lin, Y., Tan, Y. C., and Frank, R. (2019). Open Sesame: getting inside BERT's linguistic knowledge. BlackboxNLP 2019. arXiv:1906.01698.
McLeish, S., et al. (2024). Transformers can do arithmetic with the right embeddings. NeurIPS 2024. arXiv:2405.17399.
Portes, J., et al. (2023). MosaicBERT: a bidirectional encoder optimized for fast pretraining. NeurIPS 2023. arXiv:2312.17482.
Press, O., Smith, N. A., and Lewis, M. (2022). Train short, test long: attention with linear biases enables input length extrapolation. ICLR 2022.
Su, J., et al. (2024). RoFormer: enhanced transformer with rotary position embedding. Neurocomputing.
