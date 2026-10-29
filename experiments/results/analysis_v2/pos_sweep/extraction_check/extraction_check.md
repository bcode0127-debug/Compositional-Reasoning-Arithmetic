Reconciliation check; not used by any rule.

# Extraction path x device check, sinusoidal layer-3 linear position

Descriptive only; no reading beyond the differences. Same paper checkpoints (Hub path), same stratified split, same position probe (LogisticRegression, max_iter 1000, unscaled).

## Layer-3 position accuracy per seed

| Seed | Probe grid (fused, L4 GPU, committed) | Battery (hooked, pod CPU, committed) | Mac hooked CPU | Mac fused CPU |
|---|---|---|---|---|
| 0 | 0.1936 | 0.1936 | 0.1936 | 0.1943 |
| 1 | 0.1775 | 0.1779 | 0.1779 | 0.1765 |
| 2 | 0.1810 | 0.1821 | 0.1803 | 0.1807 |

## Differences

| Seed | Path at fixed device (Mac hooked - Mac fused) | Architecture for hooked (battery - Mac hooked) | Device and environment for fused (probe grid - Mac fused) |
|---|---|---|---|
| 0 | -0.0007 | +0.0000 | -0.0007 |
| 1 | +0.0014 | +0.0000 | +0.0011 |
| 2 | -0.0004 | +0.0018 | +0.0004 |

## Layer-3 activation max abs difference between extractions (Mac CPU)

The Deviation entry's bound for the path difference is 5.4e-04.

| Seed | Hooked vs legacy (all rows) | Hooked vs legacy (test rows) | Legacy vs fused layer | Hooked vs fused layer |
|---|---|---|---|---|
| 0 | 4.71e-05 | 3.34e-05 | 0.00e+00 | 4.71e-05 |
| 1 | 7.09e-05 | 4.99e-05 | 0.00e+00 | 7.09e-05 |
| 2 | 1.98e-05 | 1.85e-05 | 0.00e+00 | 1.98e-05 |

## Which fused path

Mac fused CPU is path (ii): nonlinear_probe.transformer_layerwise_masked via LEGACY_EXTRACTION = True (the extraction of the committed probe grid at 185b127; the current code's switch for it).

Supplementary, not in the table above: path (iii), each encoder layer called without hooks: torch._transformer_encoder_layer_fwd. Its layer-3 position accuracy per seed: seed 0 0.1943, seed 1 0.1765, seed 2 0.1807.

Kernel calls per path (the same expressions, counters on the torch kernels; seed 0):

| Path | torch._native_multi_head_attention | torch._transformer_encoder_layer_fwd |
|---|---|---|
| hooked | 9 | 0 |
| legacy | 9 | 0 |
| fused_layer | 0 | 9 |

3 expressions through 3 encoder layers each, so one call per expression and layer where a kernel is used.
