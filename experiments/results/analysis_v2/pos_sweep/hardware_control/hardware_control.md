# Sinusoidal hardware control

Hardware control for the sinusoidal arm: not used by any rule. The pre-registered comparison against the paper's checkpoints stays primary; nothing here is recomputed into a verdict.

Every row below is not used by any rule.

## Per seed (paper checkpoint, retrain, hardware delta = retrain minus paper)

| Seed | Selected epoch (paper / retrain) | Test accuracy % (paper / retrain / delta) | Layer-3 position (paper / retrain / delta) | Note |
|---|---|---|---|---|
| 0 | 114 / 165 | 7.90 / 7.60 / -0.30 | 0.1936 / 0.1793 / -0.0144 | not used by any rule |
| 1 | 160 / 110 | 7.70 / 7.40 / -0.30 | 0.1779 / 0.1681 / -0.0098 | not used by any rule |
| 2 | 160 / 187 | 7.90 / 7.70 / -0.20 | 0.1821 / 0.1775 / -0.0046 | not used by any rule |

## Descriptive: learned minus retrained sinusoidal, layer-3 position (seed-matched, paired bootstrap by expression, 1000 resamples)

| Seed | Learned | Retrained sinusoidal | Difference | 95% interval | Note |
|---|---|---|---|---|---|
| 0 | 0.2342 | 0.1793 | +0.0550 | [+0.0397, +0.0711] | not used by any rule |
| 1 | 0.1975 | 0.1681 | +0.0294 | [+0.0156, +0.0451] | not used by any rule |
| 2 | 0.2020 | 0.1775 | +0.0245 | [+0.0096, +0.0390] | not used by any rule |

## Environments

| Checkpoints | GPU | Environment |
|---|---|---|
| paper checkpoints | unrecorded | unrecorded |
| probe grid | L4 | torch 2.8.0+cu128, scikit-learn 1.9.1 |
| arms and NoPE | RTX 4000 Ada | torch 2.8.0+cu128, numpy 2.5.2, driver 595.91.07 |
| sinusoidal retrain | NVIDIA RTX 4000 Ada Generation | torch 2.8.0+cu128, numpy 2.5.2, python 3.12.3, driver 595.91.07, commit `2030fd54419aa85dbdd86727b5829496d03c9ab2` |
