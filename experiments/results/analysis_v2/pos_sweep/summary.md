Design: `docs/design/pos-encoding-sweep.md` (pre-registration 9365283).
Deviations entries applied: `2026-10-19: Specifications added before the first run`; `2026-10-19: Probe extraction path`; `2026-10-28: NoPE floor failed; the position reference is replaced by an order-blind baseline`.

# Positional-encoding sweep: probe battery

Run: 2026-10-29T09:29:42.673407+00:00 on NVIDIA RTX 4000 Ada Generation pod image `RunPod PyTorch template 1.5.0-2.8.0 (PyTorch 2.8.0, CUDA 12.8.1, Ubuntu 24.04)`, torch 2.8.0+cu128, scikit-learn 1.9.1, probes on cpu, test re-evaluation on cuda, code commit `89f713a9bde720dbe6568411eff8c784495a5865`.

The committed paper numbers (experiments/results/analysis_v2/nonlinear_probe/, results_v2/test_set_results.json) are unchanged by this run.

## ANCHOR (Deviation 2026-10-28)

| Baseline | Position accuracy |
|---|---|
| B1 | 0.1621 |
| B2 | 0.1894 |
| B3 | 0.1880 |

ANCHOR = B2 = 0.1894 (the highest of the three, chosen mechanically). Majority 0.0525. The 0.053 appears only in the "as pre-registered" columns.

## Decision rules

P2 specification note: the design names the P2 and trigger statistic (min over the re-injecting arms minus max over {sinusoidal, learned}) but not how min and max meet the resampling. This run takes min and max inside each resample (same 150 expressions drawn for every arm). That specifies what the design left open; it is not a quotation of the design.

- **a) Floor (check):** equivariance max diff 2.38e-06 (pass True); NoPE layer-3 0.1443 <= ANCHOR + 0.02 = 0.2094: True. **PASS.** As pre-registered (0.053 +/- 0.02): fail (delta +0.0913).

### Under ANCHOR

- **b) Injection, rope, layer 1:** lower bounds [-0.0039, 0.0007, -0.0105] -> not injected
- **b) Injection, alibi, layer 1:** lower bounds [-0.0391, -0.0425, -0.0444] -> not injected
- **c) Survives, sinusoidal, layer 3:** lower bounds [-0.0103, -0.0259, -0.0198] -> False
- **c) Survives, learned, layer 3:** lower bounds [0.0308, -0.0059, -0.0007] -> False
- **c) Survives, rope, layer 3:** lower bounds [0.29, 0.2803, 0.2945] -> True
- **c) Survives, alibi, layer 3:** lower bounds [0.213, 0.2026, 0.1763] -> True
- **e/f) P2 and trigger:** not evaluated (no re-injecting arm injected).
- **i) Kill gate:** rope: position survives and the ID gap moves; alibi: position survives and the ID gap moves

### Under as pre-registered (0.053)

- **b) Injection, rope, layer 1:** lower bounds [0.1308, 0.1371, 0.1246] -> injected
- **b) Injection, alibi, layer 1:** lower bounds [0.0979, 0.0932, 0.0906] -> not injected
- **c) Survives, sinusoidal, layer 3:** lower bounds [0.1245, 0.109, 0.1156] -> True
- **c) Survives, learned, layer 3:** lower bounds [0.1657, 0.1277, 0.134] -> True
- **c) Survives, rope, layer 3:** lower bounds [0.4223, 0.4123, 0.4279] -> True
- **c) Survives, alibi, layer 3:** lower bounds [0.3469, 0.3383, 0.3096] -> True
- **e) P2** (re-injecting arms used: ['rope']): per-seed CIs ['[+0.2461, +0.2923]', '[+0.2732, +0.3225]', '[+0.2825, +0.3291]'] -> True
- **f) Trigger** (lower bound > +0.1 on all seeds): True
- **i) Kill gate:** sinusoidal: position survives (the reference arm for the ID gap); learned: position survives but the ID gap does not move: the mechanism is encoding-specific and the behavioral gap is not; rope: position survives and the ID gap moves; alibi: position survives and the ID gap moves

### Reference-free rules

- **d) P1:** learned - sinusoidal layer-3 CIs ['[+0.0243, +0.0564]', '[+0.0048, +0.0337]', '[+0.0064, +0.0340]']; includes zero on 0 of 3 seeds -> False
- **g) ID gap, learned:** diff (points) [-0.9, -0.3, -1.9], CIs ['[-0.0240, +0.0060]', '[-0.0180, +0.0110]', '[-0.0340, -0.0040]'] -> gap moves False
- **g) ID gap, rope:** diff (points) [13.5, 10.3, 13.6], CIs ['[+0.1070, +0.1650]', '[+0.0750, +0.1300]', '[+0.1070, +0.1650]'] -> gap moves True (up)
- **g) ID gap, alibi:** diff (points) [2.5, 4.9, 5.2], CIs ['[+0.0040, +0.0450]', '[+0.0250, +0.0720]', '[+0.0300, +0.0730]'] -> gap moves True (up)
- **h) Depth code at memory, sinusoidal:** per seed [False, False, False] -> False
- **h) Depth code at memory, learned:** per seed [False, False, False] -> False
- **h) Depth code at memory, rope:** per seed [True, True, True] -> True
- **h) Depth code at memory, alibi:** per seed [False, False, False] -> False

## Per-seed position tables (accuracy; vs ANCHOR diff [95% CI]; vs 0.053 diff)

### sinusoidal, encoder_input (construction check only)

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.9947 | +0.8053 | [+0.7946, +0.8148] | +0.9417 | [+0.9386, +0.9445] |
| 1 | 0.9951 | +0.8057 | [+0.7962, +0.8159] | +0.9421 | [+0.9391, +0.9446] |
| 2 | 0.9951 | +0.8057 | [+0.7956, +0.8155] | +0.9421 | [+0.9392, +0.9446] |

### sinusoidal, layer1

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.5634 | +0.3739 | [+0.3504, +0.3990] | +0.5104 | [+0.4838, +0.5368] |
| 1 | 0.6092 | +0.4198 | [+0.3954, +0.4458] | +0.5562 | [+0.5301, +0.5842] |
| 2 | 0.6299 | +0.4405 | [+0.4132, +0.4699] | +0.5769 | [+0.5474, +0.6076] |

### sinusoidal, layer2

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.2377 | +0.0483 | [+0.0318, +0.0643] | +0.1847 | [+0.1676, +0.2018] |
| 1 | 0.2388 | +0.0494 | [+0.0340, +0.0636] | +0.1858 | [+0.1694, +0.2021] |
| 2 | 0.2626 | +0.0732 | [+0.0560, +0.0899] | +0.2096 | [+0.1911, +0.2277] |

### sinusoidal, layer3

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1936 | +0.0042 | [-0.0103, +0.0187] | +0.1406 | [+0.1245, +0.1567] |
| 1 | 0.1779 | -0.0116 | [-0.0259, +0.0042] | +0.1249 | [+0.1090, +0.1406] |
| 2 | 0.1821 | -0.0074 | [-0.0198, +0.0050] | +0.1291 | [+0.1156, +0.1427] |

### sinusoidal, mean_pool

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.2482 | +0.0588 | [+0.0414, +0.0760] | +0.1952 | [+0.1770, +0.2142] |
| 1 | 0.2426 | +0.0532 | [+0.0375, +0.0698] | +0.1896 | [+0.1717, +0.2080] |
| 2 | 0.2535 | +0.0641 | [+0.0467, +0.0808] | +0.2005 | [+0.1816, +0.2179] |

### sinusoidal, memory

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1950 | +0.0056 | [-0.0088, +0.0199] | +0.1420 | [+0.1259, +0.1579] |
| 1 | 0.1768 | -0.0126 | [-0.0266, +0.0022] | +0.1238 | [+0.1081, +0.1397] |
| 2 | 0.1821 | -0.0074 | [-0.0205, +0.0057] | +0.1291 | [+0.1149, +0.1432] |

### learned, encoder_input (construction check only)

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.9954 | +0.8060 | [+0.7952, +0.8157] | +0.9424 | [+0.9394, +0.9449] |
| 1 | 0.9958 | +0.8064 | [+0.7967, +0.8165] | +0.9428 | [+0.9400, +0.9452] |
| 2 | 0.9958 | +0.8064 | [+0.7963, +0.8163] | +0.9428 | [+0.9401, +0.9453] |

### learned, layer1

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.8655 | +0.6761 | [+0.6619, +0.6900] | +0.8125 | [+0.7978, +0.8263] |
| 1 | 0.8683 | +0.6789 | [+0.6630, +0.6939] | +0.8153 | [+0.8002, +0.8302] |
| 2 | 0.8676 | +0.6782 | [+0.6636, +0.6933] | +0.8146 | [+0.8006, +0.8288] |

### learned, layer2

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.4433 | +0.2539 | [+0.2343, +0.2734] | +0.3903 | [+0.3690, +0.4113] |
| 1 | 0.4198 | +0.2304 | [+0.2094, +0.2525] | +0.3668 | [+0.3435, +0.3915] |
| 2 | 0.3904 | +0.2010 | [+0.1789, +0.2215] | +0.3374 | [+0.3136, +0.3614] |

### learned, layer3

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.2342 | +0.0448 | [+0.0308, +0.0586] | +0.1812 | [+0.1657, +0.1968] |
| 1 | 0.1975 | +0.0081 | [-0.0059, +0.0240] | +0.1445 | [+0.1277, +0.1614] |
| 2 | 0.2020 | +0.0126 | [-0.0007, +0.0269] | +0.1490 | [+0.1340, +0.1650] |

### learned, mean_pool

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.4433 | +0.2539 | [+0.2353, +0.2710] | +0.3903 | [+0.3693, +0.4103] |
| 1 | 0.3925 | +0.2031 | [+0.1819, +0.2231] | +0.3395 | [+0.3176, +0.3618] |
| 2 | 0.3764 | +0.1870 | [+0.1679, +0.2067] | +0.3234 | [+0.3022, +0.3459] |

### learned, memory

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.2356 | +0.0462 | [+0.0327, +0.0599] | +0.1826 | [+0.1666, +0.1986] |
| 1 | 0.1971 | +0.0077 | [-0.0066, +0.0233] | +0.1441 | [+0.1282, +0.1611] |
| 2 | 0.2017 | +0.0123 | [-0.0014, +0.0264] | +0.1487 | [+0.1335, +0.1652] |

### rope, encoder_input (construction check only)

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1618 | -0.0277 | [-0.0343, -0.0210] | +0.1088 | [+0.0993, +0.1189] |
| 1 | 0.1618 | -0.0277 | [-0.0348, -0.0213] | +0.1088 | [+0.0991, +0.1184] |
| 2 | 0.1607 | -0.0287 | [-0.0345, -0.0230] | +0.1077 | [+0.0979, +0.1173] |

### rope, layer1

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1971 | +0.0077 | [-0.0039, +0.0198] | +0.1441 | [+0.1308, +0.1580] |
| 1 | 0.2048 | +0.0154 | [+0.0007, +0.0302] | +0.1518 | [+0.1371, +0.1672] |
| 2 | 0.1922 | +0.0028 | [-0.0105, +0.0163] | +0.1392 | [+0.1246, +0.1558] |

### rope, layer2

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.5508 | +0.3613 | [+0.3375, +0.3858] | +0.4978 | [+0.4682, +0.5265] |
| 1 | 0.4891 | +0.2997 | [+0.2708, +0.3284] | +0.4361 | [+0.4038, +0.4680] |
| 2 | 0.4947 | +0.3053 | [+0.2764, +0.3324] | +0.4417 | [+0.4100, +0.4730] |

### rope, layer3

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.5032 | +0.3137 | [+0.2900, +0.3379] | +0.4502 | [+0.4223, +0.4758] |
| 1 | 0.4961 | +0.3067 | [+0.2803, +0.3324] | +0.4431 | [+0.4123, +0.4731] |
| 2 | 0.5088 | +0.3193 | [+0.2945, +0.3426] | +0.4558 | [+0.4279, +0.4842] |

### rope, mean_pool

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.5546 | +0.3652 | [+0.3389, +0.3903] | +0.5016 | [+0.4725, +0.5312] |
| 1 | 0.5287 | +0.3393 | [+0.3090, +0.3676] | +0.4757 | [+0.4407, +0.5075] |
| 2 | 0.5252 | +0.3358 | [+0.3092, +0.3626] | +0.4722 | [+0.4410, +0.5014] |

### rope, memory

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.5039 | +0.3144 | [+0.2906, +0.3382] | +0.4509 | [+0.4223, +0.4771] |
| 1 | 0.4968 | +0.3074 | [+0.2803, +0.3338] | +0.4438 | [+0.4128, +0.4741] |
| 2 | 0.5105 | +0.3211 | [+0.2959, +0.3449] | +0.4575 | [+0.4292, +0.4846] |

### alibi, encoder_input (construction check only)

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1621 | -0.0273 | [-0.0330, -0.0215] | +0.1091 | [+0.0984, +0.1188] |
| 1 | 0.1621 | -0.0273 | [-0.0334, -0.0217] | +0.1091 | [+0.0989, +0.1186] |
| 2 | 0.1604 | -0.0291 | [-0.0358, -0.0225] | +0.1074 | [+0.0979, +0.1162] |

### alibi, layer1

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1625 | -0.0270 | [-0.0391, -0.0153] | +0.1095 | [+0.0979, +0.1198] |
| 1 | 0.1569 | -0.0326 | [-0.0425, -0.0229] | +0.1039 | [+0.0932, +0.1159] |
| 2 | 0.1569 | -0.0326 | [-0.0444, -0.0207] | +0.1039 | [+0.0906, +0.1161] |

### alibi, layer2

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.4188 | +0.2293 | [+0.2005, +0.2593] | +0.3658 | [+0.3349, +0.3965] |
| 1 | 0.4132 | +0.2237 | [+0.1964, +0.2504] | +0.3602 | [+0.3302, +0.3900] |
| 2 | 0.4079 | +0.2185 | [+0.1926, +0.2415] | +0.3549 | [+0.3268, +0.3813] |

### alibi, layer3

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.4258 | +0.2363 | [+0.2130, +0.2584] | +0.3728 | [+0.3469, +0.3970] |
| 1 | 0.4139 | +0.2244 | [+0.2026, +0.2445] | +0.3609 | [+0.3383, +0.3826] |
| 2 | 0.3855 | +0.1961 | [+0.1763, +0.2156] | +0.3325 | [+0.3096, +0.3560] |

### alibi, mean_pool

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.4468 | +0.2574 | [+0.2316, +0.2823] | +0.3938 | [+0.3656, +0.4202] |
| 1 | 0.4422 | +0.2528 | [+0.2294, +0.2764] | +0.3892 | [+0.3625, +0.4144] |
| 2 | 0.4023 | +0.2129 | [+0.1897, +0.2364] | +0.3493 | [+0.3220, +0.3758] |

### alibi, memory

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.4275 | +0.2381 | [+0.2155, +0.2613] | +0.3745 | [+0.3488, +0.4000] |
| 1 | 0.4142 | +0.2248 | [+0.2037, +0.2442] | +0.3612 | [+0.3387, +0.3825] |
| 2 | 0.3894 | +0.1999 | [+0.1800, +0.2202] | +0.3364 | [+0.3124, +0.3609] |

### nope, encoder_input (construction check only)

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1632 | -0.0263 | [-0.0330, -0.0196] | +0.1102 | [+0.1005, +0.1203] |

### nope, layer1

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1502 | -0.0392 | [-0.0511, -0.0267] | +0.0972 | [+0.0859, +0.1089] |

### nope, layer2

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1513 | -0.0382 | [-0.0514, -0.0254] | +0.0983 | [+0.0844, +0.1111] |

### nope, layer3

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1443 | -0.0452 | [-0.0577, -0.0331] | +0.0913 | [+0.0791, +0.1035] |

### nope, mean_pool

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1520 | -0.0375 | [-0.0500, -0.0255] | +0.0990 | [+0.0865, +0.1117] |

### nope, memory

| Seed | Accuracy | vs ANCHOR | CI vs ANCHOR | vs 0.053 | CI vs 0.053 |
|---|---|---|---|---|---|
| 0 | 0.1446 | -0.0448 | [-0.0572, -0.0330] | +0.0916 | [+0.0790, +0.1042] |

## Depth at the memory

| Arm | Seed | Linear | MLP mean | Control | Majority | vs char+position baseline (lower bound) | Rule base | Depth code |
|---|---|---|---|---|---|---|---|---|
| sinusoidal | 0 | 0.506 | 0.529 | 0.518 | 0.527 | -0.235 | no | False |
| sinusoidal | 1 | 0.512 | 0.523 | 0.513 | 0.527 | -0.239 | no | False |
| sinusoidal | 2 | 0.503 | 0.520 | 0.511 | 0.527 | -0.247 | no | False |
| learned | 0 | 0.512 | 0.526 | 0.517 | 0.527 | -0.236 | no | False |
| learned | 1 | 0.517 | 0.527 | 0.515 | 0.527 | -0.233 | no | False |
| learned | 2 | 0.509 | 0.525 | 0.509 | 0.527 | -0.236 | no | False |
| rope | 0 | 0.708 | 0.802 | 0.509 | 0.527 | +0.031 | yes | True |
| rope | 1 | 0.727 | 0.776 | 0.508 | 0.527 | +0.008 | yes | True |
| rope | 2 | 0.698 | 0.779 | 0.510 | 0.527 | +0.005 | yes | True |
| alibi | 0 | 0.675 | 0.731 | 0.498 | 0.527 | -0.042 | yes | False |
| alibi | 1 | 0.697 | 0.742 | 0.517 | 0.527 | -0.030 | yes | False |
| alibi | 2 | 0.676 | 0.715 | 0.497 | 0.527 | -0.057 | yes | False |

## In-distribution test accuracy (Study 1, 1,000 examples, autoregressive greedy, exact match)

| Arm | Seed | Test accuracy (%) | Source |
|---|---|---|---|
| sinusoidal (re-evaluated, used by the rule) | 0 | 7.90 | this run, pod, cuda |
| sinusoidal (committed) | 0 | 7.90 | results_v2/test_set_results.json (torch 2.13.0, CPU); shown only, the ID-gap rule used the re-evaluated value |
| sinusoidal (re-evaluated, used by the rule) | 1 | 7.70 | this run, pod, cuda |
| sinusoidal (committed) | 1 | 7.70 | results_v2/test_set_results.json (torch 2.13.0, CPU); shown only, the ID-gap rule used the re-evaluated value |
| sinusoidal (re-evaluated, used by the rule) | 2 | 7.90 | this run, pod, cuda |
| sinusoidal (committed) | 2 | 7.90 | results_v2/test_set_results.json (torch 2.13.0, CPU); shown only, the ID-gap rule used the re-evaluated value |
| learned | 0 | 7.00 | run dir test_results.json (runner, cuda) |
| learned | 1 | 7.40 | run dir test_results.json (runner, cuda) |
| learned | 2 | 6.00 | run dir test_results.json (runner, cuda) |
| rope | 0 | 21.40 | run dir test_results.json (runner, cuda) |
| rope | 1 | 18.00 | run dir test_results.json (runner, cuda) |
| rope | 2 | 21.50 | run dir test_results.json (runner, cuda) |
| alibi | 0 | 10.40 | run dir test_results.json (runner, cuda) |
| alibi | 1 | 12.60 | run dir test_results.json (runner, cuda) |
| alibi | 2 | 13.10 | run dir test_results.json (runner, cuda) |

## For the paper sentence: sinusoidal layer-3 position minus ANCHOR

| Seed | Layer-3 accuracy | minus ANCHOR | 95% CI |
|---|---|---|---|
| 0 | 0.1936 | +0.0042 | [-0.0103, +0.0187] |
| 1 | 0.1779 | -0.0116 | [-0.0259, +0.0042] |
| 2 | 0.1821 | -0.0074 | [-0.0198, +0.0050] |
