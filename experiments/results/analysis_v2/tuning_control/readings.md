Tuning control per docs/design/tuning-control.md (pre-registered at TUNE_COMMIT).

## Cells: fit count and classification under each rule

| Cell | Encoding | LR | WD | Fitting seeds, max_train (primary: maximum training accuracy at least 80%) | Classification, max_train | Fitting seeds, selected_epoch (pre-registered: training accuracy at the selected epoch at least 80%) | Classification, selected_epoch | Non-fit kind, max_train | Non-fit kind, selected_epoch | Mean val % | Mean test % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | sinusoidal | 0.0003 | 0 | 3/3 | informative | 1/3 | uninformative | none | seed0: stopped before cap; seed1: stopped before cap | 7.23 | 7.30 |
| E1a | sinusoidal | 0.0001 | 0 | 1/3 | uninformative | 0/3 | uninformative | seed1: epoch cap; seed2: epoch cap | seed0: epoch cap; seed1: epoch cap; seed2: epoch cap | 7.07 | 7.57 |
| E1b | sinusoidal | 0.001 | 0 | 0/3 | uninformative | 0/3 | uninformative | seed0: stopped before cap; seed1: stopped before cap; seed2: stopped before cap | seed0: stopped before cap; seed1: stopped before cap; seed2: stopped before cap | 1.90 | 1.47 |
| E3a | sinusoidal | 0.0003 | 0.01 | 3/3 | informative | 3/3 | informative | none | none | 6.83 | 7.33 |
| R2 | rope | 0.0003 | 0 | 3/3 | informative | 1/3 | uninformative | none | seed0: stopped before cap; seed2: stopped before cap | 19.53 | 19.60 |
| E2a | rope | 0.0001 | 0 | 3/3 | informative | 0/3 | uninformative | none | seed0: stopped before cap; seed1: stopped before cap; seed2: stopped before cap | 19.40 | 20.50 |
| E2b | rope | 0.001 | 0 | 0/3 | uninformative | 0/3 | uninformative | seed0: stopped before cap; seed1: stopped before cap; seed2: stopped before cap | seed0: stopped before cap; seed1: stopped before cap; seed2: stopped before cap | 3.27 | 2.20 |
| E3b | sinusoidal | 0.0003 | 0.1 | 3/3 | informative | 3/3 | informative | none | none | 7.80 | 7.33 |

Classification differs between rules: R1 (informative under max_train, uninformative under selected_epoch); R2 (informative under max_train, uninformative under selected_epoch); E2a (informative under max_train, uninformative under selected_epoch)
All other cells: same classification under both rules.

Verdict differs between rules: reading 1 E2a (no under max_train, uninformative under selected_epoch); reading 5 E3a (no under max_train, uninformative under selected_epoch); reading 5 E3b (no under max_train, uninformative under selected_epoch); reading 3 (up under max_train, not evaluable (fewer than 2 eligible seeds) under selected_epoch); reading 4 (up under max_train, not evaluable (fewer than 2 eligible seeds) under selected_epoch)


Fit rule for this section: maximum training accuracy over the run at least 80% (primary, Deviation 2026-11-01).

Under the primary rule, no informative cell moves test accuracy against its reference (E2a, E3a, E3b: 0 of 3 seeds exclude zero in each); the LSTM-minus-Transformer gap holds at R1 and the RoPE gain holds at R2 against R1, each on 3/3 seeds; and three rates are not covered: sinusoidal 1e-4, which does not fit within 200 epochs, and 1e-3 for both encodings, which stops before the cap under the 400-step warmup.

Under the pre-registered fit rule, readings 3 and 4 cannot be computed (1 and 0 eligible seeds), because at the selected epoch R1 fits on 1/3 seeds and R2 on 1/3; the 2026-11-01 Deviations entry is what makes the two central comparisons evaluable.

Descriptive: two cells have a higher mean test accuracy than their reference, E1a (7.57% against R1's 7.30%) and E2a (20.50% against R2's 19.60%); neither counts as movement, E1a because 1/3 of its seeds fit and E2a because no seed's interval excludes zero (per-seed differences +1.00, -0.30 and +2.00 points).

Test accuracy is the endpoint; every test number is shown beside its validation number. Paired bootstraps resample the 1000 test expressions (N_BOOT 1000), seed-matched. Learning rates are selected on mean validation accuracy over seeds 0, 1 and 2, never on test.

Interpretation notes: the learning-rate grid for readings 2 to 4 is the weight-decay-0 cells of each encoding (sinusoidal: E1a, R1, E1b; RoPE: E2a, R2, E2b); E3a and E3b enter reading 5 only. A seed is eligible for a comparison when both runs fit under this section's fit rule (stated above); the LSTM paper checkpoints count as fitting.

## Per-run table

| Cell | Seed | Selected epoch | Train acc % at selected epoch | Max train acc % | Final train acc % | Max minus final | Spike flag | Val acc % | Test acc % | Fits | Diverged | Epoch cap | 70-90% band |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 0 | 102 | 77.83 | 91.60 | 91.54 | 0.06 | False | 7.70 | 7.80 | True | False | False | False |
| R1 | 1 | 114 | 76.56 | 90.44 | 90.28 | 0.16 | False | 7.00 | 7.10 | True | False | False | False |
| R1 | 2 | 135 | 85.92 | 91.89 | 91.76 | 0.12 | False | 7.00 | 7.00 | True | False | False | False |
| E1a | 0 | 157 | 75.38 | 80.91 | 80.44 | 0.47 | False | 7.10 | 7.30 | True | False | True | True |
| E1a | 1 | 193 | 75.25 | 76.04 | 75.69 | 0.35 | False | 7.10 | 7.60 | False | False | True | True |
| E1a | 2 | 174 | 76.86 | 78.89 | 78.75 | 0.14 | False | 7.00 | 7.80 | False | False | True | True |
| E1b | 0 | 5 | 1.65 | 1.71 | 1.05 | 0.66 | False | 1.90 | 1.60 | False | False | False | False |
| E1b | 1 | 7 | 1.49 | 1.73 | 1.09 | 0.64 | False | 1.90 | 1.30 | False | False | False | False |
| E1b | 2 | 3 | 1.38 | 1.52 | 1.07 | 0.45 | False | 1.90 | 1.50 | False | False | False | False |
| E3a | 0 | 112 | 86.51 | 95.46 | 95.20 | 0.26 | False | 6.70 | 7.60 | True | False | False | False |
| E3a | 1 | 121 | 82.95 | 92.72 | 92.72 | 0.00 | False | 6.50 | 7.10 | True | False | False | False |
| E3a | 2 | 147 | 90.09 | 94.09 | 93.85 | 0.24 | False | 7.30 | 7.30 | True | False | False | False |
| R2 | 0 | 52 | 67.44 | 98.70 | 96.95 | 1.75 | False | 20.20 | 19.60 | True | False | False | False |
| R2 | 1 | 63 | 85.36 | 99.69 | 99.19 | 0.50 | False | 19.10 | 19.70 | True | False | False | False |
| R2 | 2 | 42 | 48.49 | 96.97 | 96.97 | 0.00 | False | 19.30 | 19.50 | True | False | False | False |
| E2a | 0 | 71 | 66.42 | 96.05 | 95.65 | 0.40 | False | 18.80 | 20.60 | True | False | False | False |
| E2a | 1 | 73 | 70.47 | 96.40 | 96.40 | 0.00 | False | 19.00 | 19.40 | True | False | False | False |
| E2a | 2 | 81 | 78.31 | 97.94 | 97.21 | 0.72 | False | 20.40 | 21.50 | True | False | False | False |
| E2b | 0 | 5 | 2.60 | 2.91 | 1.45 | 1.46 | False | 3.30 | 2.60 | False | False | False | False |
| E2b | 1 | 7 | 2.59 | 2.59 | 1.43 | 1.16 | False | 3.60 | 2.40 | False | False | False | False |
| E2b | 2 | 3 | 1.89 | 2.64 | 1.43 | 1.21 | False | 2.90 | 1.60 | False | False | False | False |
| E3b | 0 | 159 | 98.15 | 98.95 | 98.90 | 0.05 | False | 8.00 | 7.10 | True | False | True | False |
| E3b | 1 | 108 | 84.41 | 97.33 | 97.33 | 0.00 | False | 7.30 | 7.50 | True | False | False | False |
| E3b | 2 | 140 | 96.56 | 98.86 | 98.86 | 0.00 | False | 8.10 | 7.40 | True | False | False | False |

## Reading 1: tuning effect per cell (cell minus its reference)

### E1a against R1

fewer than 2 of 3 seeds fit; reported as uninformative, not as a negative result

### E1b against R1

fewer than 2 of 3 seeds fit; reported as uninformative, not as a negative result

### E2a against R2

*all fitting seeds:* no (0/3 eligible seeds exclude 0 in one direction).

| Seed | E2a test % (val %) | R2 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 20.60 (18.80) | 19.60 (20.20) | +1.00 | [-1.70, +4.00] | yes |
| 1 | 19.40 (19.00) | 19.70 (19.10) | -0.30 | [-3.20, +2.60] | yes |
| 2 | 21.50 (20.40) | 19.50 (19.30) | +2.00 | [-0.80, +4.70] | yes |

*excluding runs with 70-90% training accuracy:* no (0/3 eligible seeds exclude 0 in one direction).

| Seed | E2a test % (val %) | R2 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 20.60 (18.80) | 19.60 (20.20) | +1.00 | [-1.70, +4.00] | yes |
| 1 | 19.40 (19.00) | 19.70 (19.10) | -0.30 | [-3.20, +2.60] | yes |
| 2 | 21.50 (20.40) | 19.50 (19.30) | +2.00 | [-0.80, +4.70] | yes |

### E2b against R2

fewer than 2 of 3 seeds fit; reported as uninformative, not as a negative result


## Reading 2: best over the grid

Maxima and per-rate means over seeds 0, 1 and 2 are labelled as such.

### sinusoidal

Maximum test accuracy over its learning rates and seeds: 7.80%, reached by two runs: R1 seed 0, which fits (maximum training accuracy 91.60%), and E1a seed 2, which does not (78.89%). Best configuration by mean validation accuracy over all of the encoding's cells (E1a, R1, E1b, E3a, E3b): **E3b**; best learning-rate-grid cell: **R1**.

| Cell | LR | Mean val % (mean over seeds) | Mean test % (mean over seeds) |
|---|---|---|---|
| E1a | 0.0001 | 7.07 | 7.57 |
| R1 | 0.0003 | 7.23 | 7.30 |
| E1b | 0.001 | 1.90 | 1.47 |

Expected maximum validation accuracy after n configurations (Dodge et al.), descriptive: n=1: 6.17, n=2: 7.18, n=3: 7.49, n=4: 7.63, n=5: 7.72, n=6: 7.78, n=7: 7.83, n=8: 7.87, n=9: 7.90, n=10: 7.93, n=11: 7.95, n=12: 7.97, n=13: 7.99, n=14: 8.00, n=15: 8.01

### rope

Maximum test accuracy over its learning rates and seeds: 21.50%, reached by one run: E2a seed 2, which fits (maximum training accuracy 97.94%). Best configuration by mean validation accuracy over all of the encoding's cells (E2a, R2, E2b): **R2**; best learning-rate-grid cell: **R2**.

| Cell | LR | Mean val % (mean over seeds) | Mean test % (mean over seeds) |
|---|---|---|---|
| E2a | 0.0001 | 19.40 | 20.50 |
| R2 | 0.0003 | 19.53 | 19.60 |
| E2b | 0.001 | 3.27 | 2.20 |

Expected maximum validation accuracy after n configurations (Dodge et al.), descriptive: n=1: 14.07, n=2: 17.83, n=3: 19.18, n=4: 19.70, n=5: 19.94, n=6: 20.06, n=7: 20.14, n=8: 20.19, n=9: 20.23

## Reading 3: the architecture gap under tuning

Sinusoidal learning rate with the highest mean validation accuracy: cell R1. LSTM minus Transformer.

Selection pool, sinusoidal (mean validation accuracy over seeds 0, 1 and 2, whatever their fit): E1a 7.07%, R1 7.23%, E1b 1.90%. Selected: R1.

*all fitting seeds:* gap holds up (3/3 eligible seeds exclude 0 in one direction).

| Seed | LSTM test % (val %) | R1 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 33.60 (35.10) | 7.80 (7.70) | +25.80 | [+22.90, +29.10] | yes |
| 1 | 35.50 (35.90) | 7.10 (7.00) | +28.40 | [+25.30, +31.60] | yes |
| 2 | 34.60 (36.10) | 7.00 (7.00) | +27.60 | [+24.80, +30.70] | yes |

*excluding runs with 70-90% training accuracy:* gap holds up (3/3 eligible seeds exclude 0 in one direction).

| Seed | LSTM test % (val %) | R1 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 33.60 (35.10) | 7.80 (7.70) | +25.80 | [+22.90, +29.10] | yes |
| 1 | 35.50 (35.90) | 7.10 (7.00) | +28.40 | [+25.30, +31.60] | yes |
| 2 | 34.60 (36.10) | 7.00 (7.00) | +27.60 | [+24.80, +30.70] | yes |

### Descriptive: the gap at the best configuration overall (cell E3b, all sinusoidal cells including weight decay)

Not part of the pre-registered reading 3.

*all fitting seeds:* gap holds up (3/3 eligible seeds exclude 0 in one direction).

| Seed | LSTM test % (val %) | E3b test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 33.60 (35.10) | 7.10 (8.00) | +26.50 | [+23.50, +29.60] | yes |
| 1 | 35.50 (35.90) | 7.50 (7.30) | +28.00 | [+24.90, +31.20] | yes |
| 2 | 34.60 (36.10) | 7.40 (8.10) | +27.20 | [+24.30, +30.20] | yes |

*excluding runs with 70-90% training accuracy:* gap holds up (3/3 eligible seeds exclude 0 in one direction).

| Seed | LSTM test % (val %) | E3b test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 33.60 (35.10) | 7.10 (8.00) | +26.50 | [+23.50, +29.60] | yes |
| 1 | 35.50 (35.90) | 7.50 (7.30) | +28.00 | [+24.90, +31.20] | yes |
| 2 | 34.60 (36.10) | 7.40 (8.10) | +27.20 | [+24.30, +30.20] | yes |

## Reading 4: RoPE's gain under tuning

Best-validation cells: RoPE R2, sinusoidal R1. RoPE minus sinusoidal.

Selection pool, rope (mean validation accuracy over seeds 0, 1 and 2, whatever their fit): E2a 19.40%, R2 19.53%, E2b 3.27%. Selected: R2.
Selection pool, sinusoidal (mean validation accuracy over seeds 0, 1 and 2, whatever their fit): E1a 7.07%, R1 7.23%, E1b 1.90%. Selected: R1.

*all fitting seeds:* gain holds up (3/3 eligible seeds exclude 0 in one direction).

| Seed | RoPE test % (val %) | sinusoidal test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 19.60 (20.20) | 7.80 (7.70) | +11.80 | [+9.30, +14.50] | yes |
| 1 | 19.70 (19.10) | 7.10 (7.00) | +12.60 | [+9.90, +15.30] | yes |
| 2 | 19.50 (19.30) | 7.00 (7.00) | +12.50 | [+9.80, +15.30] | yes |

*excluding runs with 70-90% training accuracy:* gain holds up (3/3 eligible seeds exclude 0 in one direction).

| Seed | RoPE test % (val %) | sinusoidal test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 19.60 (20.20) | 7.80 (7.70) | +11.80 | [+9.30, +14.50] | yes |
| 1 | 19.70 (19.10) | 7.10 (7.00) | +12.60 | [+9.90, +15.30] | yes |
| 2 | 19.50 (19.30) | 7.00 (7.00) | +12.50 | [+9.80, +15.30] | yes |


## Reading 5: weight decay (reading 1 applied to E3a and E3b)

### E3a against R1

*all fitting seeds:* no (0/3 eligible seeds exclude 0 in one direction).

| Seed | E3a test % (val %) | R1 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 7.60 (6.70) | 7.80 (7.70) | -0.20 | [-1.30, +1.00] | yes |
| 1 | 7.10 (6.50) | 7.10 (7.00) | +0.00 | [-1.20, +1.30] | yes |
| 2 | 7.30 (7.30) | 7.00 (7.00) | +0.30 | [-1.00, +1.50] | yes |

*excluding runs with 70-90% training accuracy:* no (0/3 eligible seeds exclude 0 in one direction).

| Seed | E3a test % (val %) | R1 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 7.60 (6.70) | 7.80 (7.70) | -0.20 | [-1.30, +1.00] | yes |
| 1 | 7.10 (6.50) | 7.10 (7.00) | +0.00 | [-1.20, +1.30] | yes |
| 2 | 7.30 (7.30) | 7.00 (7.00) | +0.30 | [-1.00, +1.50] | yes |

### E3b against R1

*all fitting seeds:* no (0/3 eligible seeds exclude 0 in one direction).

| Seed | E3b test % (val %) | R1 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 7.10 (8.00) | 7.80 (7.70) | -0.70 | [-2.00, +0.60] | yes |
| 1 | 7.50 (7.30) | 7.10 (7.00) | +0.40 | [-0.90, +1.60] | yes |
| 2 | 7.40 (8.10) | 7.00 (7.00) | +0.40 | [-1.00, +1.70] | yes |

*excluding runs with 70-90% training accuracy:* no (0/3 eligible seeds exclude 0 in one direction).

| Seed | E3b test % (val %) | R1 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 7.10 (8.00) | 7.80 (7.70) | -0.70 | [-2.00, +0.60] | yes |
| 1 | 7.50 (7.30) | 7.10 (7.00) | +0.40 | [-0.90, +1.60] | yes |
| 2 | 7.40 (8.10) | 7.00 (7.00) | +0.40 | [-1.00, +1.70] | yes |

Descriptive: with weight decay, maximum training accuracy is higher than R1's on every seed (E3a 92.72–95.46, E3b 97.33–98.95, R1 90.44–91.89), and training accuracy at the selected epoch is higher on average (means E3a 86.52, E3b 93.04, R1 80.10), while test accuracy does not move (reading 5).


## Descriptive: RTX 4000 Ada runs beside the same-GPU references (never references)

### sinusoidal_retrain beside R1

Mean val 7.23%, mean test 7.57% (RTX 4000 Ada) against the pod reference. Difference RTX 4000 Ada minus pod:

| Seed | sinusoidal_retrain test % (val %) | R1 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 7.60 (7.30) | 7.80 (7.70) | -0.20 | [-1.50, +1.10] | yes |
| 1 | 7.40 (6.60) | 7.10 (7.00) | +0.30 | [-1.00, +1.50] | yes |
| 2 | 7.70 (7.80) | 7.00 (7.00) | +0.70 | [-0.50, +2.10] | yes |

### rope beside R2

Mean val 21.33%, mean test 20.30% (RTX 4000 Ada) against the pod reference. Difference RTX 4000 Ada minus pod:

| Seed | rope test % (val %) | R2 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 21.40 (22.70) | 19.60 (20.20) | +1.80 | [-1.10, +4.90] | yes |
| 1 | 18.00 (19.80) | 19.70 (19.10) | -1.70 | [-4.70, +1.50] | yes |
| 2 | 21.50 (21.50) | 19.50 (19.30) | +2.00 | [-0.90, +5.30] | yes |

## Out-of-distribution accuracy, operations 4-7 (descriptive, means over seeds 0, 1 and 2)

| Cell | ops4 | ops5 | ops6 | ops7 |
|---|---|---|---|---|
| R1 | 0.53 | 0.40 | 0.27 | 0.13 |
| E1a | 0.60 | 0.20 | 0.23 | 0.23 |
| E1b | 0.50 | 0.33 | 0.37 | 0.03 |
| E3a | 0.43 | 0.27 | 0.23 | 0.27 |
| R2 | 1.63 | 0.87 | 0.40 | 0.47 |
| E2a | 1.97 | 0.47 | 0.37 | 0.33 |
| E2b | 0.60 | 0.47 | 0.33 | 0.23 |
| E3b | 0.33 | 0.23 | 0.23 | 0.33 |
| lstm | 14.83 | 8.23 | 2.70 | 2.13 |

## Compute

Total training time over all 24 runs: 7.56 run-hours; reference runs (R1, R2): 2.00 run-hours. sum of per-run training wall time; two runs shared the pod, so pod wall-clock is shorter.

---

## Under the pre-registered fit rule (selected-epoch training accuracy)


Fit rule for this section: training accuracy at the selected epoch at least 80% (as pre-registered).

Test accuracy is the endpoint; every test number is shown beside its validation number. Paired bootstraps resample the 1000 test expressions (N_BOOT 1000), seed-matched. Learning rates are selected on mean validation accuracy over seeds 0, 1 and 2, never on test.

Interpretation notes: the learning-rate grid for readings 2 to 4 is the weight-decay-0 cells of each encoding (sinusoidal: E1a, R1, E1b; RoPE: E2a, R2, E2b); E3a and E3b enter reading 5 only. A seed is eligible for a comparison when both runs fit under this section's fit rule (stated above); the LSTM paper checkpoints count as fitting.

## Per-run table

| Cell | Seed | Selected epoch | Train acc % at selected epoch | Max train acc % | Final train acc % | Max minus final | Spike flag | Val acc % | Test acc % | Fits | Diverged | Epoch cap | 70-90% band |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 0 | 102 | 77.83 | 91.60 | 91.54 | 0.06 | False | 7.70 | 7.80 | False | False | False | True |
| R1 | 1 | 114 | 76.56 | 90.44 | 90.28 | 0.16 | False | 7.00 | 7.10 | False | False | False | True |
| R1 | 2 | 135 | 85.92 | 91.89 | 91.76 | 0.12 | False | 7.00 | 7.00 | True | False | False | True |
| E1a | 0 | 157 | 75.38 | 80.91 | 80.44 | 0.47 | False | 7.10 | 7.30 | False | False | True | True |
| E1a | 1 | 193 | 75.25 | 76.04 | 75.69 | 0.35 | False | 7.10 | 7.60 | False | False | True | True |
| E1a | 2 | 174 | 76.86 | 78.89 | 78.75 | 0.14 | False | 7.00 | 7.80 | False | False | True | True |
| E1b | 0 | 5 | 1.65 | 1.71 | 1.05 | 0.66 | False | 1.90 | 1.60 | False | False | False | False |
| E1b | 1 | 7 | 1.49 | 1.73 | 1.09 | 0.64 | False | 1.90 | 1.30 | False | False | False | False |
| E1b | 2 | 3 | 1.38 | 1.52 | 1.07 | 0.45 | False | 1.90 | 1.50 | False | False | False | False |
| E3a | 0 | 112 | 86.51 | 95.46 | 95.20 | 0.26 | False | 6.70 | 7.60 | True | False | False | True |
| E3a | 1 | 121 | 82.95 | 92.72 | 92.72 | 0.00 | False | 6.50 | 7.10 | True | False | False | True |
| E3a | 2 | 147 | 90.09 | 94.09 | 93.85 | 0.24 | False | 7.30 | 7.30 | True | False | False | False |
| R2 | 0 | 52 | 67.44 | 98.70 | 96.95 | 1.75 | False | 20.20 | 19.60 | False | False | False | False |
| R2 | 1 | 63 | 85.36 | 99.69 | 99.19 | 0.50 | False | 19.10 | 19.70 | True | False | False | True |
| R2 | 2 | 42 | 48.49 | 96.97 | 96.97 | 0.00 | False | 19.30 | 19.50 | False | False | False | False |
| E2a | 0 | 71 | 66.42 | 96.05 | 95.65 | 0.40 | False | 18.80 | 20.60 | False | False | False | False |
| E2a | 1 | 73 | 70.47 | 96.40 | 96.40 | 0.00 | False | 19.00 | 19.40 | False | False | False | True |
| E2a | 2 | 81 | 78.31 | 97.94 | 97.21 | 0.72 | False | 20.40 | 21.50 | False | False | False | True |
| E2b | 0 | 5 | 2.60 | 2.91 | 1.45 | 1.46 | False | 3.30 | 2.60 | False | False | False | False |
| E2b | 1 | 7 | 2.59 | 2.59 | 1.43 | 1.16 | False | 3.60 | 2.40 | False | False | False | False |
| E2b | 2 | 3 | 1.89 | 2.64 | 1.43 | 1.21 | False | 2.90 | 1.60 | False | False | False | False |
| E3b | 0 | 159 | 98.15 | 98.95 | 98.90 | 0.05 | False | 8.00 | 7.10 | True | False | True | False |
| E3b | 1 | 108 | 84.41 | 97.33 | 97.33 | 0.00 | False | 7.30 | 7.50 | True | False | False | True |
| E3b | 2 | 140 | 96.56 | 98.86 | 98.86 | 0.00 | False | 8.10 | 7.40 | True | False | False | False |

## Reading 1: tuning effect per cell (cell minus its reference)

### E1a against R1

fewer than 2 of 3 seeds fit; reported as uninformative, not as a negative result

### E1b against R1

fewer than 2 of 3 seeds fit; reported as uninformative, not as a negative result

### E2a against R2

fewer than 2 of 3 seeds fit; reported as uninformative, not as a negative result

### E2b against R2

fewer than 2 of 3 seeds fit; reported as uninformative, not as a negative result


## Reading 2: best over the grid

Maxima and per-rate means over seeds 0, 1 and 2 are labelled as such.

### sinusoidal

Maximum test accuracy over its learning rates and seeds: 7.80%, reached by two runs: R1 seed 0, which does not fit (training accuracy at the selected epoch 77.83%), and E1a seed 2, which does not either (76.86%). Best configuration by mean validation accuracy over all of the encoding's cells (E1a, R1, E1b, E3a, E3b): **E3b**; best learning-rate-grid cell: **R1**.

| Cell | LR | Mean val % (mean over seeds) | Mean test % (mean over seeds) |
|---|---|---|---|
| E1a | 0.0001 | 7.07 | 7.57 |
| R1 | 0.0003 | 7.23 | 7.30 |
| E1b | 0.001 | 1.90 | 1.47 |

Expected maximum validation accuracy after n configurations (Dodge et al.), descriptive: n=1: 6.17, n=2: 7.18, n=3: 7.49, n=4: 7.63, n=5: 7.72, n=6: 7.78, n=7: 7.83, n=8: 7.87, n=9: 7.90, n=10: 7.93, n=11: 7.95, n=12: 7.97, n=13: 7.99, n=14: 8.00, n=15: 8.01

### rope

Maximum test accuracy over its learning rates and seeds: 21.50%, reached by one run: E2a seed 2, which does not fit (training accuracy at the selected epoch 78.31%). Best configuration by mean validation accuracy over all of the encoding's cells (E2a, R2, E2b): **R2**; best learning-rate-grid cell: **R2**.

| Cell | LR | Mean val % (mean over seeds) | Mean test % (mean over seeds) |
|---|---|---|---|
| E2a | 0.0001 | 19.40 | 20.50 |
| R2 | 0.0003 | 19.53 | 19.60 |
| E2b | 0.001 | 3.27 | 2.20 |

Expected maximum validation accuracy after n configurations (Dodge et al.), descriptive: n=1: 14.07, n=2: 17.83, n=3: 19.18, n=4: 19.70, n=5: 19.94, n=6: 20.06, n=7: 20.14, n=8: 20.19, n=9: 20.23

## Reading 3: the architecture gap under tuning

Sinusoidal learning rate with the highest mean validation accuracy: cell R1. LSTM minus Transformer.

Selection pool, sinusoidal (mean validation accuracy over seeds 0, 1 and 2, whatever their fit): E1a 7.07%, R1 7.23%, E1b 1.90%. Selected: R1.

*all fitting seeds:* not evaluable (fewer than 2 eligible seeds) (1/1 eligible seeds exclude 0 in one direction).

| Seed | LSTM test % (val %) | R1 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 33.60 (35.10) | 7.80 (7.70) | +25.80 | [+22.90, +29.10] | no |
| 1 | 35.50 (35.90) | 7.10 (7.00) | +28.40 | [+25.30, +31.60] | no |
| 2 | 34.60 (36.10) | 7.00 (7.00) | +27.60 | [+24.80, +30.70] | yes |

*excluding runs with 70-90% training accuracy:* not evaluable (fewer than 2 eligible seeds) (0/0 eligible seeds exclude 0 in one direction).

| Seed | LSTM test % (val %) | R1 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 33.60 (35.10) | 7.80 (7.70) | +25.80 | [+22.90, +29.10] | no |
| 1 | 35.50 (35.90) | 7.10 (7.00) | +28.40 | [+25.30, +31.60] | no |
| 2 | 34.60 (36.10) | 7.00 (7.00) | +27.60 | [+24.80, +30.70] | no |

### Descriptive: the gap at the best configuration overall (cell E3b, all sinusoidal cells including weight decay)

Not part of the pre-registered reading 3.

*all fitting seeds:* gap holds up (3/3 eligible seeds exclude 0 in one direction).

| Seed | LSTM test % (val %) | E3b test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 33.60 (35.10) | 7.10 (8.00) | +26.50 | [+23.50, +29.60] | yes |
| 1 | 35.50 (35.90) | 7.50 (7.30) | +28.00 | [+24.90, +31.20] | yes |
| 2 | 34.60 (36.10) | 7.40 (8.10) | +27.20 | [+24.30, +30.20] | yes |

*excluding runs with 70-90% training accuracy:* gap holds up (2/2 eligible seeds exclude 0 in one direction).

| Seed | LSTM test % (val %) | E3b test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 33.60 (35.10) | 7.10 (8.00) | +26.50 | [+23.50, +29.60] | yes |
| 1 | 35.50 (35.90) | 7.50 (7.30) | +28.00 | [+24.90, +31.20] | no |
| 2 | 34.60 (36.10) | 7.40 (8.10) | +27.20 | [+24.30, +30.20] | yes |

## Reading 4: RoPE's gain under tuning

Best-validation cells: RoPE R2, sinusoidal R1. RoPE minus sinusoidal.

Selection pool, rope (mean validation accuracy over seeds 0, 1 and 2, whatever their fit): E2a 19.40%, R2 19.53%, E2b 3.27%. Selected: R2.
Selection pool, sinusoidal (mean validation accuracy over seeds 0, 1 and 2, whatever their fit): E1a 7.07%, R1 7.23%, E1b 1.90%. Selected: R1.

*all fitting seeds:* not evaluable (fewer than 2 eligible seeds) (0/0 eligible seeds exclude 0 in one direction).

| Seed | RoPE test % (val %) | sinusoidal test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 19.60 (20.20) | 7.80 (7.70) | +11.80 | [+9.30, +14.50] | no |
| 1 | 19.70 (19.10) | 7.10 (7.00) | +12.60 | [+9.90, +15.30] | no |
| 2 | 19.50 (19.30) | 7.00 (7.00) | +12.50 | [+9.80, +15.30] | no |

*excluding runs with 70-90% training accuracy:* not evaluable (fewer than 2 eligible seeds) (0/0 eligible seeds exclude 0 in one direction).

| Seed | RoPE test % (val %) | sinusoidal test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 19.60 (20.20) | 7.80 (7.70) | +11.80 | [+9.30, +14.50] | no |
| 1 | 19.70 (19.10) | 7.10 (7.00) | +12.60 | [+9.90, +15.30] | no |
| 2 | 19.50 (19.30) | 7.00 (7.00) | +12.50 | [+9.80, +15.30] | no |


## Reading 5: weight decay (reading 1 applied to E3a and E3b)

### E3a against R1

fewer than 2 of 3 seeds fit; reported as uninformative, not as a negative result

### E3b against R1

fewer than 2 of 3 seeds fit; reported as uninformative, not as a negative result

Descriptive: with weight decay, maximum training accuracy is higher than R1's on every seed (E3a 92.72–95.46, E3b 97.33–98.95, R1 90.44–91.89), and training accuracy at the selected epoch is higher on average (means E3a 86.52, E3b 93.04, R1 80.10), while test accuracy does not move (reading 5).


## Descriptive: RTX 4000 Ada runs beside the same-GPU references (never references)

### sinusoidal_retrain beside R1

Mean val 7.23%, mean test 7.57% (RTX 4000 Ada) against the pod reference. Difference RTX 4000 Ada minus pod:

| Seed | sinusoidal_retrain test % (val %) | R1 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 7.60 (7.30) | 7.80 (7.70) | -0.20 | [-1.50, +1.10] | no |
| 1 | 7.40 (6.60) | 7.10 (7.00) | +0.30 | [-1.00, +1.50] | no |
| 2 | 7.70 (7.80) | 7.00 (7.00) | +0.70 | [-0.50, +2.10] | yes |

### rope beside R2

Mean val 21.33%, mean test 20.30% (RTX 4000 Ada) against the pod reference. Difference RTX 4000 Ada minus pod:

| Seed | rope test % (val %) | R2 test % (val %) | Difference (points) | 95% interval (points) | Eligible |
|---|---|---|---|---|---|
| 0 | 21.40 (22.70) | 19.60 (20.20) | +1.80 | [-1.10, +4.90] | no |
| 1 | 18.00 (19.80) | 19.70 (19.10) | -1.70 | [-4.70, +1.50] | no |
| 2 | 21.50 (21.50) | 19.50 (19.30) | +2.00 | [-0.90, +5.30] | no |

## Out-of-distribution accuracy, operations 4-7 (descriptive, means over seeds 0, 1 and 2)

| Cell | ops4 | ops5 | ops6 | ops7 |
|---|---|---|---|---|
| R1 | 0.53 | 0.40 | 0.27 | 0.13 |
| E1a | 0.60 | 0.20 | 0.23 | 0.23 |
| E1b | 0.50 | 0.33 | 0.37 | 0.03 |
| E3a | 0.43 | 0.27 | 0.23 | 0.27 |
| R2 | 1.63 | 0.87 | 0.40 | 0.47 |
| E2a | 1.97 | 0.47 | 0.37 | 0.33 |
| E2b | 0.60 | 0.47 | 0.33 | 0.23 |
| E3b | 0.33 | 0.23 | 0.23 | 0.33 |
| lstm | 14.83 | 8.23 | 2.70 | 2.13 |

## Compute

Total training time over all 24 runs: 7.56 run-hours; reference runs (R1, R2): 2.00 run-hours. sum of per-run training wall time; two runs shared the pod, so pod wall-clock is shorter.
