#!/usr/bin/env python3
"""Model vs constant-output baseline, paired at the example level.

Answers the questions the aggregate comparison cannot: is the LSTM genuinely
above the trivial baseline at four through six operations, is it distinguishable
from it at seven, and is the Transformer really below it on all four buckets?

Method:
  - Pairing is exact: both the model's and the baseline's correctness vectors
    are indexed by the same dataset order, so example i is the same expression
    for both.
  - McNemar uses the exact binomial on discordant pairs. Discordant counts here
    can be small (single digits), where the chi-square approximation is unsafe.
  - The paired bootstrap resamples EXAMPLES, applying one index vector to both
    correctness vectors so pairing is preserved. 10,000 resamples, percentile CI.
  - A CI containing zero means the split does not distinguish model from
    baseline; that is a real finding, not a failure.

Reads results_v2/all_splits_per_example.json and
results_v2/constant_output_baseline.json. Writes results_v2/baseline_paired_stats.json.
"""
import json
import math
import random
from pathlib import Path

N_BOOT = 10_000
BOOT_SEED = 20260809
SEEDS = (0, 1, 2)
SPLITS = {
    "study1": ["val.json", "test.json", "ood_ops4.json", "ood_ops5.json",
               "ood_ops6.json", "ood_ops7.json"],
    "study2": ["val.json", "test.json", "ood.json"],
}


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar. b = model correct & baseline wrong,
    c = model wrong & baseline correct."""
    n = b + c
    if n == 0:
        return 0, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n, min(1.0, 2 * tail)


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        s = [rng.randrange(n) for _ in range(n)]
        diffs.append((sum(a[i] for i in s) - sum(b[i] for i in s)) / n)
    diffs.sort()
    return 100.0 * diffs[int(0.025 * n_boot)], 100.0 * diffs[int(0.975 * n_boot) - 1]


def main():
    models = json.loads(Path("results_v2/all_splits_per_example.json").read_text())
    base = json.loads(Path("results_v2/constant_output_baseline.json").read_text())
    report = {}

    for model_type in ("lstm", "transformer"):
        print("\n" + "=" * 126)
        print(f"{model_type.upper()} vs constant-output baseline (paired, per example)")
        print("=" * 126)
        print(f"{'study':<8}{'split':<15}{'seed':<5}{'model%':>8}{'base%':>7}{'diff':>8}"
              f"{'both':>6}{'M only':>8}{'B only':>8}{'neither':>8}{'disc':>6}"
              f"{'McNemar p':>12}{'bootstrap 95% CI':>22}  verdict")
        print("-" * 126)
        for study, files in SPLITS.items():
            for f in files:
                B = base[study]["splits"][f]["per_example_correct"]
                for s in SEEDS:
                    key = f"{model_type}_{study}_seed{s}"
                    M = models[key]["splits"][f]["per_example_correct"]
                    assert len(M) == len(B)
                    both = sum(1 for m, b in zip(M, B) if m and b)
                    m_only = sum(1 for m, b in zip(M, B) if m and not b)
                    b_only = sum(1 for m, b in zip(M, B) if b and not m)
                    neither = sum(1 for m, b in zip(M, B) if not m and not b)
                    n_disc, p = mcnemar_exact(m_only, b_only)
                    diff = 100.0 * (sum(M) - sum(B)) / len(M)
                    lo, hi = paired_bootstrap(M, B)
                    if lo > 0:
                        verdict = "above baseline"
                    elif hi < 0:
                        verdict = "BELOW baseline"
                    else:
                        verdict = "indistinguishable"
                    report.setdefault(model_type, {}).setdefault(study, {}).setdefault(f, {})[f"seed{s}"] = {
                        "model_acc": 100.0 * sum(M) / len(M), "baseline_acc": 100.0 * sum(B) / len(B),
                        "diff_pp": diff, "both": both, "model_only": m_only, "baseline_only": b_only,
                        "neither": neither, "n_discordant": n_disc, "mcnemar_exact_p": p,
                        "bootstrap_ci95_pp": [lo, hi], "verdict": verdict}
                    print(f"{study:<8}{f.replace('.json',''):<15}{s:<5}"
                          f"{100.0*sum(M)/len(M):>7.1f}%{100.0*sum(B)/len(B):>6.1f}%{diff:>+8.2f}"
                          f"{both:>6}{m_only:>8}{b_only:>8}{neither:>8}{n_disc:>6}"
                          f"{p:>12.3g}{'[' + f'{lo:+.2f}, {hi:+.2f}' + ']':>22}  {verdict}")

    Path("results_v2/baseline_paired_stats.json").write_text(json.dumps(report, indent=2))
    print("\nWrote results_v2/baseline_paired_stats.json")


if __name__ == "__main__":
    main()
