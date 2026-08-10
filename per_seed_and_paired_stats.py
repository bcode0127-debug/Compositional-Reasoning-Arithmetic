#!/usr/bin/env python3
"""Item 10: per-seed results and uncertainty.

Every figure is read from a committed artifact, none typed:
  matrix_summary.txt                      selected epoch, val accuracy
  results_v2/*/*/seed*/history.json       final epoch, train acc at selected epoch
  results_v2/*/*/seed*/ood_results.json   per-bucket OOD accuracy
  results_v2/test_set_results.json        test accuracy + per-example correctness

Reporting choices, deliberate:
  - Across seeds (n=3) we report mean and SD descriptively only. No t-tests and
    no intervals: three seeds cannot support them.
  - Example-level tests run WITHIN a seed, where n=1000 paired observations is
    a real sample. McNemar uses the exact binomial on discordant pairs rather
    than the chi-square approximation, since discordant counts here are small.
  - The paired bootstrap resamples EXAMPLES (not seeds), preserving the pairing
    by resampling index vectors applied to both models at once.

Writes results_v2/per_seed_and_paired_stats.json.
"""
import json
import math
import random
import statistics as st
from pathlib import Path

SEEDS = (0, 1, 2)
N_BOOT = 10_000
BOOT_SEED = 12345


def load(p):
    return json.loads(Path(p).read_text())


def matrix_summary():
    out = {}
    for line in Path("matrix_summary.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        out[parts[0]] = {
            "val_accuracy": float(parts[1].split("=")[1].rstrip("%")),
            "best_epoch": int(parts[2].split("=")[1]),
            "epochs_run": int(parts[3].split("=")[1]),
        }
    return out


def mcnemar_exact(b, c):
    """Two-sided exact binomial McNemar on discordant pairs.
    b = model A correct & B wrong, c = A wrong & B correct.
    Returns (statistic_n_discordant, p_value)."""
    n = b + c
    if n == 0:
        return 0, 1.0
    k = min(b, c)
    # two-sided exact: 2 * P(X <= k) under Binomial(n, 0.5), capped at 1
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n, min(1.0, 2 * tail)


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile CI on mean(a) - mean(b), resampling example indices so the
    pairing between the two models is preserved."""
    rng = random.Random(seed)
    n = len(a)
    idx = range(n)
    diffs = []
    for _ in range(n_boot):
        sample = [rng.randrange(n) for _ in idx]
        diffs.append((sum(a[i] for i in sample) - sum(b[i] for i in sample)) / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]
    return 100.0 * lo, 100.0 * hi


def main():
    ms = matrix_summary()
    tsr = load("results_v2/test_set_results.json")
    report = {"per_seed": [], "paired_seed_differences": {}, "example_level": {}}

    # ---------------- STEP 2: per-seed table ----------------
    print("=" * 118)
    print("STEP 2 -- per-seed results (12 rows)")
    print("=" * 118)
    hdr = f"{'model':<12}{'study':<8}{'seed':<5}{'sel.ep':>7}{'fin.ep':>7}{'train@sel':>11}{'val':>8}{'test':>8}   OOD"
    print(hdr)
    print("-" * 118)
    for study in ("study1", "study2"):
        for model in ("lstm", "transformer"):
            for s in SEEDS:
                tag = f"{model}_{study}_seed{s}"
                hist = load(f"results_v2/{model}/{study}/seed{s}/history.json")
                sel = ms[tag]["best_epoch"]
                fin = len(hist["train_accuracies"]) - 1
                tr = hist["train_accuracies"][sel]
                val = ms[tag]["val_accuracy"]
                test = tsr[tag]["test_accuracy"]
                ood_raw = load(f"results_v2/{model}/{study}/seed{s}/ood_results.json")
                if study == "study1":
                    ood = {f"ops{n}": ood_raw[f"study1/ood_ops{n}.json"] for n in (4, 5, 6, 7)}
                    ood_s = " ".join(f"{k}={v:.1f}" for k, v in ood.items())
                else:
                    ood = {"ood": ood_raw["study2/ood.json"]}
                    ood_s = f"{ood['ood']:.1f}"
                row = {"model": model, "study": study, "seed": s, "selected_epoch": sel,
                       "final_epoch": fin, "train_acc_at_selected": tr,
                       "val_accuracy": val, "test_accuracy": test, "ood": ood}
                report["per_seed"].append(row)
                print(f"{model:<12}{study:<8}{s:<5}{sel:>7}{fin:>7}{tr:>10.2f}%{val:>7.1f}%{test:>7.2f}%   {ood_s}")

    # ---------------- STEP 3: paired per-seed differences ----------------
    print()
    print("=" * 118)
    print("STEP 3 -- paired per-seed differences on test (LSTM minus Transformer), n=3, descriptive only")
    print("=" * 118)
    for study in ("study1", "study2"):
        diffs = [tsr[f"lstm_{study}_seed{s}"]["test_accuracy"]
                 - tsr[f"transformer_{study}_seed{s}"]["test_accuracy"] for s in SEEDS]
        report["paired_seed_differences"][study] = {
            "per_seed": diffs, "mean": st.mean(diffs), "sd": st.stdev(diffs), "n_seeds": 3}
        per = "  ".join(f"seed{s}={d:+.2f}" for s, d in zip(SEEDS, diffs))
        print(f"{study}: {per}   mean={st.mean(diffs):+.2f}  SD={st.stdev(diffs):.2f}  (n=3, no interval reported)")

    # ---------------- STEPS 4+5: example-level paired comparison ----------------
    print()
    print("=" * 118)
    print("STEPS 4-5 -- example-level paired comparison on the test set (n=1000 per cell)")
    print("=" * 118)
    print(f"{'study':<8}{'seed':<5}{'both':>7}{'LSTM only':>11}{'TFM only':>10}{'neither':>9}"
          f"{'discord':>9}{'McNemar p':>12}{'diff (pp)':>11}{'bootstrap 95% CI':>24}")
    print("-" * 118)
    for study in ("study1", "study2"):
        report["example_level"][study] = {}
        for s in SEEDS:
            L = tsr[f"lstm_{study}_seed{s}"]["per_example_correct"]
            T = tsr[f"transformer_{study}_seed{s}"]["per_example_correct"]
            assert len(L) == len(T), "paired vectors must align"
            both = sum(1 for a, b in zip(L, T) if a and b)
            l_only = sum(1 for a, b in zip(L, T) if a and not b)
            t_only = sum(1 for a, b in zip(L, T) if b and not a)
            neither = sum(1 for a, b in zip(L, T) if not a and not b)
            n_disc, p = mcnemar_exact(l_only, t_only)
            diff = 100.0 * (sum(L) - sum(T)) / len(L)
            lo, hi = paired_bootstrap(L, T)
            report["example_level"][study][f"seed{s}"] = {
                "n": len(L), "both_correct": both, "lstm_only": l_only,
                "transformer_only": t_only, "neither": neither,
                "n_discordant": n_disc, "mcnemar_exact_p": p,
                "accuracy_diff_pp": diff, "bootstrap_ci95_pp": [lo, hi]}
            print(f"{study:<8}{s:<5}{both:>7}{l_only:>11}{t_only:>10}{neither:>9}"
                  f"{n_disc:>9}{p:>12.3g}{diff:>10.2f}p{'[' + f'{lo:+.2f}, {hi:+.2f}' + ']':>24}")

    Path("results_v2/per_seed_and_paired_stats.json").write_text(json.dumps(report, indent=2))
    print("\nWrote results_v2/per_seed_and_paired_stats.json")


if __name__ == "__main__":
    main()
