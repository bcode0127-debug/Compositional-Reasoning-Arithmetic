#!/usr/bin/env python3
"""Chance baselines for the 'noise floor' claim (reviewer item 6).

Computes four reference accuracies on Study 1's four OOD buckets, so the
Transformer's ~0.3-0.7% OOD accuracy can be compared against what trivial
strategies achieve on the same data rather than against zero.

Baselines
  1. most-common-answer   -- always predict the modal training target.
  2. length-cond-uniform  -- given the true answer's string length L, guess
                             uniformly among all integers whose decimal string
                             (minus sign included) has length L. Reported as the
                             exact expected accuracy, mean_i 1/N(L_i), not a sample.
  3. unigram              -- character unigram fit on training targets, plus an
                             EOS token; P(exact) = prod p(c) * p(EOS), averaged.
                             Again exact expectation, not sampled.
  4. nn-edit-retrieval    -- predict the target of the training expression with
                             the smallest Levenshtein distance to the query.

Read-only; writes results_v2/chance_baselines.json.
"""
import json
from collections import Counter
from pathlib import Path

DATA = Path("datasets_v2/study1")
OOD = [f"ood_ops{n}.json" for n in (4, 5, 6, 7)]


def load(p):
    return json.loads((DATA / p).read_text())["data"]


def n_ints_of_len(L):
    """How many integers have a decimal string representation of length L."""
    if L == 1:
        return 10                      # 0-9
    pos = 9 * 10 ** (L - 1)            # L-digit positives, no leading zero
    neg = 9 * 10 ** (L - 2) if L >= 2 else 0   # '-' + (L-1) digits
    return pos + neg


def main():
    train = load("train.json")
    train_targets = [str(s["output"]) for s in train]
    train_exprs = [s["input"] for s in train]

    modal, modal_n = Counter(train_targets).most_common(1)[0]
    print(f"modal training target: {modal!r} ({modal_n}/{len(train_targets)} = "
          f"{100*modal_n/len(train_targets):.2f}% of training)\n")

    # character unigram over training targets, with EOS
    chars = Counter()
    for t in train_targets:
        chars.update(t)
        chars["<EOS>"] += 1
    total = sum(chars.values())
    p = {c: n / total for c, n in chars.items()}

    from rapidfuzz import process, distance

    results = {}
    print(f"{'bucket':<8}{'most-common':>14}{'len-uniform':>16}{'unigram':>16}{'nn-edit':>12}")
    for f in OOD:
        items = load(f)
        targets = [str(s["output"]) for s in items]
        exprs = [s["input"] for s in items]
        n = len(items)

        acc_modal = 100.0 * sum(t == modal for t in targets) / n
        acc_len = 100.0 * sum(1.0 / n_ints_of_len(len(t)) for t in targets) / n
        acc_uni = 100.0 * sum(
            __import__("math").prod([p.get(c, 0.0) for c in t]) * p["<EOS>"] for t in targets
        ) / n

        # nearest training expression by Levenshtein, predict its target
        idx = process.cdist(exprs, train_exprs, scorer=distance.Levenshtein.distance,
                            workers=-1).argmin(axis=1)
        acc_nn = 100.0 * sum(train_targets[j] == t for j, t in zip(idx, targets)) / n

        results[f] = {"n": n, "most_common_answer": acc_modal,
                      "length_conditioned_uniform": acc_len,
                      "unigram": acc_uni, "nn_edit_retrieval": acc_nn}
        print(f"{f.replace('ood_','').replace('.json',''):<8}"
              f"{acc_modal:>13.3f}%{acc_len:>15.4f}%{acc_uni:>15.4f}%{acc_nn:>11.2f}%")

    Path("results_v2/chance_baselines.json").write_text(json.dumps(results, indent=2))
    print("\nWrote results_v2/chance_baselines.json")


def constant_output_all_populations():
    """Constant-output baseline (always predict the modal TRAINING target) on
    every population the paper reports, with per-example correctness recorded so
    it can be paired against model predictions.

    The modal target is computed per study from that study's own training split:
    Study 2's training distribution differs from Study 1's, so reusing Study 1's
    modal answer would not be the correct trivial baseline for Study 2.
    """
    from collections import Counter

    POPS = {
        "study1": ["val.json", "test.json",
                   "ood_ops4.json", "ood_ops5.json", "ood_ops6.json", "ood_ops7.json"],
        "study2": ["val.json", "test.json", "ood.json"],
    }
    out = {}
    print("\n" + "=" * 78)
    print("Constant-output baseline (modal training target) on all populations")
    print("=" * 78)
    for study, files in POPS.items():
        root = Path("datasets_v2") / study
        train_targets = [str(s["output"]) for s in json.loads((root / "train.json").read_text())["data"]]
        modal, modal_n = Counter(train_targets).most_common(1)[0]
        out[study] = {"modal_target": modal,
                      "modal_train_count": modal_n,
                      "modal_train_frac": modal_n / len(train_targets),
                      "splits": {}}
        print(f"\n{study}: modal training target = {modal!r} "
              f"({modal_n}/{len(train_targets)} = {100*modal_n/len(train_targets):.2f}% of training)")
        for f in files:
            items = json.loads((root / f).read_text())["data"]
            per_example = [int(str(it["output"]) == modal) for it in items]
            acc = 100.0 * sum(per_example) / len(per_example)
            out[study]["splits"][f] = {"n": len(items), "n_correct": sum(per_example),
                                       "accuracy": acc, "per_example_correct": per_example}
            print(f"  {f:<18} n={len(items):<6} correct={sum(per_example):<5} acc={acc:.2f}%")
    Path("results_v2/constant_output_baseline.json").write_text(json.dumps(out, indent=2))
    print("\nWrote results_v2/constant_output_baseline.json")
    return out


if __name__ == "__main__":
    main()
    constant_output_all_populations()
