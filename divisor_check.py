#!/usr/bin/env python3
"""Reviewer item 8, divisor check.

The integer-division constraint forces the generator to construct a divisor
that divides its branch exactly. This asks what that constraint actually
produced: if most division nodes divide by 1, then "division" in this dataset
is largely an identity operation, and per-operation division accuracy measures
something other than division.

Parses expressions structurally (matched parentheses) rather than by regex, so
the operand of a division node is the whole sub-expression, not the adjacent
token. Read-only; writes results_v2/divisor_distribution.json.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path("datasets_v2")
FILES = {
    "study1": ["train.json", "val.json", "test.json",
               "ood_ops4.json", "ood_ops5.json", "ood_ops6.json", "ood_ops7.json"],
    "study2": ["train.json", "val.json", "test.json", "ood.json"],
}


def split_top(expr):
    """If expr is '(A op B)', return (A, op, B); else None."""
    if not (expr.startswith("(") and expr.endswith(")")):
        return None
    inner = expr[1:-1]
    depth = 0
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "+-*/" and i > 0 and inner[i-1] == " ":
            return inner[:i-1], ch, inner[i+2:]
    return None


def evaluate(expr):
    p = split_top(expr)
    if p is None:
        return int(expr)
    a, op, b = p
    x, y = evaluate(a), evaluate(b)
    return {"+": x+y, "-": x-y, "*": x*y, "/": x//y}[op]


def division_nodes(expr, out):
    """Collect (divisor_value, quotient_value) for every division node."""
    p = split_top(expr)
    if p is None:
        return
    a, op, b = p
    if op == "/":
        d = evaluate(b)
        out.append((d, evaluate(a) // d))
    division_nodes(a, out)
    division_nodes(b, out)


def main():
    report = {}
    for study, files in FILES.items():
        print("=" * 92)
        print(f"{study}")
        print("=" * 92)
        study_div = Counter(); study_q = Counter()
        n_expr_with_div = 0; n_expr_with_div1 = 0; n_expr_total = 0
        report[study] = {"splits": {}}
        for f in files:
            items = json.loads((ROOT / study / f).read_text())["data"]
            divs = Counter(); quots = Counter()
            with_div = 0; with_div1 = 0
            for it in items:
                nodes = []
                division_nodes(it["input"], nodes)
                if nodes:
                    with_div += 1
                    if any(d == 1 for d, _ in nodes):
                        with_div1 += 1
                for d, q in nodes:
                    divs[d] += 1; quots[abs(q)] += 1
            tot = sum(divs.values())
            frac1 = 100.0 * divs[1] / tot if tot else 0.0
            report[study]["splits"][f] = {
                "n_expressions": len(items),
                "n_division_nodes": tot,
                "divisor_is_1_count": divs[1],
                "divisor_is_1_pct": frac1,
                "divisor_distribution": dict(sorted(divs.items())),
                "expressions_with_division": with_div,
                "expressions_with_divide_by_1": with_div1,
                "pct_of_division_expressions_with_divide_by_1":
                    (100.0 * with_div1 / with_div) if with_div else 0.0,
            }
            study_div += divs; study_q += quots
            n_expr_with_div += with_div; n_expr_with_div1 += with_div1; n_expr_total += len(items)
            print(f"  {f:<16} div-nodes={tot:<6} divisor==1: {divs[1]:<5} ({frac1:5.1f}%)   "
                  f"expr with div: {with_div:<5} of which >=1 divide-by-1: {with_div1:<5} "
                  f"({(100.0*with_div1/with_div) if with_div else 0:5.1f}%)")
        tot = sum(study_div.values())
        print(f"\n  STUDY TOTAL: {tot} division nodes")
        print(f"    divisor == 1 : {study_div[1]} ({100.0*study_div[1]/tot:.1f}%)")
        print(f"    divisor distribution (top 12): "
              f"{ {k: v for k, v in sorted(study_div.items(), key=lambda kv: -kv[1])[:12]} }")
        print(f"    quotient |value| distribution (top 12): "
              f"{ {k: v for k, v in sorted(study_q.items(), key=lambda kv: -kv[1])[:12]} }")
        print(f"    expressions containing division: {n_expr_with_div} of {n_expr_total}; "
              f"with at least one divide-by-1: {n_expr_with_div1} "
              f"({100.0*n_expr_with_div1/n_expr_with_div:.1f}% of division expressions)\n")
        report[study]["totals"] = {
            "division_nodes": tot,
            "divisor_is_1": study_div[1],
            "divisor_is_1_pct": 100.0 * study_div[1] / tot,
            "divisor_distribution": dict(sorted(study_div.items())),
            "quotient_abs_distribution": dict(sorted(study_q.items())),
            "expressions_with_division": n_expr_with_div,
            "expressions_with_divide_by_1": n_expr_with_div1,
            "pct_division_expressions_with_divide_by_1": 100.0 * n_expr_with_div1 / n_expr_with_div,
        }
    Path("results_v2/divisor_distribution.json").write_text(json.dumps(report, indent=2))
    print("Wrote results_v2/divisor_distribution.json")


if __name__ == "__main__":
    main()
