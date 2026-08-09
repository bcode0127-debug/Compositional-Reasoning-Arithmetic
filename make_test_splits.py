#!/usr/bin/env python3
"""Generate independent in-distribution TEST splits for Study 1 and Study 2.

Additive only: writes datasets_v2/study1/test.json and datasets_v2/study2/test.json
and touches nothing else. Bucket specs mirror each study's existing val split
exactly (same ops/depth/shape/counts), with fresh seeds, so the new files are a
second independent draw from the same in-distribution generator, not a reshuffle.

Cross-split dedup is enforced against every expression in all nine existing
datasets_v2 files AND against the other new file, via the single
`seen_expressions` set threaded through generate_bucket().
"""
import json
from pathlib import Path

from data.generate_controlled import generate_bucket, save_dataset_v2

DATA_ROOT = Path("datasets_v2")

# mirrors generate_study_datasets_v2()'s val specs, new seeds
STUDY1_TEST_BUCKETS = [
    {"ops": 2, "depth": 2, "shape": "balanced", "count": 333, "seed": 5252 + 0},
    {"ops": 3, "depth": 2, "shape": "balanced", "count": 333, "seed": 5252 + 1},
    {"ops": 3, "depth": 3, "shape": "chain",    "count": 334, "seed": 5252 + 2},
]
STUDY2_TEST_BUCKETS = [
    {"ops": 3, "depth": 2, "shape": "balanced", "count": 1000, "seed": 5353},
]


def load_existing_expressions():
    """Every expression already committed under datasets_v2/, so the new test
    splits can be deduped against all of them."""
    seen = set()
    per_file = {}
    for p in sorted(DATA_ROOT.glob("*/*.json")):
        if p.name == "test.json":
            continue  # don't seed from a previous run of this script
        data = json.loads(p.read_text())["data"]
        exprs = {s["input"] for s in data}
        per_file[str(p)] = len(exprs)
        seen |= exprs
    return seen, per_file


def build(study, buckets, seen):
    samples, reports = [], []
    for spec in buckets:
        got, rej = generate_bucket(
            n=spec["count"], num_ops=spec["ops"], depth=spec["depth"],
            shape=spec["shape"], seed=spec["seed"], seen_expressions=seen,
        )
        samples.extend(got)
        reports.append({
            "ops": spec["ops"], "depth": spec["depth"], "shape": spec["shape"],
            "count": len(got), "rejection_rate": rej,
        })
        print(f"  [{study}/test] ops={spec['ops']} depth={spec['depth']} "
              f"shape={spec['shape']}: {len(got)} samples, rejection_rate={rej:.4f}")
    out = DATA_ROOT / study / "test.json"
    save_dataset_v2(samples, str(out), study=study, split="test",
                    buckets=reports, seed=[s["seed"] for s in buckets])
    return samples, reports


if __name__ == "__main__":
    seen, per_file = load_existing_expressions()
    print(f"Seeded dedup set with {len(seen)} expressions from {len(per_file)} existing files:")
    for f, n in per_file.items():
        print(f"    {f}: {n}")
    print()

    s1, r1 = build("study1", STUDY1_TEST_BUCKETS, seen)
    s2, r2 = build("study2", STUDY2_TEST_BUCKETS, seen)

    print(f"\nGenerated study1/test.json: {len(s1)}, study2/test.json: {len(s2)}")
    print(f"dedup set grew to {len(seen)} expressions")
