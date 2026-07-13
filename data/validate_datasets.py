"""Standalone validator for datasets_v2/ output.

Loads every generated v2 file and asserts:
  - bucket counts match the file's own recorded metadata exactly
  - ops/depth are exact per bucket, checked against every sample in that
    bucket's slice of the data (not just the recorded count)
  - zero cross-split duplicate expression strings across all files checked
  - operator distribution is present and internally consistent
  - the v2 metadata header (study, split, buckets, seed, generator_commit,
    timestamp, operator_distribution_actual) is present

Prints a pass/fail table and exits non-zero if anything fails. Read-only:
does not modify any file under datasets_v2/.
"""
import json
import sys
from pathlib import Path
from collections import Counter

REQUIRED_META_KEYS = {
    'study', 'split', 'buckets', 'seed', 'generator_commit',
    'timestamp', 'operator_distribution_actual', 'data',
}


def find_v2_files(root: Path):
    return sorted(root.glob("*/*.json"))


def validate_file(path: Path, seen_expressions: dict) -> list:
    """Returns a list of (check_name, passed: bool, detail: str) tuples."""
    checks = []

    try:
        payload = json.loads(path.read_text())
    except Exception as e:
        return [("json_parse", False, f"failed to parse: {e}")]

    missing_keys = REQUIRED_META_KEYS - set(payload.keys())
    checks.append((
        "metadata_present", not missing_keys,
        "all required keys present" if not missing_keys else f"missing keys: {sorted(missing_keys)}"
    ))
    if missing_keys:
        return checks  # can't do the rest without the data key

    data = payload['data']
    buckets = payload['buckets']

    # bucket counts exact
    recorded_total = sum(b['count'] for b in buckets)
    checks.append((
        "bucket_counts_sum_matches_data_len",
        recorded_total == len(data),
        f"sum(bucket counts)={recorded_total} vs len(data)={len(data)}"
    ))

    # slice the flat data list into buckets in the order buckets are recorded,
    # by consuming len(data) in bucket['count']-sized chunks
    offset = 0
    all_ops_depth_ok = True
    bucket_detail = []
    for b in buckets:
        n = b['count']
        slice_ = data[offset:offset + n]
        offset += n
        bad = [s for s in slice_ if s.get('num_operations') != b['ops'] or s.get('depth') != b['depth']]
        ok = len(bad) == 0 and len(slice_) == n
        all_ops_depth_ok = all_ops_depth_ok and ok
        bucket_detail.append(
            f"ops={b['ops']}/depth={b['depth']}/shape={b['shape']}: "
            f"{len(slice_)}/{n} samples, {len(bad)} mismatched"
        )
    checks.append((
        "bucket_ops_depth_exact", all_ops_depth_ok, "; ".join(bucket_detail)
    ))

    # cross-split dedup: check every expression in this file against the
    # global seen_expressions dict (expression -> first file it appeared in)
    collisions = []
    for s in data:
        expr = s.get('expression')
        if expr in seen_expressions and seen_expressions[expr] != str(path):
            collisions.append((expr, seen_expressions[expr]))
        else:
            seen_expressions[expr] = str(path)
    checks.append((
        "cross_split_dedup", len(collisions) == 0,
        "no collisions" if not collisions
        else f"{len(collisions)} collisions, e.g. {collisions[:3]}"
    ))

    # operator distribution reported and consistent with recomputation from data
    recomputed = Counter()
    for s in data:
        for op, c in s.get('operator_counts', {}).items():
            recomputed[op] += c
    reported = payload['operator_distribution_actual']['counts']
    op_match = all(recomputed.get(op, 0) == reported.get(op, 0) for op in ('+', '-', '*', '/'))
    checks.append((
        "operator_distribution_consistent", op_match,
        f"recomputed={dict(recomputed)} vs reported={reported}"
    ))

    return checks


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets_v2")
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        sys.exit(1)

    files = find_v2_files(root)
    if not files:
        print(f"ERROR: no *.json files found under {root}/*/")
        sys.exit(1)

    seen_expressions: dict = {}
    all_passed = True
    rows = []

    for path in files:
        checks = validate_file(path, seen_expressions)
        file_passed = all(passed for _, passed, _ in checks)
        all_passed = all_passed and file_passed
        rows.append((path, checks, file_passed))

    print("=" * 100)
    print("DATASET V2 VALIDATION REPORT")
    print("=" * 100)
    for path, checks, file_passed in rows:
        status = "PASS" if file_passed else "FAIL"
        print(f"\n[{status}] {path}")
        for name, passed, detail in checks:
            mark = "  ok " if passed else " FAIL"
            print(f"  [{mark}] {name}: {detail}")

    print("\n" + "=" * 100)
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'} ({len(files)} files checked, "
          f"{len(seen_expressions)} unique expressions across all files)")
    print("=" * 100)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
