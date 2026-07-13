import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

SEP = "=" * 60

from data.tree import (
    ExpressionTreeNode,
    create_leaf,
    create_operator_node,
    create_division_d1,
    GerationError,
    tree_statistics,
)

# Config
Max_Intermediate_ABS = 10000
Max_Results_ABS = 10000
Max_Tokens = 120
Max_Sample_attempts = 500
Output_Dir = Path("datasets")

VALID_TREE_SHAPES = {"balanced", "chain", "mixed"}


def _resolve_chain_direction() -> str:
    # Pick a skew direction for a chain tree. Resolved once per sample (see
    # generate_controlled_tree) so the whole tree skews consistently one way,
    # not per-node (which would zigzag instead of forming a real chain).
    return random.choice(["left", "right"])


# Tree generation with controlled properties
def generate_controlled_tree(num_ops: int, depth_limit: int, tree_shape: str = "balanced",
                              _chain_dir: Optional[str] = None) -> ExpressionTreeNode:
    # Recursively generate a controlled binary expression tree.

    if tree_shape not in VALID_TREE_SHAPES:
        raise ValueError(
            f"Unknown tree_shape: {tree_shape!r}. Must be one of {sorted(VALID_TREE_SHAPES)}."
        )

    if num_ops == 0 or depth_limit <= 0:
        # leaf node - draw a non zero integer
        val = random.randint(1, 20)
        return create_leaf(val)

    if tree_shape == "mixed":
        # Each node independently resolves to balanced or chain, so a single
        # "mixed" tree can contain both kinds of substructure.
        node_shape = random.choice(["balanced", "chain"])
    else:
        node_shape = tree_shape

    if node_shape == "chain":
        # Chain direction is resolved once per sample and threaded through
        # recursive calls (see below) so a pure "chain" tree skews one
        # consistent way; under "mixed", each chain node picks its own
        # direction independently since _chain_dir is not threaded there.
        direction = _chain_dir if _chain_dir is not None else _resolve_chain_direction()
        if direction == "left":
            left_ops = num_ops - 1
            right_ops = 0
        else:
            left_ops = 0
            right_ops = num_ops - 1
    else:  # balanced
        direction = None
        left_ops = num_ops // 2
        right_ops = num_ops - 1 - left_ops  # -1 accounts for root operator

    pass_dir = direction if tree_shape == "chain" else None

    left_child = generate_controlled_tree(left_ops, depth_limit - 1, tree_shape, _chain_dir=pass_dir)
    right_child = generate_controlled_tree(right_ops, depth_limit - 1, tree_shape, _chain_dir=pass_dir)

    # Select an operator
    operation = random.choice(['+', '-', '*', '/'])

    return create_operator_node(operation, left_child, right_child)

def enforce_d1(tree: ExpressionTreeNode) -> None:
    # Enforce D1 division constraints on the tree.
    
    if tree.is_leaf:
        return
    
    # Enforce on children first
    if tree.left:
        enforce_d1(tree.left)
    if tree.right:
        enforce_d1(tree.right)

    if tree.operator == '/':
        # Apply D1 constraints
        left_value = tree.left.evaluate() if tree.left else 0

        # get a valid (divider, quotient) pair
        divider, quotient = create_division_d1(
            left_value,
            max_intermediate = Max_Intermediate_ABS,
            max_result = Max_Results_ABS
            )

        # Rebuild the right child to reflect the new divider
        tree.right = create_leaf(divider)

        # update the node value stored for consistency
        tree.value = quotient
    
# Sample generation loop
def generate_sample(num_ops: int, depth_limit: int, seed_id: int, tree_shape: str = "balanced") -> Dict[str, Any]:
    
    # Try generating a valid sample within max attempts
    for attempt in range(Max_Sample_attempts):
        try:
            # Generate tree
            tree = generate_controlled_tree(num_ops, depth_limit, tree_shape)

            # Enforce D1 constraints
            enforce_d1(tree)

            # Evaluate tree
            result = tree.evaluate()
            if abs(result) > Max_Results_ABS:
                continue

            # Render tree to string (fully parenthesized, Regime P)
            expression_str = tree.to_string(parenthesis=True)

            # Check token length
            if len(expression_str.replace(" ", "")) > Max_Tokens:
                continue

            # Collect statistics
            stats = tree_statistics(tree, seed_id=seed_id, include_operator_counts=False)

            stats['expression'] = expression_str
            stats['result'] = result

            stats['input'] = expression_str
            stats['output'] = str(result)

            return stats
        
        except (GerationError, ValueError, ZeroDivisionError):
            # Retry on generation errors
            continue
    
    # If all attempts fail, raise an error
    raise GerationError(f"Failed to generate valid sample after {Max_Sample_attempts} attempts.")


def generate_controlled_dataset(num_samples: int, num_ops_range: tuple, depth_limit: int,
                                 seed: int = None, tree_shape: str = "balanced") -> List[Dict[str, Any]]:
    # Generate a controlled dataset with specified properties.

    if tree_shape not in VALID_TREE_SHAPES:
        raise ValueError(
            f"Unknown tree_shape: {tree_shape!r}. Must be one of {sorted(VALID_TREE_SHAPES)}."
        )

    if seed is not None:
        random.seed(seed)

    min_ops, max_ops = num_ops_range
    dataset: List[Dict[str, Any]] = []

    print(f"Generating {num_samples} controlled samples...")
    print(f"Operations range: {min_ops}-{max_ops}, Depth limit: {depth_limit}, Tree shape: {tree_shape}")
    print("-" * 60)

    for i in range(num_samples):
        num_ops = random.randint(min_ops, max_ops)

        try:
            sample = generate_sample(num_ops, depth_limit, seed_id=i, tree_shape=tree_shape)
            dataset.append(sample)

            if (i + 1) % 100 == 0:
                print(f"✓ Generated {i + 1}/{num_samples} samples")

        except GerationError as e:
            print(f"✗ Failed to generate sample {i}: {e}")
            continue

    print(f"\nSuccessfully generated {len(dataset)}/{num_samples} samples")
    return dataset


def generate_bucket(n: int, num_ops: int, depth: int, shape: str, seed: int = None,
                     seen_expressions: Optional[Set[str]] = None) -> Tuple[List[Dict[str, Any]], float]:
    """Generate exactly n samples that, after D1 repair, have exactly `num_ops`
    operations and exactly `depth` depth. Generate-and-filter: any sample whose
    post-D1 ops/depth drifted from the target (D1 repair can collapse a
    non-leaf right child of a division node down to a single leaf, silently
    reducing both ops and depth) is rejected and retried, as is any sample
    whose expression string collides with `seen_expressions` (cross-split
    dedup) if that set is provided.

    Returns (samples, rejection_rate) where rejection_rate = rejected / total_attempts.
    """
    if shape not in VALID_TREE_SHAPES:
        raise ValueError(
            f"Unknown tree_shape: {shape!r}. Must be one of {sorted(VALID_TREE_SHAPES)}."
        )
    if seed is not None:
        random.seed(seed)

    samples: List[Dict[str, Any]] = []
    attempts = 0
    rejections = 0
    seed_id_counter = 0
    # Generous global cap: bucket-level attempts can be much higher than a
    # single generate_sample's retry budget, since structural (ops, depth)
    # mismatches - not just D1/magnitude/length failures - can also reject.
    max_total_attempts = max(Max_Sample_attempts * n, 5000)

    while len(samples) < n:
        if attempts >= max_total_attempts:
            raise GerationError(
                f"generate_bucket: failed to reach n={n} samples for "
                f"num_ops={num_ops}, depth={depth}, shape={shape!r} after "
                f"{attempts} attempts ({rejections} rejected, "
                f"{len(samples)} accepted)."
            )
        attempts += 1
        try:
            # depth_limit=depth caps the tree at the target depth (depth_limit
            # is a ceiling, never a target - see generate_controlled_tree);
            # whether the tree actually reaches that depth is checked below.
            tree = generate_controlled_tree(num_ops, depth, shape)
            enforce_d1(tree)

            result = tree.evaluate()
            if abs(result) > Max_Results_ABS:
                rejections += 1
                continue

            expression_str = tree.to_string(parenthesis=True)
            if len(expression_str.replace(" ", "")) > Max_Tokens:
                rejections += 1
                continue

            actual_depth = tree.get_depth()
            actual_ops = tree.count_operations()
            if actual_depth != depth or actual_ops != num_ops:
                rejections += 1
                continue

            if seen_expressions is not None and expression_str in seen_expressions:
                rejections += 1
                continue

            stats = tree_statistics(tree, seed_id=seed_id_counter, include_operator_counts=True)
            stats['expression'] = expression_str
            stats['result'] = result
            stats['input'] = expression_str
            stats['output'] = str(result)
            samples.append(stats)
            seed_id_counter += 1
            if seen_expressions is not None:
                seen_expressions.add(expression_str)

        except (GerationError, ValueError, ZeroDivisionError):
            rejections += 1
            continue

    rejection_rate = rejections / attempts if attempts else 0.0
    return samples, rejection_rate


def save_dataset(data: List[Dict[str, Any]], output_path: str) -> None:
    # Save dataset to JSON file.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'data': data}, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset to {output_path}")


def _get_generator_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _aggregate_operator_distribution(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = {'+': 0, '-': 0, '*': 0, '/': 0}
    for sample in data:
        counts = sample.get('operator_counts')
        if not counts:
            continue
        for op in totals:
            totals[op] += counts.get(op, 0)
    grand_total = sum(totals.values())
    fractions = {
        op: (count / grand_total if grand_total else 0.0)
        for op, count in totals.items()
    }
    return {'counts': totals, 'fractions': fractions}


def save_dataset_v2(data: List[Dict[str, Any]], output_path: str, *, study: str, split: str,
                     buckets: List[Dict[str, Any]], seed: Any) -> None:
    """Save a dataset with the v2 metadata header. `data` still lives under the
    same top-level "data" key as the original save_dataset format (all
    existing consumers - main.py's evaluate_model, data/dataloader.py's
    load_data/get_dataloaders_file, and the notebook's json.load(...)["data"]
    - read that key directly and ignore unknown sibling keys, so this is a
    backward-compatible superset of the old schema, not a breaking change)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'study': study,
        'split': split,
        'buckets': buckets,
        'seed': seed,
        'generator_commit': _get_generator_commit(),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'operator_distribution_actual': _aggregate_operator_distribution(data),
        'data': data,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved dataset (v2) to {output_path}")


def generate_study_datasets_v2(output_root: str = "datasets_v2") -> Dict[str, Any]:
    """Generate Study 1 and Study 2 datasets to the locked bucket spec.

    This is a NEW function alongside the original generate_study_datasets()
    (in main.py) rather than a replacement - it writes to `output_root`
    (default datasets_v2/), leaving datasets/ and the original generator
    untouched, and it performs cross-split deduplication (a single
    seen_expressions set shared across every bucket in every split generated
    by this call, the strictest reading of "no string appears in two
    splits") and per-bucket rejection-rate tracking that the original
    generate_study_datasets() does not do.

    Returns a report dict: {study: {split: {buckets: [...], rejection info}}}
    plus total elapsed generation time, for the caller to print/inspect.
    """
    import time
    t_start = time.time()

    out_root = Path(output_root)
    seen_expressions: Set[str] = set()
    report: Dict[str, Any] = {}

    def build_split(study: str, split: str, bucket_specs: List[Dict[str, Any]], out_file: str) -> Dict[str, Any]:
        all_samples: List[Dict[str, Any]] = []
        bucket_reports = []
        for spec in bucket_specs:
            samples, rejection_rate = generate_bucket(
                n=spec['count'], num_ops=spec['ops'], depth=spec['depth'],
                shape=spec['shape'], seed=spec['seed'], seen_expressions=seen_expressions,
            )
            all_samples.extend(samples)
            bucket_reports.append({
                'ops': spec['ops'], 'depth': spec['depth'], 'shape': spec['shape'],
                'count': len(samples), 'rejection_rate': rejection_rate,
            })
            print(f"  [{study}/{split}] ops={spec['ops']} depth={spec['depth']} "
                  f"shape={spec['shape']}: {len(samples)} samples, "
                  f"rejection_rate={rejection_rate:.4f}")

        study_dir = out_root / study
        study_dir.mkdir(parents=True, exist_ok=True)
        save_dataset_v2(
            all_samples, str(study_dir / out_file),
            study=study, split=split, buckets=bucket_reports,
            seed=[spec['seed'] for spec in bucket_specs],
        )
        return {'buckets': bucket_reports, 'total': len(all_samples), 'file': str(study_dir / out_file)}

    print("\n" + SEP)
    print("GENERATING CONTROLLED DATASETS (v2, bucketed)")
    print(SEP)

    # ---- Study 1 ----
    print("\nGenerating Study 1 (Length Generalization) - v2 buckets...")
    report.setdefault('study1', {})
    report['study1']['train'] = build_split(
        'study1', 'train',
        [
            {'ops': 2, 'depth': 2, 'shape': 'balanced', 'count': 2666, 'seed': 42 + 0},
            {'ops': 3, 'depth': 2, 'shape': 'balanced', 'count': 2667, 'seed': 42 + 1},
            {'ops': 3, 'depth': 3, 'shape': 'chain',    'count': 2667, 'seed': 42 + 2},
        ],
        'train.json',
    )
    report['study1']['val'] = build_split(
        'study1', 'val',
        [
            {'ops': 2, 'depth': 2, 'shape': 'balanced', 'count': 333, 'seed': 4242 + 0},
            {'ops': 3, 'depth': 2, 'shape': 'balanced', 'count': 333, 'seed': 4242 + 1},
            {'ops': 3, 'depth': 3, 'shape': 'chain',    'count': 334, 'seed': 4242 + 2},
        ],
        'val.json',
    )
    for op_count in (4, 5, 6, 7):
        report['study1'][f'ood_ops{op_count}'] = build_split(
            'study1', f'ood_ops{op_count}',
            [{'ops': op_count, 'depth': 3, 'shape': 'balanced', 'count': 1000, 'seed': 424242 + op_count}],
            f'ood_ops{op_count}.json',
        )

    # ---- Study 2 ----
    print("\nGenerating Study 2 (Depth Generalization) - v2 buckets...")
    report.setdefault('study2', {})
    report['study2']['train'] = build_split(
        'study2', 'train',
        [{'ops': 3, 'depth': 2, 'shape': 'balanced', 'count': 8000, 'seed': 43}],
        'train.json',
    )
    report['study2']['val'] = build_split(
        'study2', 'val',
        [{'ops': 3, 'depth': 2, 'shape': 'balanced', 'count': 1000, 'seed': 4343}],
        'val.json',
    )
    report['study2']['ood'] = build_split(
        'study2', 'ood',
        [{'ops': 3, 'depth': 3, 'shape': 'chain', 'count': 1000, 'seed': 434343}],
        'ood.json',
    )

    elapsed = time.time() - t_start
    report['_elapsed_seconds'] = elapsed
    report['_total_unique_expressions'] = len(seen_expressions)

    print("\n" + SEP)
    print(f"DATASET GENERATION (v2) COMPLETE in {elapsed:.1f}s")
    print(SEP)

    return report


def generate_verification_samples(num_samples: int = 40, seed: int = 42) -> List[Dict[str, Any]]:
    # Generate verification samples for professor review.
    random.seed(seed)
    samples = []
    
    # Study 1 Training Distribution: ops {2,3}, depth≤3
    print("\nGenerating Study 1 TRAINING samples (ops 2-3, depth≤3)...")
    for i in range(10):
        num_ops = random.choice([2, 3])
        try:
            sample = generate_sample(num_ops, depth_limit=3, seed_id=i)
            # Add missing keys
            sample['study'] = 'Study1_Train'
            sample['num_ops'] = sample.get('num_operations', num_ops)  # Add num_ops
            samples.append(sample)
        except GerationError as e:
            print(f"  Warning: Failed sample {i}: {e}")
    
    # Study 1 OOD: ops {4,5,6,7}, depth≤3
    print("Generating Study 1 OOD samples (ops 4-7, depth≤3)...")
    ops_list = [4, 5, 6, 7]
    for i in range(10):
        num_ops = ops_list[i % 4]
        try:
            sample = generate_sample(num_ops, depth_limit=3, seed_id=100 + i)
            # Add missing keys
            sample['study'] = f'Study1_OOD_ops{num_ops}'
            sample['num_ops'] = sample.get('num_operations', num_ops)
            samples.append(sample)
        except GerationError as e:
            print(f"  Warning: Failed sample {100+i}: {e}")
    
    # Study 2 Training: ops=3, depth=2
    print("Generating Study 2 TRAINING samples (ops=3, depth=2)...")
    for i in range(10):
        try:
            sample = generate_sample(num_ops=3, depth_limit=2, seed_id=200 + i, tree_shape='balanced')
            sample['study'] = 'Study2_Train'
            sample['num_ops'] = sample.get('num_operations', 3)
            samples.append(sample)
        except GerationError as e:
            print(f"  Warning: Failed sample {200+i}: {e}")
    
    # Study 2 OOD: ops=3, depth=3
    print("Generating Study 2 OOD samples (ops=3, depth=3)...")
    for i in range(10):
        try:
            sample = generate_sample(num_ops=3, depth_limit=3, seed_id=300 + i, tree_shape='chain') 
            sample['study'] = 'Study2_OOD_depth3'
            sample['num_ops'] = sample.get('num_operations', 3)
            samples.append(sample)
        except GerationError as e:
            print(f"  Warning: Failed sample {300+i}: {e}")
    
    print(f"✓ Generated {len(samples)} verification samples")
    return samples

def print_verification_samples(samples: List[Dict[str, Any]]):
    # Print verification samples for professor review.
    print("\n" + SEP)
    print("VERIFICATION SAMPLES FOR PROFESSOR REVIEW")
    print(SEP)
    print(f"\nParameters:")
    print(f"  - Operand range: 1-20 (positive integers only)")
    print(f"  - Operation distribution: 25% each (+, -, *, /)")
    print(f"  - Magnitude caps: |intermediate| ≤ 10,000, |result| ≤ 10,000")
    print(f"  - Parenthesization: Regime P (fully parenthesized)")
    print(f"  - Division: D1 constraint (integer division, no remainders)")
    print("\n" + SEP)
    
    # Group samples by study
    study1_train = [s for s in samples if s.get('study') == 'Study1_Train']
    study1_ood = [s for s in samples if 'Study1_OOD' in s.get('study', '')]
    study2_train = [s for s in samples if s.get('study') == 'Study2_Train']
    study2_ood = [s for s in samples if s.get('study') == 'Study2_OOD_depth3']
    
    # Print Study 1 Training
    print(f"\nSTUDY 1 TRAINING (ops 2-3, depth≤3): {len(study1_train)} samples")
    print("-" + SEP)
    for idx, s in enumerate(study1_train[:5], 1):
        print(f"\nSample {idx}:")
        print(f"  Expression: {s.get('expression', s.get('input', 'N/A'))}")
        print(f"  Result: {s.get('result', s.get('output', 'N/A'))}")
        print(f"  Num_ops: {s.get('num_ops', s.get('num_operations', 'N/A'))}, Depth: {s.get('depth', 'N/A')}")
        print(f"  Intermediate_max: {s.get('intermediate_max', 'N/A')}")
    
    # Print Study 1 OOD
    print(f"\n{SEP}")
    print(f"\nSTUDY 1 OOD (ops 4-7, depth≤3): {len(study1_ood)} samples")
    print("-" + SEP)
    for idx, s in enumerate(study1_ood[:5], 1):
        print(f"\nSample {idx}:")
        print(f"  Expression: {s.get('expression', s.get('input', 'N/A'))}")
        print(f"  Result: {s.get('result', s.get('output', 'N/A'))}")
        print(f"  Num_ops: {s.get('num_ops', s.get('num_operations', 'N/A'))}, Depth: {s.get('depth', 'N/A')}")
        print(f"  Intermediate_max: {s.get('intermediate_max', 'N/A')}")
    
    # Print Study 2 Training
    print(f"\n{SEP}")
    print(f"\nSTUDY 2 TRAINING (ops=3, depth=2): {len(study2_train)} samples")
    print("-" + SEP)
    for idx, s in enumerate(study2_train[:5], 1):
        print(f"\nSample {idx}:")
        print(f"  Expression: {s.get('expression', s.get('input', 'N/A'))}")
        print(f"  Result: {s.get('result', s.get('output', 'N/A'))}")
        print(f"  Num_ops: {s.get('num_ops', s.get('num_operations', 'N/A'))}, Depth: {s.get('depth', 'N/A')}")
        print(f"  Intermediate_max: {s.get('intermediate_max', 'N/A')}")
    
    # Print Study 2 OOD
    print(f"\n{SEP}")
    print(f"\nSTUDY 2 OOD (ops=3, depth=3): {len(study2_ood)} samples")
    print("-" + SEP)
    for idx, s in enumerate(study2_ood[:5], 1):
        print(f"\nSample {idx}:")
        print(f"  Expression: {s.get('expression', s.get('input', 'N/A'))}")
        print(f"  Result: {s.get('result', s.get('output', 'N/A'))}")
        print(f"  Num_ops: {s.get('num_ops', s.get('num_operations', 'N/A'))}, Depth: {s.get('depth', 'N/A')}")
        print(f"  Intermediate_max: {s.get('intermediate_max', 'N/A')}")
    
    print("\n" + SEP)
    print(f"Total samples: {len(samples)}")
    print(f"  Study 1 Train: {len(study1_train)}")
    print(f"  Study 1 OOD: {len(study1_ood)}")
    print(f"  Study 2 Train: {len(study2_train)}")
    print(f"  Study 2 OOD: {len(study2_ood)}")
    print(SEP)

def save_verification_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    # Save verification samples to JSON file.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'data': samples}, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Verification samples saved to: {output_path}")

def main():
    # Main entry point for dataset generation.
    parser = argparse.ArgumentParser(description='Generate verification samples')
    parser.add_argument('--num-samples', type=int, default=40,  # Changed from 30
                        help='Number of samples (default: 40)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='verification_samples.json',
                        help='Output file (default: verification_samples.json)')
    
    args = parser.parse_args()
    
    print("\n" + SEP)
    print("CONTROLLED DATASET GENERATION")
    print(SEP)
    print(f"Samples: {args.num_samples}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print(SEP)
    
    samples = generate_verification_samples(num_samples=args.num_samples, seed=args.seed)
    print_verification_samples(samples)
    save_verification_samples(samples, args.output)
    
    print("\nVERIFICATION COMPLETE!")
    
    print("\n" + SEP)
    print("DATASET GENERATION COMPLETE!")
    print(SEP)

if __name__ == "__main__":
    main()