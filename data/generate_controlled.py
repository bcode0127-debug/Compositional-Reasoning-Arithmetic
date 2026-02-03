import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

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

# Tree generation with controlled properties
def generate_controlled_tree(num_ops: int, depth_limit: int, tree_shape: str = "balanced") -> ExpressionTreeNode:
    # Recursively generate a controlled binary expression tree.

    if num_ops == 0:
        # leaf node - draw a non zero integer 
        val = random.randint(1, 20)
        return create_leaf(val)
    
    if tree_shape == "chain":
        left_ops = num_ops - 1
        right_ops = 0
    else:
        left_ops = num_ops // 2
        right_ops = num_ops - 1 - left_ops # -1 accounts for root operator

    left_child = generate_controlled_tree(left_ops, depth_limit -1, tree_shape)
    right_child = generate_controlled_tree(right_ops, depth_limit -1, tree_shape)

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


def generate_controlled_dataset(num_samples: int, num_ops_range: tuple, depth_limit: int, seed: int = None) -> List[Dict[str, Any]]:
    # Generate a controlled dataset with specified properties.
     
    if seed is not None:
        random.seed(seed)
    
    min_ops, max_ops = num_ops_range
    dataset: List[Dict[str, Any]] = []
    
    print(f"Generating {num_samples} controlled samples...")
    print(f"Operations range: {min_ops}-{max_ops}, Depth limit: {depth_limit}")
    print("-" * 60)
    
    for i in range(num_samples):
        num_ops = random.randint(min_ops, max_ops)
        
        try:
            sample = generate_sample(num_ops, depth_limit, seed_id=i)
            dataset.append(sample)
            
            if (i + 1) % 100 == 0:
                print(f"✓ Generated {i + 1}/{num_samples} samples")
        
        except GerationError as e:
            print(f"✗ Failed to generate sample {i}: {e}")
            continue
    
    print(f"\nSuccessfully generated {len(dataset)}/{num_samples} samples")
    return dataset


def save_dataset(data: List[Dict[str, Any]], output_path: str) -> None:
    # Save dataset to JSON file.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'data': data}, f, indent=2, ensure_ascii=False)
    
    print(f"Saved dataset to {output_path}")

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
            sample = generate_sample(num_ops=3, depth_limit=3, seed_id=300 + i, tree_shape='chain')  #CHANGED
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
    
    print("\n✅ VERIFICATION COMPLETE!")
    
    print("\n" + SEP)
    print("✅ DATASET GENERATION COMPLETE!")
    print(SEP)

if __name__ == "__main__":
    main()