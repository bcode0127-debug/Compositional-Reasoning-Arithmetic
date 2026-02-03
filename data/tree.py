from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


class GerationError(Exception):
    """Exception raised for errors during generation."""
    pass


@dataclass
class ExpressionTreeNode:
    # Binary tree node for representing  mathematical expressions.
    
    value: int
    operator: Optional[str]  = None
    left: Optional['ExpressionTreeNode'] = None
    right: Optional['ExpressionTreeNode'] = None
    
    @property
    def is_leaf(self) -> bool:
        # Check if the node is a leaf node (no children).
        return self.operator is None 
    
    def get_depth(self) -> int:
        # check if this is a leaf node
    
        if self.is_leaf:
            return 0

        left_depth = self.left.get_depth() if self.left else 0
        right_depth = self.right.get_depth() if self.right else 0

        return 1 + max(left_depth, right_depth)

    def count_operations(self) -> int:
        # Count the number of operations in the tree.
        
        if self.is_leaf:
            return 0
        
        left_count = self.left.count_operations() if self.left else 0
        right_count = self.right.count_operations() if self.right else 0
        
        return 1 + left_count + right_count
    
    def evaluate(self) -> int:
        # Evaluate the expression represented by the tree and return the integer result.
        
        if self.is_leaf:
            return self.value
        
        left_value = self.left.evaluate() if self.left else 0
        right_value = self.right.evaluate() if self.right else 0

        if self.operator == '+':
            return left_value + right_value
        elif self.operator == '-':
            return left_value - right_value
        elif self.operator == '*':
            return left_value * right_value
        elif self.operator == '/':
            if right_value == 0:
                raise ValueError("Division by zero.")
            return left_value // right_value  # integer division
        else:
            raise ValueError(f"Unsupported operator: {self.operator}")
    
    def to_string(self, parenthesis: bool = True) -> str:
        # Convert the expression tree to a string representation.
        
        if self.is_leaf:
            return str(self.value)
        
        left_str = self.left.to_string(parenthesis) if self.left else "?"
        right_str = self.right.to_string(parenthesis) if self.right else "?"

        expr = f"{left_str} {self.operator} {right_str}"

        if parenthesis:
            expr = f"({expr})"
        
        return expr

    def __repr__(self) -> str:
        # String representation of the node for debugging.
        
        if self.is_leaf:
            return f"Leaf({self.value})"
        return f"ops({self.operator}, val={self.value}, depth={self.get_depth()})"
    
def create_leaf(value: int) -> ExpressionTreeNode:
    # Create a leaf node with the given value.

    if not isinstance(value, int):
        raise TypeError(f"Leaf value must be a int, got {type(value)}")

    return ExpressionTreeNode(value=value, left=None, right=None)

def create_operator_node(operator: str, left: ExpressionTreeNode, right: ExpressionTreeNode) -> ExpressionTreeNode:
    
    # Create an operator node, returns ExpressionTreeNode with computed integer value.
    
    # Compute the value based on the operator
    if operator not in {'+', '-', '*', '/'}:
        raise ValueError(f"Unsupported operator: {operator}")
    
    # Create the operator node
    if operator == '+':
        result = left.value + right.value
    elif operator == '-':
        result = left.value - right.value
    elif operator == '*':
        result = left.value * right.value
    elif operator == '/':
        if right.value == 0:
            raise ValueError("Division by zero is not allowed.")
        # check that division results in an integer
        if left.value % right.value != 0:
            raise ValueError("Division must result in an integer.")
        
        result = left.value // right.value # integer division

    return ExpressionTreeNode(operator=operator, value=result, left=left, right=right)


def create_division_d1(dividend: int, max_intermediate: int = 10000, max_result: int = 10000) -> Tuple[int, int]:
    # Create a (divisor, quotient) pair for D1 division given a dividend.

    if dividend == 0:
        # For 0, any non-zero divisor works, return quotient=0
        return (1, 0)
    
    abs_dividend = abs(dividend)
    
    # Find divisors of abs_dividend
    divisors = []
    for i in range(1, min(int(abs_dividend**0.5) + 1, max_intermediate + 1)):
        if abs_dividend % i == 0:
            divisors.append(i)
            if i != abs_dividend // i and abs_dividend // i <= max_intermediate:
                divisors.append(abs_dividend // i)
    
    if not divisors:
        raise GerationError(f"No valid divisors found for dividend {dividend}")
    
    # Try each divisor
    for divisor in sorted(divisors):
        quotient = dividend // divisor
        if abs(quotient) <= max_result:
            
            signed_divisor = divisor if dividend > 0 else -divisor
            return (signed_divisor, quotient)

    raise GerationError(f"No valid (divisor, quotient) pair found for dividend {dividend}")


def tree_statistics(tree: ExpressionTreeNode, seed_id: int = 0, include_operator_counts: bool = True) -> Dict[str, Any]:
    # Gather statistics about the expression tree.
    
    # Calculate intermediate max by evaluating all subtrees
    def get_max_intermediate(node: ExpressionTreeNode) -> int:
        """Recursively find the maximum absolute intermediate value."""
        if node.is_leaf:
            return abs(node.value)
        
        # Get max from children
        left_max = get_max_intermediate(node.left) if node.left else 0
        right_max = get_max_intermediate(node.right) if node.right else 0
        
        # Evaluate this node
        current_value = abs(node.evaluate())
        
        # Return the maximum of all intermediate values
        return max(current_value, left_max, right_max)
    
    stats: Dict[str, Any] = {
        'seed_id': seed_id,
        'depth': tree.get_depth(),
        'num_operations': tree.count_operations(),
        'result': tree.evaluate(),
        'intermediate_max': get_max_intermediate(tree),
    }
    
    if include_operator_counts:
        op_counts = {'+': 0, '-': 0, '*': 0, '/': 0}
        _count_operators(tree, op_counts)
        stats['operator_counts'] = op_counts
    
    return stats


def _count_operators(tree: ExpressionTreeNode, op_counts: Dict[str, int]) -> None:
    # Helper function to count operators in tree.
    if tree.is_leaf:
        return
    
    if tree.operator in op_counts:
        op_counts[tree.operator] += 1
    
    if tree.left:
        _count_operators(tree.left, op_counts)
    if tree.right:
        _count_operators(tree.right, op_counts)


