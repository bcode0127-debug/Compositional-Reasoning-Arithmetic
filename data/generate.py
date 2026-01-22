import random
import ast
import operator as op
import json
import os
from typing import List, Dict, Any, Optional, Union

# Define which operators are allowed in our expressions
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}

def _safe_eval(node: ast.AST) -> Union[int, float]:

    # If node is an expression, evaluate its body
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_type = type(node.op)

        # check if operator is allowed
        if op_type not in _ALLOWED_OPS:
            raise ValueError(f"Unsupported operator: {op_type}")
        
        # check for fiviion by zero
        if op_type is ast.Div and right == 0:
            raise ZeroDivisionError("division by zero")
        
        # apply operator to left nad right values
        return _ALLOWED_OPS[op_type](left, right)
    
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only int/float constnats allowed")
    # Older Python versions use ast.Num for numbers
    if isinstance(node, ast.Num):
        return node.n
    
    # If we encounter any other type of node, raise an error
    raise ValueError(f"Unsupported AST node: {type(node)}")

def safe_eval_expr(expr: str) -> Union[int, float]:

    # Parse the expression string into an AST (Abstract Syntax Tree)
    parsed = ast.parse(expr, mode="eval")
    # Evaluate the AST safely using our custom evaluator
    return _safe_eval(parsed)

def save_json(data: Any, path: str) -> None:
    
    # Open a file for writing
    with open(path, "w", encoding="utf-8") as f:
        # Write data as formatted JSON (indent=2 makes it readable)
        json.dump(data, f, indent=2, ensure_ascii=False)

# LSTM- Friendly
# Operations: +, -, *
def generate_lvl1(
        num_samples: int = 1000,
        max_num: int = 15,
        seed: Optional[int] = None,
)-> List[Dict[str, Any]]:
    # Level 1: Addition, subtraction, multiplication only.
    if seed is not None:
        random.seed(seed)

    operators = ['+', '-', '*']
    dataset: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    max_attempts = num_samples * 10  

    while len(dataset) < num_samples and attempts < max_attempts:
        num1 = random.randint(1, max_num)
        num2 = random.randint(1, max_num)
        num3 = random.randint(1, max_num)
        
        # Randomly choose an operator
        op1 = random.choice(operators)
        op2 = random.choice(operators)

        # Create the expression string
        expr = f"{num1} {op1} {num2} {op2} {num3}"

        if expr in seen:
            attempts += 1
            continue

        try:
            # Safely evaluate the expression to get the answer
            val = safe_eval_expr(expr)

            if isinstance(val, float) and val.is_integer():
                val = int(val)   

            dataset.append({
                "input": expr,
                "output": str(val),
                "level": "lvl1"
            }) 
            seen.add(expr)


        except (ZeroDivisionError):
            attempts += 1
            continue
        except Exception as e:
            print(f"Unexpected error evaluating expression '{expr}': {e}")
            attempts += 1
            continue
        
        attempts += 1

    if len(dataset) < num_samples:
        print(f"Warning: Only generated {len(dataset)} unique valid samples out of requested {num_samples}.")

    return dataset


# Medium Complexity
# Operations: +, -, *, /

def generate_lvl2(
        num_samples: int = 1000,
        max_num: int = 15,
        seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    # Level 2: All four basic operations.

    if seed is not None:
        random.seed(seed + 1000)

    operators = ['+', '-', '*', '/']
    dataset: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    max_attempts = num_samples * 10  

    # Generate num_samples examples
    while len(dataset) < num_samples and  attempts < max_attempts:
        num1 = random.randint(1, max_num)
        num2 = random.randint(1, max_num)
        num3 = random.randint(1, max_num)
        
        # Randomly choose an operator
        op1 = random.choice(operators)
        op2 = random.choice(operators)

        # Create the expression string
        expr = f"{num1} {op1} {num2} {op2} {num3}"

        if expr in seen:
            attempts += 1
            continue

        try:
            # Safely evaluate the expression to get the answer
            val = safe_eval_expr(expr)

            if isinstance(val, float):
                val = round(val, 2)  # Round to avoid floating-point precision issues
                if val.is_integer():
                   val = int(val)
                else:
                    attempts += 1
                    continue   

            dataset.append({
                "input": expr,
                "output": str(val),
                "level": "lvl2"
            }) 
            seen.add(expr)
        except (ZeroDivisionError):
            attempts += 1
            continue

        except Exception as e:
            print(f"Unexpected error evaluating expression '{expr}': {e}")
            attempts += 1
            continue

        attempts += 1

    if len(dataset) < num_samples:
        print(f"Warning: Only generated {len(dataset)} unique valid samples out of requested {num_samples}.")

    return dataset

# Full Complexity
# Operations: +, -, *, /, parentheses
def generate_lvl3(
        num_samples: int = 1000,
        max_num: int = 15,
        seed: Optional[int] = None,
        parentheses_prob: float = 0.7,
) -> List[Dict[str, Any]]:
    # Level 3: All operations with parentheses.

    if seed is not None:
        random.seed(seed + 2000)

    operators = ['+', '-', '*', '/']
    dataset: List[Dict[str, Any]] = []
    seen = set()
    attempts = 0
    max_attempts = num_samples * 10  

    # Generate num_samples examples
    while len(dataset) < num_samples and attempts < max_attempts:
        num1 = random.randint(1, max_num)
        num2 = random.randint(1, max_num)
        num3 = random.randint(1, max_num)
        
        # Randomly choose an operator
        op1 = random.choice(operators)
        op2 = random.choice(operators)

        # Randomly decide to add parentheses
        if random.random() < parentheses_prob:

            if random.random() < 0.5:
               expr = f"({num1} {op1} {num2}) {op2} {num3}"
            else:
                expr = f"{num1} {op1} ({num2} {op2} {num3})"
        else:
            expr = f"{num1} {op1} {num2} {op2} {num3}"        
        
        if expr in seen:
            attempts += 1
            continue

        try:
            # Safely evaluate the expression to get the answer
            val = safe_eval_expr(expr)

            if isinstance(val, float):
                val = round(val, 2)  # Round to avoid floating-point precision issues
                if val.is_integer():
                   val = int(val)
                else:
                    attempts += 1
                    continue   

            dataset.append({
                "input": expr,
                "output": str(val),
                "level": "lvl3"
            }) 
            seen.add(expr)

        except (ZeroDivisionError):
            attempts += 1
            continue
        except Exception as e:
            print(f"Unexpected error evaluating expression '{expr}': {e}")
            attempts += 1
            continue

        attempts += 1
    if len(dataset) < num_samples:
        print(f"Warning: Only generated {len(dataset)} unique valid samples out of requested {num_samples}.")

    return dataset
