"""
CP4 & CP5: Operation breakdown and failure onset tracing
Works on CPU, no GPU needed
"""

import json
import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenizer import create_tokenizer
from main import create_model, TRAINING_CONFIG
from experiments.config import CHECKPOINTS, ANALYSIS_CONFIG


class CompositionAnalyzer:
    # Analyze model failures by operation type and trace failure onset
    
    def __init__(self, model_type: str, study: str, device: str = 'cpu'):
        """
        Args:
            model_type: 'transformer' or 'lstm'
            study: 'study1' or 'study2'
            device: 'cpu' or 'cuda'
        """
        self.model_type = model_type
        self.study = study
        self.device = device
        
        # Load tokenizer
        self.tokenizer = create_tokenizer()
        
        # Load model
        ckpt_key = f'{model_type}_{study}'
        ckpt_path = CHECKPOINTS[ckpt_key]
        
        self.model = create_model(model_type, self.tokenizer.vocab_size)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(device).eval()
        
        self.max_input_len = TRAINING_CONFIG['max_input_len']
        self.max_output_len = TRAINING_CONFIG['max_output_len']
        
        print(f"Loaded {model_type} on {study}")
        print(f"   Checkpoint: {ckpt_path}")
        print(f"   Val accuracy: {ckpt['val_accuracy']:.2f}%")
    
    # CP4: Accuracy by number of operations
    def analyze_by_operation(self, dataset_path: str) -> Dict:
        # CP4: Break down accuracy by number of operations.
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        dataset = data if isinstance(data, list) else data.get('data', [])
        
        results = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        print(f"\n CP4: Analyzing {len(dataset)} examples...")
        
        with torch.no_grad():
            for idx, sample in enumerate(dataset):
                if idx % 100 == 0:
                    print(f"  Progress: {idx}/{len(dataset)}", end='\r')
                
                expr = sample.get("input", "")
                target = str(sample.get("output", ""))
                num_ops = sample.get("num_operations")
                
                if num_ops is None or not expr:
                    continue
                
                try:
                    pred = self._predict(expr).strip()
                    target = target.strip()
                    results[num_ops]['total'] += 1
                    
                    if pred == target:
                        results[num_ops]['correct'] += 1
                except Exception as e:
                    results[num_ops]['total'] += 1
        
        # Calculate and print results
        print(f"\n{'='*60}")
        print(f"CP4: ACCURACY BY NUMBER OF OPERATIONS")
        print(f"{'='*60}")
        
        for num_ops in sorted(results.keys()):
            total = results[num_ops]['total']
            correct = results[num_ops]['correct']
            accuracy = 100.0 * correct / total if total > 0 else 0.0
            print(f"{num_ops} operations: {accuracy:.1f}% ({correct}/{total})")
            results[num_ops]['accuracy'] = accuracy
        
        return dict(results)
    
    # CP5: Failure onset tracing
    def _decode_token_safe(self, token_id: int) -> str:
        # Safely decode a token, handling special tokens 
        # Special tokens: PAD=0, SOS=1, EOS=2
        if token_id == 0:
            return '<PAD>'
        elif token_id == 1:
            return '<SOS>'
        elif token_id == 2:
            return '<EOS>'
        else:
            decoded = self.tokenizer.decode([token_id]).strip()
            return decoded if decoded else f'<T{token_id}>'
    
    def trace_failure(self, expr: str, target: str) -> Dict:
        # Trace decoding step-by-step to find where model first goes wrong
        src_ids = self.tokenizer.encode(expr)
        src_ids = src_ids + [0] * (self.max_input_len - len(src_ids))
        src_ids = src_ids[:self.max_input_len]
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
        
        target_ids = self.tokenizer.encode(target)
        target_ids = [1] + target_ids + [2]  # SOS + target + EOS
        
        dec_token_ids = [1]  # Start with SOS
        steps = []
        failure_onset = None
        
        with torch.no_grad():
            for step in range(self.max_output_len - 1):
                # Pad decoder input
                current_dec = dec_token_ids + [0] * (self.max_output_len - len(dec_token_ids))
                current_dec = current_dec[:self.max_output_len]
                dec_tensor = torch.tensor([current_dec], dtype=torch.long).to(self.device)
                
                # Get prediction
                logits = self.model(src_tensor, dec_tensor)
                nxt_token_id = logits[0, len(dec_token_ids) - 1, :].argmax(dim=-1).item()
                
                # Get expected token
                expected_token_id = target_ids[step + 1] if step + 1 < len(target_ids) else 2
                is_correct = nxt_token_id == expected_token_id
                
                # Track first error
                if not is_correct and failure_onset is None:
                    failure_onset = step
                
                # Decode FIRST, then convert to string EXPLICITLY
                pred_str = str(self._decode_token_safe(nxt_token_id))
                exp_str = str(self._decode_token_safe(expected_token_id))
                
                steps.append({
                    'step': step,
                    'predicted_token': pred_str,  # Store as STRING
                    'expected_token': exp_str,     # Store as STRING
                    'correct': bool(is_correct),
                    'is_first_error': bool(failure_onset == step)
                })
                
                # Stop on EOS or PAD
                if nxt_token_id == 2 or nxt_token_id == 0:
                    break
                
                dec_token_ids.append(nxt_token_id)
        
        return {
            'input': expr,
            'target': target,
            'steps': steps,
            'failure_onset': failure_onset,
        }
    
    def trace_multiple_failures(self, dataset_path: str, num_examples: int = 5) -> List[Dict]:
        # Trace multiple failure examples
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        dataset = data if isinstance(data, list) else data.get('data', [])
        
        failures = []
        count = 0
        
        print(f"\n CP5: Finding failure examples...")
        
        with torch.no_grad():
            for sample in dataset:
                if count >= num_examples:
                    break
                
                expr = sample.get("input", "")
                target = str(sample.get("output", ""))
                
                if not expr:
                    continue
                
                try:
                    pred = self._predict(expr).strip()
                    target_stripped = target.strip()
                    
                    if pred != target_stripped:
                        trace = self.trace_failure(expr, target_stripped)
                        failures.append(trace)
                        count += 1
                        print(f"  Found failure {count}/{num_examples}")
                except Exception as e:
                    pass
        
        return failures

    # Helper methods
    def _predict(self, expr: str) -> str:
        # Predict answer for expression (greedy decoding)
        src_ids = self.tokenizer.encode(expr)
        src_ids = src_ids + [0] * (self.max_input_len - len(src_ids))
        src_ids = src_ids[:self.max_input_len]
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
        
        dec_token_ids = [1]  # Start with SOS
        
        with torch.no_grad():
            for _ in range(self.max_output_len - 1):
                current_dec = dec_token_ids + [0] * (self.max_output_len - len(dec_token_ids))
                current_dec = current_dec[:self.max_output_len]
                dec_tensor = torch.tensor([current_dec], dtype=torch.long).to(self.device)
                
                logits = self.model(src_tensor, dec_tensor)
                nxt_token_id = logits[0, len(dec_token_ids) - 1, :].argmax(dim=-1).item()
                
                if nxt_token_id == 2 or nxt_token_id == 0:  # EOS or PAD
                    break
                
                dec_token_ids.append(nxt_token_id)
        
        return self.tokenizer.decode(dec_token_ids[1:])