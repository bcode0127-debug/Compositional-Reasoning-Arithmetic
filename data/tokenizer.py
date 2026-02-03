"""
Character-level tokenizer for mathematical expressions.
Handles encoding/decoding and padding.
"""

from typing import List
import torch


class MathTokenizer:
    """Character-level tokenizer for math expressions."""
    
    def __init__(self):
        """Initialize tokenizer with vocabulary."""
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        
        chars = list('0123456789+-*/() ')
        self.vocab = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN] + chars
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.char_to_idx[self.PAD_TOKEN]
        self.sos_idx = self.char_to_idx[self.SOS_TOKEN]
        self.eos_idx = self.char_to_idx[self.EOS_TOKEN]
    
    def encode(self, text: str) -> List[int]:
        """Convert text to list of token indices."""
        return [self.char_to_idx.get(char, self.char_to_idx[self.PAD_TOKEN]) for char in text]
    
    def decode(self, indices: List[int]) -> str:
        """Convert list of indices back to text."""
        chars = [self.idx_to_char.get(idx, '') for idx in indices 
                 if idx not in [self.pad_idx, self.sos_idx, self.eos_idx]]
        return ''.join(chars).strip()
    
    def encode_batch(self, texts: List[str], max_length: int, add_sos: bool = False, add_eos: bool = False) -> torch.Tensor:
        """Encode batch of texts with padding."""
        batch = []
        for text in texts:
            indices = self.encode(text)
            if add_sos:
                indices = [self.sos_idx] + indices
            if add_eos:
                indices = indices + [self.eos_idx]
            if len(indices) < max_length:
                indices = indices + [self.pad_idx] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
            batch.append(indices)
        return torch.LongTensor(batch)
    
    def decode_batch(self, tensor: torch.Tensor) -> List[str]:
        """Decode batch of tensors to strings."""
        return [self.decode(indices.tolist()) for indices in tensor]


def create_tokenizer() -> MathTokenizer:
    """Create and return a tokenizer instance."""
    return MathTokenizer()