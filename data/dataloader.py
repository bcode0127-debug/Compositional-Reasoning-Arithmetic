import json
import os
from typing import Dict
import torch
from torch.utils.data import TensorDataset, DataLoader
from data.tokenizer import MathTokenizer, create_tokenizer

class MathDataPipeline:
    """
        Data pipeline for loading and batching math expression datasets.

        Args:
        data_dir (str): Directory where dataset files are located.
        max_input_len (int): Maximum length of input token sequences.
        max_output_len (int): Maximum length of output token sequences.
        batch_size (int): Batch size for DataLoader.
        tokenizer (MathTokenizer): Tokenizer instance for encoding/decoding expressions.
        """
    def __init__(self, data_dir: str = "datasets", max_input_len: int = 20, max_output_len: int = 10, batch_size: int = 128):
        # Data pipeline for loading and batching math expression datasets.

        self.data_dir = data_dir
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self.tokenizer = create_tokenizer()
    
    def load_data(self, level: str) -> list:
        # Load dataset for a specific level.

        file_path = os.path.join(self.data_dir, f"level{level}")

        candidates = [
            os.path.join(file_path, f"lvl_{level}_controlled.json"),
            os.path.join(file_path, f"lvl_{level}.json")
        ]

        file_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                file_path = candidate
                break
        
        if file_path is None:
            raise FileNotFoundError(f"No dataset file found for level {level} in {self.data_dir}")
        
        print(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        return data_json['data']
    
    def prepare_sequences(self, raw_data: list) -> tuple:
        """
        Prepare encoder and decoder sequences from raw data.

        For each sample:
         - Encoder input: tokenized input expression (padded)
         - Decoder input: <SOS> + tokenized answer (padded)
         - Decoder target: tokenized answer + <EOS> (padded)
         """
        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []

        for item in raw_data:
            input_expr = item['input']
            output_expr = str(item['output'])

            # Encoder input expression (no special tokens)
            enc_input = self.tokenizer.encode(input_expr)
            enc_input = enc_input + [self.tokenizer.pad_idx] * (self.max_input_len - len(enc_input))
            enc_input = enc_input[:self.max_input_len]
            encoder_inputs.append(enc_input)
        
            # Encoder output answer tokens
            answer_tokens = self.tokenizer.encode(output_expr)
        
            # Decoder input: <SOS> + answer tokens (teacher forcing input)
            dec_input = [self.tokenizer.sos_idx] + answer_tokens
            dec_input = dec_input + [self.tokenizer.pad_idx] * (self.max_output_len - len(dec_input))
            dec_input = dec_input[:self.max_output_len]
            decoder_inputs.append(dec_input)
        
            # Decoder target: answer tokens + <EOS> (what model should predict)
            answer_with_eos = answer_tokens + [self.tokenizer.eos_idx]
            dec_target = answer_with_eos[:self.max_output_len]
            if len(dec_target) < self.max_output_len:
                dec_target = dec_target + [self.tokenizer.pad_idx] * (self.max_output_len - len(dec_target))
            decoder_targets.append(dec_target)

        return (
            torch.LongTensor(encoder_inputs),
            torch.LongTensor(decoder_inputs),
            torch.LongTensor(decoder_targets)
        )
    
    def get_dataloader(self, level: int, shuffle: bool = True) -> DataLoader:
        # Get DataLoader for specified level.

        print(f"Preparing Level {level} Data")
        print(f"{'='*60}")

        raw_data = self.load_data(str(level))
        print(f"✓ Loaded {len(raw_data)} samples")

        enc_inputs, dec_inputs, dec_targets = self.prepare_sequences(raw_data)
        print(f"✓ Tokenized to shapes: {enc_inputs.shape}, {dec_inputs.shape}, {dec_targets.shape}")
        
        dataset = TensorDataset(enc_inputs, dec_inputs, dec_targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        print(f"✓ DataLoader created: {len(dataloader)} batches\n")
        
        return dataloader

    def get_dataloaders_file(self, filename: str, shuffle: bool = True) -> DataLoader:
        # Get DataLoaders for a specific file.

        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No dataset file found: {file_path}")
        
        print(f"Loading data from {filename}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data_json = json.load(f)
        raw_data = data_json['data']
        print(f"✓ Loaded {len(raw_data)} samples")

        enc_inputs, dec_inputs, dec_targets = self.prepare_sequences(raw_data)
        print(f"✓ Tokenized to shapes: {enc_inputs.shape}, {dec_inputs.shape}, {dec_targets.shape}")

        dataset = TensorDataset(enc_inputs, dec_inputs, dec_targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        print(f"✓ DataLoader created: {len(dataloader)} batches\n")

        return dataloader
    