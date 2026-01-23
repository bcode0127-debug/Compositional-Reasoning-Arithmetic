import json
import os
from typing import List, Dict, Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader
from data.tokenizer import MathTokenizer, create_tokenizer

class MathDataPipeline:
    """Data pipeline for loading and batching math expression datasets."""
    
    def __init__(self, data_dir: str = "datasets", max_input_len: int = 20, max_output_len: int = 10, batch_size: int = 128):
        """Initialize the data pipeline."""
        self.data_dir = data_dir
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self.tokenizer = create_tokenizer()
    
    def load_data(self, level: str) -> list:
        """Load dataset split and return DataLoader."""
        file_path = os.path.join(self.data_dir, f"level{level}", f"lvl_{level}.json")
        print(f"Loading data from {file_path}...")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data['data'] # Return only the data part
    
    def prepare_sequences(self, raw_data: list) -> tuple:
        """Prepare input and output sequences from raw data."""
        encoder_inputs = []
        decoder_inputs = []
        decoder_targets = []

        for item in raw_data:
            input_expr = item['input']
            output_expr = str(item['output'])

           # Encode input expression (no special tokens)
            enc_input = self.tokenizer.encode(input_expr)
            enc_input = enc_input + [self.tokenizer.pad_idx] * (self.max_input_len - len(enc_input))
            enc_input = enc_input[:self.max_input_len]
            encoder_inputs.append(enc_input)
            
            # Decode input: <SOS> + answer tokens
            dec_input = [self.tokenizer.sos_idx] + self.tokenizer.encode(output_expr)
            dec_input = dec_input + [self.tokenizer.pad_idx] * (self.max_output_len - len(dec_input))
            dec_input = dec_input[:self.max_output_len]
            decoder_inputs.append(dec_input)
            
            # Decode target: answer tokens + <EOS>
            dec_target = self.tokenizer.encode(output_expr) + [self.tokenizer.eos_idx]
            dec_target = dec_target + [self.tokenizer.pad_idx] * (self.max_output_len - len(dec_target))
            dec_target = dec_target[:self.max_output_len]
            decoder_targets.append(dec_target)

        return (
            torch.LongTensor(encoder_inputs),
            torch.LongTensor(decoder_inputs),
            torch.LongTensor(decoder_targets)
        )

    def get_dataloader(self, level: int, shuffle: bool = True) -> DataLoader:
        """Get DataLoader for specified level."""
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

    def get_all_dataloaders(self) -> Dict[str, DataLoader]:
        """Get DataLoaders for all levels."""

        return {
            "level1": self.get_dataloader(level=1, shuffle=True),
            "level2": self.get_dataloader(level=2, shuffle=False),
            "level3": self.get_dataloader(level=3, shuffle=False)
        }
    def get_train_val_dataloaders(self, level: int, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders for a specific level with train/val split."""
        print(f"Preparing Level {level} Data (Train/Val Split)")
        print(f"{'='*60}")
    
        raw_data = self.load_data(str(level))
        print(f"✓ Loaded {len(raw_data)} samples")
    
        # Split into train/val
        split_idx = int(len(raw_data) * train_split)
        train_data = raw_data[:split_idx]
        val_data = raw_data[split_idx:]
    
        # Prepare sequences
        enc_inputs_train, dec_inputs_train, dec_targets_train = self.prepare_sequences(train_data)
        enc_inputs_val, dec_inputs_val, dec_targets_val = self.prepare_sequences(val_data)
    
        # Create dataloaders
        train_dataset = TensorDataset(enc_inputs_train, dec_inputs_train, dec_targets_train)
        val_dataset = TensorDataset(enc_inputs_val, dec_inputs_val, dec_targets_val)
    
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
    
        print(f"✓ Train samples: {len(train_data)}, batches: {len(train_loader)}")
        print(f"✓ Val samples: {len(val_data)}, batches: {len(val_loader)}\n")
    
        return train_loader, val_loader

def get_dataloaders(batch_size: int = 128) -> Dict[str, DataLoader]:
    """Utility function to get dataloaders for all levels."""
    pipeline = MathDataPipeline(batch_size=batch_size)
    return pipeline.get_all_dataloaders()