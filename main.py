import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
from data.generate import generate_lvl1, generate_lvl2, generate_lvl3, save_json
from data.tokenizer import create_tokenizer
from data.dataloader import MathDataPipeline
from models.lstm import create_lstm_model
from utils.trainer import train_model


def test_tokenizer():
    """Function to test the MathTokenizer"""
    print("="*60)
    print("TESTING MATH TOKENIZER")
    print("="*60)

    tokenizer = create_tokenizer()
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {tokenizer.vocab}")

    test_cases = [
        "12 + 34 - 5",
        "(3.5 * 2) / 7",
        "100 / (25 - 5) + 3.14"
    ]
    print("\n" + "ENCODING/DECODING TESTS")
    print("="*60)

    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        match = "✓" if text == decoded else "✗"
        print(f"Original: {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Match: {match}")
        print("-"*40)
    
    batch_texts = ["5 + 3", "10 - 2", "7 * 4"]
    batch_tensor = tokenizer.encode_batch(batch_texts, max_length=20, add_sos=True, add_eos=True)
    
    print(f"\nInput texts: {batch_texts}")
    print(f"Batch tensor shape: {batch_tensor.shape}")
    decoded_batch = tokenizer.decode_batch(batch_tensor)
    print(f"Decoded batch: {decoded_batch}")
    
    all_match = all(orig == dec for orig, dec in zip(batch_texts, decoded_batch))
    print(f"All match: {'✓' if all_match else '✗'}")
    
    print("\n" + "="*60)
    print("✅ TOKENIZER TESTS COMPLETE!")
    print("="*60)


def test_dataloader():
    """Test the data pipeline and DataLoaders"""
    print("\n" + "="*60)
    print("TESTING DATA PIPELINE")
    print("="*60 + "\n")
    
    pipeline = MathDataPipeline(batch_size=128)
    dataloaders = pipeline.get_all_dataloaders()
    
    for level, dataloader in dataloaders.items():
        print(f"\n{level.upper()}:")
        print(f"  Total batches: {len(dataloader)}")
        
        # Get first batch
        batch = next(iter(dataloader))
        enc_input, dec_input, dec_target = batch
        
        print(f"  Batch shapes:")
        print(f"    Encoder input: {enc_input.shape}")
        print(f"    Decoder input: {dec_input.shape}")
        print(f"    Decoder target: {dec_target.shape}")
    
    print("\n" + "="*60)
    print("✅ DATA PIPELINE TESTS COMPLETE!")
    print("="*60)


def test_lstm():
    """Test LSTM model architecture"""
    print("\n" + "="*60)
    print("TESTING LSTM MODEL ARCHITECTURE")
    print("="*60 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Create model
    model = create_lstm_model(embedding_dim=128, hidden_size=256, vocab_size=21)
    model = model.to(device)
    print(f"✅ Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Test forward pass with random data
    batch_size, src_len, tgt_len = 32, 20, 10
    enc_input = torch.randint(0, 21, (batch_size, src_len)).to(device)
    dec_input = torch.randint(0, 21, (batch_size, tgt_len)).to(device)
    
    output = model(enc_input, dec_input)
    print(f"Forward pass successful!")
    print(f"  Input shape: {enc_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected shape: [batch={batch_size}, seq_len={tgt_len}, vocab_size=21]")
    
    assert output.shape == (batch_size, tgt_len, 21), "Output shape mismatch!"
    print(f"\n✅ Shape assertion passed!")
    print("="*60)


def train_lstm_on_level(level: str, num_epochs: int = 20, batch_size: int = 32, learning_rate: float = 0.001):
    """Train LSTM model on a specific difficulty level"""
    print(f"\n{'='*60}")
    print(f"TRAINING LSTM ON {level.upper()}")
    print(f"{'='*60}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data with train/val split
    pipeline = MathDataPipeline(batch_size=batch_size)
    level_num = int(level[-1])  # Extract number from "level1" -> 1
    train_loader, val_loader = pipeline.get_train_val_dataloaders(level_num, train_split=0.8)
    
    # Create model
    model = create_lstm_model(embedding_dim=128, hidden_size=256, vocab_size=21)
    
    # Setup checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "results", "lstm", level)
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=save_path,
        pad_idx=0,
        early_stopping_patience=5
    )
    
    # Save training history
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete for {level}!")
    print(f"   Best model saved to: {save_path}")
    print(f"   History saved to: {history_path}")
    
    return history


def main():
    """Main function to generate math reasoning datasets, test pipeline, and train LSTM"""
    # Test tokenizer first
    test_tokenizer()
    print("\n")
    
    print("🚀 Generating datasets for math reasoning research...\n")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "datasets")
    os.makedirs(os.path.join(output_dir, "level1"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "level2"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "level3"), exist_ok=True)
    
    # Level 1
    lvl1_path = os.path.join(output_dir, "level1", "lvl_1.json")
    if not os.path.exists(lvl1_path):
        print("🔄 Generating Level 1 (LSTM-friendly: +, -, *)...")
        lvl1_data = generate_lvl1(num_samples=5000, max_num=15, seed=42)
        save_json(
            {"metadata": {"level": "lvl1", "total_samples": len(lvl1_data)}, "data": lvl1_data},
            lvl1_path
        )
        print(f"✅ Level 1: {len(lvl1_data)} samples saved\n")
    else:
        print("✅ Level 1 already exists, skipping\n")
    
    # Level 2
    lvl2_path = os.path.join(output_dir, "level2", "lvl_2.json")
    if not os.path.exists(lvl2_path):
        print("🔄 Generating Level 2 (Medium: +, -, *, /)...")
        lvl2_data = generate_lvl2(num_samples=5000, max_num=15, seed=42)
        save_json(
            {"metadata": {"level": "lvl2", "total_samples": len(lvl2_data)}, "data": lvl2_data},
            lvl2_path
        )
        print(f"✅ Level 2: {len(lvl2_data)} samples saved\n")
    else:
        print("✅ Level 2 already exists, skipping\n")
    
    # Level 3
    lvl3_path = os.path.join(output_dir, "level3", "lvl_3.json")
    if not os.path.exists(lvl3_path):
        print("🔄 Generating Level 3 (Hard: +, -, *, /, parentheses)...")
        lvl3_data = generate_lvl3(num_samples=5000, max_num=15, seed=42, parentheses_prob=0.7)
        save_json(
            {"metadata": {"level": "lvl3", "total_samples": len(lvl3_data)}, "data": lvl3_data},
            lvl3_path
        )
        print(f"✅ Level 3: {len(lvl3_data)} samples saved\n")
    else:
        print("✅ Level 3 already exists, skipping\n")
    
    print("="*60)
    print("✅ Dataset generation complete!")
    print("="*60)
    
    # Test dataloader pipeline
    test_dataloader()
    
    # Test LSTM model
    test_lstm()
    
    # Train LSTM on all levels
    print("\n" + "🚀 STARTING LSTM TRAINING ON ALL LEVELS\n")
    
    results = {}
    for level in ["level1", "level2", "level3"]:
        history = train_lstm_on_level(level, num_epochs=20, batch_size=32, learning_rate=0.001)
        results[level] = history
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    for level, history in results.items():
        print(f"\n{level.upper()}:")
        print(f"  Best validation loss: {history['best_val_loss']:.4f}")
        print(f"  Final validation accuracy: {history['val_accuracies'][-1]:.2f}%")
        print(f"  Training epochs: {len(history['train_losses'])}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()