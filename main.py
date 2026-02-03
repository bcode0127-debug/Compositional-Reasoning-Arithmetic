import os
import json
import argparse
from pathlib import Path
import torch
from data.generate_controlled import (
    save_dataset,
    generate_controlled_dataset,
    generate_verification_samples,
    print_verification_samples,
    save_verification_samples,
)
from data.tokenizer import create_tokenizer
from data.dataloader import MathDataPipeline
from models.lstm import create_lstm_model
from utils.trainer import train_model
from torch.utils.data import random_split, DataLoader

SEP = "=" * 60

# Tokenizer tests
def test_tokenizer():
    # Function to test the MathTokenizer
    print(SEP)
    print("TESTING MATH TOKENIZER")
    print(SEP)

    tokenizer = create_tokenizer()
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Vocabulary: {tokenizer.vocab}")

    # Integer-only encoding/decoding tests
    test_cases = [
        "12 + 34 - 5",
        "(3 * 2) / 7",
        "100 / (25 - 5) + 3"
    ]
    print("\nENCODING/DECODING TESTS")
    print(SEP)

    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        match = "✓" if text == decoded else "✗"
        print(f"Original: {text}")
        print(f"Encoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Match: {match}")
        print(SEP)
    
    batch_texts = ["5 + 3", "10 - 2", "7 * 4"]
    batch_tensor = tokenizer.encode_batch(batch_texts, max_length=20, add_sos=True, add_eos=True)
    
    print(f"\nInput texts: {batch_texts}")
    print(f"Batch tensor shape: {batch_tensor.shape}")
    decoded_batch = tokenizer.decode_batch(batch_tensor)
    print(f"Decoded batch: {decoded_batch}")
    
    all_match = all(orig == dec for orig, dec in zip(batch_texts, decoded_batch))
    print(f"All match: {'✓' if all_match else '✗'}")
    
    print("\n" + SEP)
    print("TOKENIZER TESTS COMPLETE!")
    print(SEP)

def generate_verification(num_samples: int = 40, seed: int = 42):
    # Generate 40 verification samples for professor review.
    print("\n" + SEP)
    print("GENERATING VERIFICATION SAMPLES")
    print(SEP)
    
    # Generate samples
    samples = generate_verification_samples(num_samples=40, seed=42)
    
    # Print to console
    print_verification_samples(samples)
    
    # Save to JSON
    verification_dir = Path(__file__).parent / "datasets" / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    save_verification_samples(samples, str(verification_dir / "samples_40.json"))   
    
    print("\n" + SEP)
    print("✅ VERIFICATION COMPLETE!")
    print(SEP)
    
   

# Data Pipeline Testing
def test_dataloader(data_dir: str = "datasets"):
    # Test the data pipeline and DataLoaders
    print("\n" + SEP)
    print("TESTING DATA PIPELINE")
    print(SEP + "\n")
    
    pipeline = MathDataPipeline(data_dir=data_dir, batch_size=128)
    dataloaders = pipeline.get_all_dataloaders()
    
    for level, dataloader in dataloaders.items():
        print(f"\n{level.upper()}:")
        print(f"  Total batches: {len(dataloader)}")
        
        # Get first batch
        batch = next(iter(dataloader))
        enc_input, dec_input, dec_target = batch
        
        print(f"Batch shapes:")
        print(f"Encoder input: {enc_input.shape}")
        print(f"Decoder input: {dec_input.shape}")
        print(f"Decoder target: {dec_target.shape}")
    
    print("\n" + SEP)
    print("DATA PIPELINE TESTS COMPLETE!")
    print(SEP)

# LSTM model testing
def test_lstm():
    # Test LSTM model architecture
    print("\n" + SEP)
    print("TESTING LSTM MODEL ARCHITECTURE")
    print(SEP + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Create model
    model = create_lstm_model(embedding_dim=128, hidden_size=256, vocab_size=21)
    model = model.to(device)
    print(f"Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Test forward pass with random data
    batch_size, src_len, tgt_len = 32, 20, 10
    enc_input = torch.randint(0, 21, (batch_size, src_len)).to(device)
    dec_input = torch.randint(0, 21, (batch_size, tgt_len)).to(device)
    
    output = model(enc_input, dec_input)
    print(f"Forward pass successful!")
    print(f"Input shape: {enc_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [batch={batch_size}, seq_len={tgt_len}, vocab_size=21]")
    
    assert output.shape == (batch_size, tgt_len, 21), "Output shape mismatch!"
    print(f"\nShape assertion passed!")
    print(SEP)



# Sanity Check: Overfit on tiny dataset
def overfit_sanity_check(num_samples: int = 30, num_epochs: int = 50):
    """Verify the model can overfit on a tiny dataset"""
    print("\n" + "="*60)
    print("SANITY CHECK: Overfit Test")
    print(f"Testing if model can memorize {num_samples} samples")
    print("="*60 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Generate tiny dataset
    print("Generating tiny controlled dataset...")
    tiny_data = generate_controlled_dataset(num_samples=num_samples, seed=999) 
    
    # Save temporarily
    temp_dir = Path(__file__).parent / "datasets" / "sanity_temp"  
    temp_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(tiny_data, str(temp_dir / "data.json"))  
    
    # Create dataloaders
    pipeline = MathDataPipeline(data_dir=str(temp_dir), batch_size=8)
    
    # Load data
    with open(temp_dir / "data.json", 'r') as f:
        data = json.load(f)
    
    # Create dataset
    dataset = pipeline._create_dataset(data)  
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Create model
    model = create_lstm_model(
        embedding_dim=128, 
        hidden_size=256, 
        vocab_size=21
    )
    
    # Train
    checkpoint_dir = Path(__file__).parent / "results" / "lstm_baseline" / "sanity_check"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("Training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=0.001,
        device=device,
        save_path=str(checkpoint_dir / "sanity_model.pt"), 
        pad_idx=0,
        early_stopping_patience=50
    )
    
    # Evaluate
    final_train_acc = history['train_accuracies'][-1]
    final_val_acc = history['val_accuracies'][-1]
    
    print("\n" + "="*60)
    print("SANITY CHECK RESULTS")
    print("="*60)
    print(f"Final train accuracy: {final_train_acc:.2f}%")
    print(f"Final val accuracy: {final_val_acc:.2f}%")
    
    # Determine pass/fail
    threshold = 95.0
    passed = final_val_acc >= threshold
    
    if passed:
        print(f"\n✅ PASS: Model achieved {final_val_acc:.2f}% (>= {threshold}%)")
        print("Architecture is working correctly. Model can memorize.")
    else:
        print(f"\n❌ FAIL: Model only achieved {final_val_acc:.2f}% (< {threshold}%)")
        print("Possible issues:")
        print("  - Bug in model architecture")
        print("  - Loss function not working")
        print("  - Padding mask incorrect")
        print("  - Learning rate too low/high")
    
    print("="*60 + "\n")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    return passed
    
# Training
def train_lstm_on_level(level: str, num_epochs: int = 20, batch_size: int = 32, learning_rate: float = 0.001, data_dir: str = "datasets") -> dict:
    # Train LSTM model on a specific difficulty level
    print("\n" + SEP)
    print(f"TRAINING LSTM ON {level.upper()}")
    print(SEP + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data with train/val split
    pipeline = MathDataPipeline(data_dir=data_dir, batch_size=batch_size)
    level_num = int(level[-1])  # Extract number from "level1" -> 1
    train_loader, val_loader = pipeline.get_train_val_dataloaders(level_num, train_split=0.8)
    
    # Create model
    model = create_lstm_model(embedding_dim=128, hidden_size=256, vocab_size=21)
    
    # Setup checkpoint directory
    checkpoint_dir = Path(__file__).parent / "results" / "lstm_baseline" 
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "best_model.pt"
    
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
    
    print(f"\nTraining complete for {level}!")
    print(f"Best model saved to: {save_path}")
    print(f"History saved to: {history_path}")
    
    return history

# Evaluation
def evaluate_model(level: str, dataset: list, checkpoint_path: str, device: str = "cpu") -> dict:
    # Evaluation of a trained LSTM
    print("\n" + SEP)
    print(f"EVALUATING {level.upper()}")
    print(SEP + "\n")

    # Load model
    vocab_size = 21
    max_input_len = 20
    max_output_len = 10

    model = create_lstm_model(embedding_dim=128, hidden_size=256, vocab_size=vocab_size)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both checkpoint types
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()

    tokenizer = create_tokenizer()

    correct, total = 0, len(dataset)
    errors = []

    with torch.no_grad():
        for sample in dataset:
            expr = sample["input"]
            target = sample["output"]

            try:
                # Encode source expression
                src_ids = tokenizer.encode(expr)
                src_ids = src_ids + [tokenizer.pad_idx] * (max_input_len - len(src_ids))
                src_ids = src_ids[:max_input_len]
                src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

                # decoding: start with <SOS> token, generate one 
                dec_token_ids = [tokenizer.sos_idx]
                pred_ids = []

                for step in range(max_output_len):
                    # Pad dcurrent sequence
                    current_dec = dec_token_ids + [tokenizer.pad_idx] * (max_output_len - len(dec_token_ids))
                    current_dec = current_dec[:max_output_len]
                    dec_tensor = torch.tensor([current_dec], dtype=torch.long).to(device)

                    logits = model(src_tensor, dec_tensor)  # [1, seq_len, vocab_size]

                    # Get next token 
                    nxt_tooken_id = logits[0, step, :].argmax(dim=-1).item()
                    pred_ids.append(nxt_tooken_id)

                    # stop at <EOS> or <PAD>
                    if nxt_tooken_id == tokenizer.eos_idx or nxt_tooken_id == tokenizer.pad_idx:
                        break

                    # add sequence for next step
                    dec_token_ids.append(nxt_tooken_id)

                # Decode predicted 
                pred_str = tokenizer.decode(pred_ids)

                # Compare
                if pred_str == target:
                    correct += 1
                elif len(errors) < 10:
                    errors.append({
                        "input": expr,
                        "expected": target,
                        "predicted": pred_str
                    })
            except Exception as e:
                if len(errors) < 10:
                    errors.append({
                        "input": expr,
                        "error": str(e)
                    })
                continue

    acc = 100.0 * correct / total if total else 0.0
    print(f"Accuracy: {acc:.2f}% ({correct}/{total})")
    
    if errors:
        print("\nSample errors:")
        for e in errors:
            if "error" in e:
                print(f"  Input: {e['input']}")
                print(f"  Error: {e['error']}\n")
            else:
                print(f"  Input    : {e['input']}")
                print(f"  Expected : {e['expected']}")
                print(f"  Got      : {e['predicted']}\n")
    
    return {"accuracy": acc, "correct": correct, "total": total}

# Main Entry point
def main() -> None:
    # Main entry point for the pipeline.
    parser = argparse.ArgumentParser(
        description="Math reasoning pipeline – dataset generation, training, evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "all", "train", "eval"],
        default="verify",
        help="Which stages to run"
    )
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples for controlled datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # Always run tokenizer test
    test_tokenizer()

    # Mode: verify (generate verification samples only)
    if args.mode == "verify":
        generate_verification(num_samples=args.num_samples, seed=args.seed)
        return

    # Mode: all, train, eval (legacy modes - keep for now but warn)
    print("Use --mode verify to generate verification samples first.\n")

    # Always run tokenizer test for other modes
    test_tokenizer()

    # Training
    if args.mode in ("all", "train"):
        print("\n❌ ERROR: Cannot train without full datasets.")
        print("   Generate verification samples first with --mode verify")
        return
    
    # Evaluation (only after training is complete)
    if args.mode in ("all", "eval"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("\n🚀 EVALUATING MODELS ON CONTROLLED DATASETS\n")

        #Load data from disk to match training/eval scenario
        pipeline = MathDataPipeline(data_dir="datasets", batch_size=32) 
        
        for lvl in ("level1", "level2", "level3"):
            ckpt_path = (
                Path(__file__).parent
                / "results"
                / "lstm_baseline"
                / lvl
                / "best_model.pt"
            )
            if ckpt_path.is_file():
                # Load controlled dataset from disk
                level_num = int(lvl[-1])
                eval_data = pipeline.load_data(str(level_num))

                evaluate_model(
                    level=lvl,
                    dataset=eval_data,
                    checkpoint_path=str(ckpt_path),
                    device=device,
                )
            else:
                print(f"No checkpoint found for {lvl} – skipping evaluation")

    print("\n" + SEP)
    print("ALL DONE!")
    print(SEP)


if __name__ == "__main__":
    main()