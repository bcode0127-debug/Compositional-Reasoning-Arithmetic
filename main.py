import os
import json
import argparse
from pathlib import Path
import torch
import data
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

def generate_study_datasets():
    # Generate controlled datasets for studies
    print("\n" + SEP)
    print("GENERATING CONTROLLED DATASETS")
    print(SEP)

    print("Generating Study 1 (Length Generalization)...")
    study1_train = generate_controlled_dataset(
        num_samples=8000,
        num_ops_range=(2, 3),
        depth_limit=3,
        seed=42
    )
    study1_val = generate_controlled_dataset(
        num_samples=1000,
        num_ops_range=(2, 3),
        depth_limit=3,
        seed=4242
    )
    study1_ood = generate_controlled_dataset(
        num_samples=1000,
        num_ops_range=(4, 7),
        depth_limit=3,
        seed=424242
    )   

    study1_dir = Path(__file__).parent / "datasets" / "study1"
    study1_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(study1_train, str(study1_dir / "train.json"))
    save_dataset(study1_val, str(study1_dir / "val.json"))
    save_dataset(study1_ood, str(study1_dir / "ood.json"))
    print(f"✅ Study 1 saved to {study1_dir}\n")

    print("Generating Study 2 (Depth Generalization)...")
    study2_train = generate_controlled_dataset(
        num_samples=8000,
        num_ops_range=(3, 3),
        depth_limit=2,
        seed=43
    )
    study2_val = generate_controlled_dataset(
        num_samples=1000,
        num_ops_range=(3, 3),
        depth_limit=2,
        seed=4343
    )
    study2_ood = generate_controlled_dataset(
        num_samples=1000,
        num_ops_range=(3, 3),
        depth_limit=3,
        seed=434343
    )

    study2_dir = Path(__file__).parent / "datasets" / "study2"
    study2_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(study2_train, str(study2_dir / "train.json"))
    save_dataset(study2_val, str(study2_dir / "val.json"))
    save_dataset(study2_ood, str(study2_dir / "ood.json"))
    print(f"✅ Study 2 saved to {study2_dir}\n")

    print("\n" + SEP)
    print("✅ DATASET GENERATION COMPLETE!")
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
    # Verify the model can overfit on a tiny dataset
    print("\n" + SEP)
    print("SANITY CHECK: Overfit Test")
    print(f"Testing if model can memorize {num_samples} samples")
    print(SEP + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Generate tiny dataset
    print("Generating tiny controlled dataset...")
    tiny_data = generate_controlled_dataset(num_samples=num_samples,
                                            num_ops_range=(2, 3),
                                            depth_limit=3,
                                            seed=999) 
    
    # Save temporarily
    temp_dir = Path(__file__).parent / "datasets" / "sanity_temp"  
    temp_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(tiny_data, str(temp_dir / "sanity.json"))  
    
    # Create dataloaders
    pipeline = MathDataPipeline(data_dir=str(temp_dir), batch_size=8)
    train_loader = pipeline.get_dataloaders_file("sanity.json", shuffle=True)
    val_loader = pipeline.get_dataloaders_file("sanity.json", shuffle=False)

    print(f"Dataset size: {len(tiny_data)} samples")

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
    
    print("\n" + SEP)
    print("SANITY CHECK RESULTS")
    print(SEP)
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
        print("- Model architecture may have bugs.")
    
    print(SEP + "\n")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    return passed
    
# Training
def train_lstm_model(study: str, dataset_split: str = "train", num_epochs: int = 20, batch_size: int = 32, learning_rate: float = 0.001, data_dir: str = "datasets") -> dict:
    # Train LSTM model on a specific study and dataset split
    print("\n" + SEP)
    print(f"TRAINING LSTM ON {study.upper()} - {dataset_split.upper()}")
    print(SEP + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data from study JSON file
    data_path = Path(data_dir) / study / f"{dataset_split}.json"
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    # Create dataloaders
    pipeline = MathDataPipeline(data_dir=data_dir, batch_size=batch_size)
    data_file = f"{study}/{dataset_split}.json"
    train_loader = pipeline.get_dataloaders_file(data_file, shuffle=True)
    val_loader = pipeline.get_dataloaders_file(data_file, shuffle=False)
    
    # Create model
    model = create_lstm_model(embedding_dim=128, hidden_size=256, vocab_size=21)
    
    # Setup checkpoint directory per study
    checkpoint_dir = Path(__file__).parent / "results" / "lstm_baseline" / study
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / f"{dataset_split}_best_model.pt"
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=str(save_path),
        pad_idx=0,
        early_stopping_patience=5
    )
    
    # Save training history
    history_path = checkpoint_dir / f"{dataset_split}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete for {study} ({dataset_split})!")
    print(f"Best model saved to: {save_path}")
    print(f"History saved to: {history_path}")
    
    return history

# Evaluation
def evaluate_model(study: str, dataset_split: str, checkpoint_path: str, device: str = "cpu") -> dict:
    # Evaluation of a trained LSTM
    print("\n" + SEP)
    print(f"EVALUATING {study.upper()} - {dataset_split.upper()}")
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

    # Load dataset
    data_path = Path("datasets") / study / f"{dataset_split}.json"
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Handle wrapped format
    if isinstance(data, dict) and 'data' in data:
        dataset = data['data']
    else:
        dataset = data
    
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

                # decoding: start with <SOS> token, generate tokens
                dec_token_ids = [tokenizer.sos_idx]
                pred_ids = []

                for step in range(max_output_len):
                    # Pad current sequence
                    current_dec = dec_token_ids + [tokenizer.pad_idx] * (max_output_len - len(dec_token_ids))
                    current_dec = current_dec[:max_output_len]
                    dec_tensor = torch.tensor([current_dec], dtype=torch.long).to(device)

                    logits = model(src_tensor, dec_tensor)  # [1, seq_len, vocab_size]

                    # Get next token 
                    nxt_token_id = logits[0, step, :].argmax(dim=-1).item()
                    pred_ids.append(nxt_token_id)

                    # stop at <EOS> or <PAD>
                    if nxt_token_id == tokenizer.eos_idx or nxt_token_id == tokenizer.pad_idx:
                        break

                    # add to sequence for next step
                    dec_token_ids.append(nxt_token_id)

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
        description="Math Reasoning LSTM Baseeline"
    )
    parser.add_argument(
        "--mode",
        choices=["verify", "generate", "train", "eval", "test", "sanity"],
        default="generate",
        help="Which stages to run"
    )
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    # Mode routing
    if args.mode == "verify":
        generate_verification()

    elif args.mode == "generate":
        generate_study_datasets()

    elif args.mode == "sanity":
        overfit_sanity_check()

    elif args.mode == "train":
        print("\n" + SEP)
        print("TRAINING MODELS")
        print(SEP + "\n")
        train_lstm_model("study1", "train", num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.lr)
        train_lstm_model("study2", "train", num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.lr)

    elif args.mode == "eval":
        print("\n" + SEP)
        print("EVALUATING MODELS")
        print(SEP + "\n")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Evaluate Study 1
        checkpoint_s1 = Path(__file__).parent / "results" / "lstm_baseline" / "study1" / "train_best_model.pt"
        if checkpoint_s1.exists():
            evaluate_model("study1", "val", str(checkpoint_s1), device=device)
            evaluate_model("study1", "ood", str(checkpoint_s1), device=device)
        else:
            print(f"⚠️ Study 1 checkpoint not found: {checkpoint_s1}")
        
        # Evaluate Study 2
        checkpoint_s2 = Path(__file__).parent / "results" / "lstm_baseline" / "study2" / "train_best_model.pt"
        if checkpoint_s2.exists():
            evaluate_model("study2", "val", str(checkpoint_s2), device=device)
            evaluate_model("study2", "ood", str(checkpoint_s2), device=device)
        else:
            print(f"⚠️ Study 2 checkpoint not found: {checkpoint_s2}")

    elif args.mode == "test":
        test_tokenizer()
        test_lstm()

    print("\n" + SEP)
    print("ALL DONE!")
    print(SEP)

if __name__ == "__main__":
    main()