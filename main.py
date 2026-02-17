import json
import argparse
from pathlib import Path
import torch
from data.generate_controlled import (
    save_dataset,
    generate_controlled_dataset,
)
from data.tokenizer import create_tokenizer
from data.dataloader import MathDataPipeline
from models.lstm import create_lstm_model
from utils.trainer import train_model
from models.transformer import create_transformer_model 

SEP = "=" * 60
# Configuration dictionaries for adjustments
MODEL_CONFIG = {
    'lstm': {
        'embedding_dim': 128,
        'hidden_size': 256,
    },
    'transformer': {
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
    }
}

TRAINING_CONFIG = {
    'max_input_len': 20,
    'max_output_len': 10,
    'learning_rate': {
        'lstm': 0.001,
        'transformer': 0.0001,
    }
}

# Helper functions
def create_model(model_type: str, vocab_size: int) -> torch.nn.Module:
    """
    Create a model based on type and config.
    Single source of truth for model instantiation.
    """
    if model_type == 'lstm':
        return create_lstm_model(
            embedding_dim=MODEL_CONFIG['lstm']['embedding_dim'],
            hidden_size=MODEL_CONFIG['lstm']['hidden_size'],
            vocab_size=vocab_size
        )
    elif model_type == 'transformer':
        return create_transformer_model(
            vocab_size=vocab_size,
            d_model=MODEL_CONFIG['transformer']['d_model'],
            nhead=MODEL_CONFIG['transformer']['nhead'],
            num_encoder_layers=MODEL_CONFIG['transformer']['num_encoder_layers'],
            num_decoder_layers=MODEL_CONFIG['transformer']['num_decoder_layers']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
# Tokenizer tests
def test_tokenizer():
    """
    Test the MathTokenizer with encoding/decoding operations.

    validates:
    - Single expression encoding and decoding accuracy
    - Batch encoding with special tokens (SOS, EOS)
    - Round-trip consistency (encode - decode matches original)
    """
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

def generate_study_datasets():
    """
    Generate controlled datasets for generalization studies.

    Study 1: Length Generalization
    - Train/Val: 2-3 operations, depth ≤ 3
    - OOD: 4-7 operations, depth ≤ 3 

    Study 2: Depth Generalization
    - Train/Val: 3 operations, depth ≤ 2
    - OOD: 3 operations, depth ≤ 3 
    """
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
    print(f"Study 1 saved to {study1_dir}\n")

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
    print(f"Study 2 saved to {study2_dir}\n")

    print("\n" + SEP)
    print("DATASET GENERATION COMPLETE!")
    print(SEP)
   

# LSTM model testing
def test_lstm(model_type: str = 'lstm'):
    """
    Test LSTM model architecture.

    Validates:
    - Model forward pass works correctly
    - Output shapes match expectations
    - No NaN or gradient issues

    Args:
    model_type (str): Type of model to test ('lstm' or 'transformer')
    """
    print("\n" + SEP)
    print("TESTING LSTM MODEL ARCHITECTURE")
    print(SEP + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Create model based on model_type parameter
    tokenizer = create_tokenizer()
    model = create_model(model_type, tokenizer.vocab_size)
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
def overfit_sanity_check(model_type: str = 'lstm', num_samples: int = 30, num_epochs: int = 150):
    """
    Sanity check to see if model can overfit a tiny dataset.

    Trains the model on a tiny dataset to confirm the architecture and 
    training pipeline work correctly before full-scale training.

    Args:
    model_type (str): Type of model to test ('lstm' or 'transformer')
    num_samples (int): Number of samples in the tiny dataset
    num_epochs (int): Number of epochs to train for overfitting
    """
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

    tokenizer = create_tokenizer()
    model = create_model(model_type, tokenizer.vocab_size)

    # Train
    checkpoint_dir = Path(__file__).parent / "results" / "lstm_baseline" / "sanity_check"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("Training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=0.0005,
        device=device,
        save_path=str(checkpoint_dir / "sanity_model.pt"), 
        pad_idx=0,
        early_stopping_patience=50
    )
    
    # Evaluate
    final_train_acc = history['train_accuracies'][-1]
    final_val_acc = history['val_accuracies'][-1]
    
    # After creating the model
    print("\n" + SEP)
    print("MODEL ARCHITECTURE")
    print(SEP)
    print(model)
    print(SEP + "\n")
    print("\n" + SEP)
    print("SANITY CHECK RESULTS")
    print(SEP)
    print(f"Final train accuracy: {final_train_acc:.2f}%")
    print(f"Final val accuracy: {final_val_acc:.2f}%")
    
    # Determine pass/fail
    threshold = 95.0
    passed = final_val_acc >= threshold
    
    if passed:
        print(f"\nPASS: Model achieved {final_val_acc:.2f}% (>= {threshold}%)")
        print("Architecture is working correctly. Model can memorize.")
    else:
        print(f"\nFAIL: Model only achieved {final_val_acc:.2f}% (< {threshold}%)")
        print("Possible issues:")
        print("- Model architecture may have bugs.")
    
    print(SEP + "\n")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    return passed
    
# Training
def train_model_on_study(model_type: str, study: str, dataset_split: str = "train", num_epochs: int = 20, batch_size: int = 32, learning_rate: float = 0.001, data_dir: str = "datasets") -> dict:
    """
    Train an LSTM or Transformer model on a specified dataset split.

    Args:
    model_type (str): Type of model to train ('lstm' or 'transformer')
    study (str): Which study dataset to use ('study1' or 'study2')
    dataset_split (str): Which split to train on ('train', 'val', or 'ood')
    num_epochs (int): Number of training epochs
    batch_size (int): Training batch size
    learning_rate (float): Learning rate for optimizer
    data_dir (str): Base directory where datasets are stored
    """
    print("\n" + SEP)
    print(f"TRAINING {model_type.upper()} ON {study.upper()} - {dataset_split.upper()}")
    print(SEP + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_path = Path(data_dir) / study / f"{dataset_split}.json"
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    pipeline = MathDataPipeline(data_dir=data_dir, batch_size=batch_size)
    train_file = f"{study}/train.json"
    val_file = f"{study}/val.json"
    train_loader = pipeline.get_dataloaders_file(train_file, shuffle=True)
    val_loader = pipeline.get_dataloaders_file(val_file, shuffle=False)
    
    tokenizer = create_tokenizer()
    model = create_model(model_type, tokenizer.vocab_size)
    
    if model_type == 'lstm':
        results_base = Path(__file__).parent / "results" / "lstm_baseline"
    elif model_type == 'transformer':
        results_base = Path(__file__).parent / "results" / "transformer"
    else:
        results_base = Path(__file__).parent / "results" / "unknown_model"
    checkpoint_dir = results_base / study
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / f"{dataset_split}_best_model.pt"
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=str(save_path),
        pad_idx=0,
        early_stopping_patience=25
    )
    
    history_path = checkpoint_dir / f"{dataset_split}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete for {study} ({dataset_split})!")
    print(f"Best model saved to: {save_path}")
    print(f"History saved to: {history_path}")
    
    return history

# Evaluation
def evaluate_model(model_type: str, study: str, dataset_split: str, checkpoint_path: str, device: str = "cpu") -> dict:
    """
    Evaluate a trained model on a specified dataset split.

    Args:
    model_type (str): Type of model to evaluate ('lstm' or 'transformer')
    study (str): Which study dataset to evaluate on ('study1' or 'study2')
    dataset_split (str): Which split to evaluate on ('train', 'val', or 'ood')
    checkpoint_path (str): Path to the trained model checkpoint
    device (str): Device to run evaluation on ('cpu' or 'cuda')
    """
    print("\n" + SEP)
    print(f"EVALUATING {study.upper()} - {dataset_split.upper()}")
    print(SEP + "\n")

    # Load model
    tokenizer = create_tokenizer()
    vocab_size = tokenizer.vocab_size
    max_input_len = TRAINING_CONFIG['max_input_len']
    max_output_len = TRAINING_CONFIG['max_output_len']

    tokenizer = create_tokenizer()
    model = create_model(model_type, vocab_size)
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
            target = str(sample["output"])  # Convert to string for comparison

            try:
                # Encode source expression
                src_ids = tokenizer.encode(expr)
                src_ids = src_ids + [tokenizer.pad_idx] * (max_input_len - len(src_ids))
                src_ids = src_ids[:max_input_len]
                src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

                # decoding: start with <SOS> token, generate tokens
                dec_token_ids = [tokenizer.sos_idx]
                pred_ids = []

                dec_token_ids = [tokenizer.sos_idx]
                for step in range(max_output_len - 1):
                    # Pad current sequence
                    current_dec = dec_token_ids + [tokenizer.pad_idx] * (max_output_len - len(dec_token_ids))
                    current_dec = current_dec[:max_output_len]
                    dec_tensor = torch.tensor([current_dec], dtype=torch.long).to(device)

                    logits = model(src_tensor, dec_tensor) 

                    # Get next token 
                    nxt_token_id = logits[0, len(dec_token_ids) - 1, :].argmax(dim=-1).item()

                    # stop at <EOS> or <PAD>
                    if nxt_token_id == tokenizer.eos_idx or nxt_token_id == tokenizer.pad_idx:
                        break

                    # add to sequence for next step
                    dec_token_ids.append(nxt_token_id)

                # Decode predicted 
                pred_str = tokenizer.decode(dec_token_ids[1:])

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
                print(f"Input: {e['input']}")
                print(f"Error: {e['error']}\n")
            else:
                print(f"Input    : {e['input']}")
                print(f"Expected : {e['expected']}")
                print(f"Got      : {e['predicted']}\n")
    
    return {"accuracy": acc, "correct": correct, "total": total}

# Main Entry point
def main() -> None:
    """
    Main entry point for the pipeline.

    This function parses command-line arguments to determine which mode to run in:
    - verify: Generate and save verification samples for manual inspection.
    - generate: Create controlled datasets for the generalization studies.
    - sanity: Run a sanity check to see if the model can overfit a tiny dataset.
    - train: Train the model on the specified datasets. 
    - eval: Evaluate the trained model on validation and OOD sets.
    - test: Run unit tests for the tokenizer and model architecture.
    """
    parser = argparse.ArgumentParser(
        description="Math Reasoning LSTM Baseline"
    )
    parser.add_argument('--mode', type=str, default='train',
                choices=['verify', 'generate', 'sanity', 'train', 'eval', 'test'],
                help='Mode to run')
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'transformer'],
                    help='Model architecture to use')
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    # Mode routing
    if args.mode == "generate":
        generate_study_datasets()

    elif args.mode == "sanity":
        overfit_sanity_check(model_type=args.model)

    elif args.mode == "train":
        print("\n" + SEP)
        print("TRAINING MODELS")
        print(SEP + "\n")
        lr = TRAINING_CONFIG['learning_rate'].get(args.model, 0.001) if args.lr is None else args.lr
        train_model_on_study(args.model, "study1", "train", num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=lr)
        train_model_on_study(args.model, "study2", "train", num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=lr)

    elif args.mode == "eval":
        print("\n" + SEP)
        print("EVALUATING MODELS")
        print(SEP + "\n")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine checkpoint base directory
        if args.model == 'lstm':
            checkpoint_base = Path(__file__).parent / "results" / "lstm_baseline"
        elif args.model == 'transformer':
            checkpoint_base = Path(__file__).parent / "results" / "transformer"
        else:
            checkpoint_base = Path(__file__).parent / "results" / f"{args.model}_baseline"
        # Dictionary to store all results
        eval_results = {}

        # Evaluate Study 1
        checkpoint_s1 = checkpoint_base / "study1" / "train_best_model.pt"
        if checkpoint_s1.exists():
            val_result = evaluate_model(args.model, "study1", "val", str(checkpoint_s1), device=device)
            ood_result = evaluate_model(args.model, "study1", "ood", str(checkpoint_s1), device=device)
            eval_results['study1'] = {
            'val': val_result['accuracy'],
            'ood': ood_result['accuracy']
        }
        else:
            print(f"Study 1 checkpoint not found: {checkpoint_s1}")

        # Evaluate Study 2
        checkpoint_s2 = checkpoint_base / "study2" / "train_best_model.pt"
        if checkpoint_s2.exists():
            val_result = evaluate_model(args.model, "study2", "val", str(checkpoint_s2), device=device)
            ood_result = evaluate_model(args.model, "study2", "ood", str(checkpoint_s2), device=device)
            eval_results['study2'] = {
                'val': val_result['accuracy'],
                'ood': ood_result['accuracy']
            }
        else:
            print(f"Study 2 checkpoint not found: {checkpoint_s2}")

        # Save evaluation results
        eval_results_path = checkpoint_base / "evaluation_results.json"
        with open(eval_results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n✅ Evaluation results saved to: {eval_results_path}")

    print("\n" + SEP)
    print("ALL DONE!")
    print(SEP)

if __name__ == "__main__":
    main()