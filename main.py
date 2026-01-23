import os
from data.generate import generate_lvl1, generate_lvl2, generate_lvl3, save_json
from data.tokenizer import create_tokenizer
from data.dataloader import get_dataloaders


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
    
    dataloaders = get_dataloaders(batch_size=128)
    
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


def main():
    """Main function to generate math reasoning datasets and test pipeline"""
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


if __name__ == "__main__":
    main()