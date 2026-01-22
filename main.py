import os
from data.generate import generate_lvl1, generate_lvl2, generate_lvl3, save_json

def main():
    """Main function to generate math reasoning datasets"""
    print("🚀 Generating datasets for math reasoning research...\n")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Level 1
    print("🔄 Generating Level 1 (LSTM-friendly: +, -, *)...")
    lvl1_data = generate_lvl1(num_samples=5000, max_num=15, seed=42)
    save_json(
        {"metadata": {"level": "lvl1", "total_samples": len(lvl1_data)}, "data": lvl1_data},
        os.path.join(output_dir, "level1_dataset.json")
    )
    print(f"✅ Level 1: {len(lvl1_data)} samples saved\n")
    
    # Generate Level 2
    print("🔄 Generating Level 2 (Medium: +, -, *, /)...")
    lvl2_data = generate_lvl2(num_samples=5000, max_num=15, seed=42)
    save_json(
        {"metadata": {"level": "lvl2", "total_samples": len(lvl2_data)}, "data": lvl2_data},
        os.path.join(output_dir, "level2_dataset.json")
    )
    print(f"✅ Level 2: {len(lvl2_data)} samples saved\n")
    
    # Generate Level 3
    print("🔄 Generating Level 3 (Hard: +, -, *, /, parentheses)...")
    lvl3_data = generate_lvl3(num_samples=5000, max_num=15, seed=42, parentheses_prob=0.7)
    save_json(
        {"metadata": {"level": "lvl3", "total_samples": len(lvl3_data)}, "data": lvl3_data},
        os.path.join(output_dir, "level3_dataset.json")
    )
    print(f"✅ Level 3: {len(lvl3_data)} samples saved\n")
    
    # Print samples
    print("="*60)
    print("SAMPLE OUTPUTS")
    print("="*60)
    
    print("\n📊 Level 1 Samples:")
    for item in lvl1_data[:3]:
        print(f"  {item['input']} = {item['output']}")
    
    print("\n📊 Level 2 Samples:")
    for item in lvl2_data[:3]:
        print(f"  {item['input']} = {item['output']}")
    
    print("\n📊 Level 3 Samples:")
    for item in lvl3_data[:3]:
        print(f"  {item['input']} = {item['output']}")
    
    print("\n" + "="*60)
    print("✅ All datasets generated successfully!")
    print("="*60)

if __name__ == "__main__":
    main()