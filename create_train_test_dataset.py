import os
import shutil
import random

def split_dataset(input_dataset_path, target_dataset_path, output_path, train_ratio=0.8):
    # Define output directories
    train_input_dir = os.path.join(output_path, "train/input")
    train_target_dir = os.path.join(output_path, "train/target")
    test_input_dir = os.path.join(output_path, "test/input")
    test_target_dir = os.path.join(output_path, "test/target")

    # Create directories if they don't exist
    for directory in [train_input_dir, train_target_dir, test_input_dir, test_target_dir]:
        os.makedirs(directory, exist_ok=True)

    # Get all input file names
    input_files = sorted(os.listdir(input_dataset_path))
    target_files = sorted(os.listdir(target_dataset_path))
    
    # Ensure input and target files match
    assert input_files == target_files, "Mismatch between input and target files"
    
    # Shuffle and split dataset
    random.shuffle(input_files)
    split_idx = int(len(input_files) * train_ratio)
    train_files = input_files[:split_idx]
    test_files = input_files[split_idx:]

    # Move files to respective directories
    for file in train_files:
        shutil.copy(os.path.join(input_dataset_path, file), os.path.join(train_input_dir, file))
        shutil.copy(os.path.join(target_dataset_path, file), os.path.join(train_target_dir, file))
    
    for file in test_files:
        shutil.copy(os.path.join(input_dataset_path, file), os.path.join(test_input_dir, file))
        shutil.copy(os.path.join(target_dataset_path, file), os.path.join(test_target_dir, file))
    
    print(f"Dataset split completed. Train: {len(train_files)}, Test: {len(test_files)}")
