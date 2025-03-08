import os
import shutil
import random
import sys
import torch
import datetime
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torchvision.transforms as transforms
from PIL import Image
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

from utils.utils import load_config
from datasets.EdgeSegmentationDataset import EdgeSegmentationDataset
from models.EdgeSegmentationCNN import EdgeSegmentationCNN 

# Load Configurations
config = load_config()

# -----------------------------------------------------------
# Preprocess and Load Dataset
# -----------------------------------------------------------
def load_dataset(input_folder_path, target_folder_path, dataset_size=None):
    print(f"Loading input dataset from: {input_folder_path} and targets from: {target_folder_path}")
    
    # Define Transformations
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),        # Randomly flip images horizontally
        # transforms.RandomVerticalFlip(p=0.5),          # Randomly flip images vertically
        # transforms.RandomRotation(degrees=30),         # Rotate images randomly within 30 degrees
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Change brightness and contrast
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate images
        # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Apply Gaussian blur
        transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
        # transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1] --> dont want that for BCE loss
    ])

    # Initialize dataset
    dataset = EdgeSegmentationDataset(
        input_dir=input_folder_path,
        target_dir=target_folder_path,
        image_size=(1024, 1024),
        transform=transform,
        dataset_size=dataset_size  # Optional: Limit dataset size
    )

    if len(dataset) == 0:
        raise ValueError("Error: Dataset is empty. Check the dataset folder.")
    
    print(f"Loaded {len(dataset)} images from dataset.")
    return dataset

def split_dataset(dataset, folder_path):
    train_size = ceil(len(dataset) * (1 - config["TEST_SPLIT_RATIO"]))
    test_size = len(dataset) - train_size  # Ensure sizes sum to total
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    test_file_path = os.path.join(folder_path, "test_dataset.txt")

    with open(test_file_path, "w") as file:
        for idx in test_dataset.indices:
            _, _, filename = dataset.data[idx]  # Extract filename correctly
            file.write(f"{filename}\n")

    print(f"Test file list saved to {test_file_path}")

    return train_dataset, test_dataset

def save_split_dataset(input_dataset_path, target_dataset_path, train_input_dir, train_target_dir, test_input_dir, test_target_dir, train_ratio=0.8):
    # Define output directories
    # train_input_dir = os.path.join(output_path, "train/input")
    # train_target_dir = os.path.join(output_path, "train/target")
    # test_input_dir = os.path.join(output_path, "test/input")
    # test_target_dir = os.path.join(output_path, "test/target")

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

# -----------------------------------------------------------
# Save Before and After Images
# -----------------------------------------------------------
def save_combined_image(original, output, target, filename):
    # if len(original.shape) == 4:
    #     original = original.squeeze(0)
    # if len(output.shape) == 4:
    #     output = output.squeeze(0)

    # original = original.cpu().detach().numpy()
    # output = output.cpu().detach().numpy()
    # target = target.cpu().detach().numpy()

    # if len(original.shape) == 2:
    #     original = original[:, :, np.newaxis]
    # if len(output.shape) == 2:
    #     output = output[:, :, np.newaxis]
    # if len(target.shape) == 2:
    #     target = target[:, :, np.newaxis]

    # combined_image = np.concatenate((original, output, target), axis=1)
    # plt.imsave(filename, combined_image.squeeze(), cmap='gray')

    original = original.cpu().detach().numpy()
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    if len(original.shape) == 2:
        original = original[:, :, np.newaxis]
    if len(output.shape) == 2:
        output = output[:, :, np.newaxis]
    if len(target.shape) == 2:
        target = target[:, :, np.newaxis]

    images = [original, output, target]
    titles = ["(a)", "(b)", "(c)"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# -----------------------------------------------------------
# Save Model Folder Function
# -----------------------------------------------------------
def create_folder(folder_name):
    # Create a timestamped folder for this training session
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generates folder with folder name 
    folder_path = os.path.join(folder_name, timestamp)
    os.makedirs(folder_path, exist_ok=True)

    print(f"Created training session folder: {folder_path}")

    return folder_path