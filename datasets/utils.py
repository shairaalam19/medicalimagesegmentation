import os
import sys
import torch
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
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
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

def split_dataset(dataset):
    train_size = ceil(len(dataset) * (1 - config["TEST_SPLIT_RATIO"]))
    test_size = len(dataset) - train_size  # Ensure sizes sum to total
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    return train_dataset, test_dataset

# -----------------------------------------------------------
# Save Before and After Images
# -----------------------------------------------------------
def save_combined_image(original, output, target, filename):
    # if len(original.shape) == 4:
    #     original = original.squeeze(0)
    # if len(output.shape) == 4:
    #     output = output.squeeze(0)

    original = original.cpu().detach().numpy()
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    if len(original.shape) == 2:
        original = original[:, :, np.newaxis]
    if len(output.shape) == 2:
        output = output[:, :, np.newaxis]
    if len(target.shape) == 2:
        target = target[:, :, np.newaxis]

    combined_image = np.concatenate((original, output, target), axis=1)
    plt.imsave(filename, combined_image.squeeze(), cmap='gray')