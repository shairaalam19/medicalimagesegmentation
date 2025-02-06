import os
import cv2
import torch
from torch.utils.data import Dataset
from scipy.ndimage import convolve
import numpy as np
from PIL import Image  # Import PIL for transformations
import heapq

class EdgeSegmentationDataset(Dataset):
    def __init__(self, input_dir, target_dir, image_size=(1024, 1024), transform=None, dataset_size=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_size = image_size
        self.transform = transform
        self.data = []

        # Get sorted lists of filenames in input and target directories
        input_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        target_files = set(f for f in os.listdir(target_dir) if f.endswith(('.png', '.jpg', '.jpeg')))  
        
        matching_files = [f for f in input_files if f in target_files]

        self.data = self.create_dataset(matching_files, dataset_size)

        # Limit dataset size if specified
        if dataset_size is not None:
            if dataset_size > len(self.data):
                raise ValueError(f"Requested dataset size ({dataset_size}) exceeds the available valid data ({len(self.data)}).")
            self.data = self.data[:dataset_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_image_pil, target_mask_pil, filename = self.data[idx]

        # Apply transformations (if provided)
        if self.transform:
            input_image_pil = self.transform(input_image_pil)
            target_mask_pil = self.transform(target_mask_pil)

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(np.array(input_image_pil), dtype=torch.float32)  
        target_tensor = torch.tensor(np.array(target_mask_pil), dtype=torch.float32)  

        return input_tensor, target_tensor, filename
    
    def create_dataset(self, matching_files, dataset_size): 
        data = []
        # Verify and load valid image-mask pairs
        for filename in matching_files:
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)

            if not os.path.exists(input_path) or not os.path.exists(target_path):
                print(f"Skipping {filename}: File not found.")
                continue

            input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            target_mask = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

            if input_image is None or target_mask is None:
                print(f"Skipping {filename}: Failed to load image/mask.")
                continue

            # Resize input image and target mask
            input_image = cv2.resize(input_image, self.image_size)
            target_mask = cv2.resize(target_mask, self.image_size)

            # Normalize to [0, 1]
            input_image = input_image.astype(np.float32) / 255.0
            target_mask = target_mask.astype(np.float32) / 255.0

            # Convert to PIL format (useful for applying transformations)
            input_image_pil = Image.fromarray((input_image * 255).astype('uint8'))
            target_mask_pil = Image.fromarray((target_mask * 255).astype('uint8'))

            data.append((input_image_pil, target_mask_pil, filename))
            
            if dataset_size is not None:
                if len(data) == dataset_size: 
                    break

        if not data:
            raise ValueError("No valid matching image-mask pairs found.")
        
        return data
