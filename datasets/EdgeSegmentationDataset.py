import os
import cv2
import torch
from torch.utils.data import Dataset
from scipy.ndimage import convolve
import numpy as np
from PIL import Image  # Import PIL for transformations
import heapq
import sys

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
        #matching_files = [f for f in input_files if any(f.split('.')[0] in t for t in target_files)]

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

        input_image_array = np.array(input_image_pil)
        target_mask_array = np.array(target_mask_pil)

        # print("After transformations: ")
        # print(input_image_array.shape, target_mask_array.shape) #1, 1024,1024
        # print('Input image info: ', type(input_image_array), input_image_array.dtype) # <class 'numpy.ndarray'> float32
        # print('Target mask info: ', type(target_mask_array), target_mask_array.dtype) # <class 'numpy.ndarray'> float32
        # print(len(np.unique(target_mask_array)), np.min(target_mask_array), np.max(target_mask_array)) # 10, 0.0, 1.0
        # print(np.min(input_image_array), np.max(input_image_array)) # 0.0 0.92156863

        # We want the target to be a binary mask
        #target_mask_array = (target_mask_array >= 0.5).astype(np.float32)

        # print('After converting target mask to binary array:')
        # print(target_mask_array.shape) #1, 1024,1024
        # print('Target mask info: ', type(target_mask_array), target_mask_array.dtype) # <class 'numpy.ndarray'> float32
        # print(len(np.unique(target_mask_array)), np.min(target_mask_array), np.max(target_mask_array)) #2 0.0 1.0

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_image_array, dtype=torch.float32)  
        target_tensor = torch.tensor(target_mask_array, dtype=torch.float32)  

        # print(f"Input tensor min: {input_tensor.min()}, max: {input_tensor.max()}")
        # print(f"Target tensor min: {target_tensor.min()}, max: {target_tensor.max()}")
        
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

            # print("Original shapes of input and target: ")
            # print(input_image.shape, target_mask.shape) #512,512
            # print(len(np.unique(target_mask)), np.min(target_mask), np.max(target_mask)) # 5, 0, 255
            # print(np.min(input_image), np.max(input_image)) # 12, 255

            if input_image is None or target_mask is None:
                print(f"Skipping {filename}: Failed to load image/mask.")
                continue

            # Resize input image and target mask
            input_image = cv2.resize(input_image, self.image_size)
            target_mask = cv2.resize(target_mask, self.image_size)

            # print("After resizing of input and target: ")
            # print(input_image.shape, target_mask.shape) #1024,1024
            # print(len(np.unique(target_mask)), np.min(target_mask), np.max(target_mask)) # 41, 0, 255
            # print(np.min(input_image), np.max(input_image)) # 15, 255

            # Normalize to [0, 1]
            input_image = input_image.astype(np.float32) / 255.0
            target_mask = target_mask.astype(np.float32) / 255.0

            # print("After normalizing input and target: ")
            # print(input_image.shape, target_mask.shape) #1024,1024
            # print('Input image info: ', type(input_image), input_image.dtype) # <class 'numpy.ndarray'> float32
            # print('Target mask info: ', type(target_mask), target_mask.dtype) # <class 'numpy.ndarray'> float32
            # print(len(np.unique(target_mask)), np.min(target_mask), np.max(target_mask)) # 41, 0.0, 1.0
            # print(np.min(input_image), np.max(input_image)) # 0.05882353 1.0

            # Convert to PIL format (useful for applying transformations)
            # input_image_pil = Image.fromarray((input_image * 255).astype('uint8'))
            # target_mask_pil = Image.fromarray((target_mask * 255).astype('uint8'))
            input_image_pil = Image.fromarray(input_image)
            target_mask_pil = Image.fromarray(target_mask)

            data.append((input_image_pil, target_mask_pil, filename))
            
            if dataset_size is not None:
                if len(data) == dataset_size: 
                    break

        if not data:
            raise ValueError("No valid matching image-mask pairs found.")
        
        return data
