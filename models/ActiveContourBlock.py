import torch
import torch.nn as nn
from models.RunLSA import *

class ActiveContourLayer(nn.Module):
    def __init__(self):
        super(ActiveContourLayer, self).__init__()

    def forward(self, intensity_images, initial_segmentations, acm_params):
        """
        intensity_images: (B, 1, H, W)
        initial_segmentations: (B, 1, H, W)
        acm_params: (B, 3) - Each row has (num_iter, nu, mu)
        Returns:
        final_masks: (B, 1, H, W)
        """
        B, D, H, W = intensity_images.shape
        final_masks = torch.zeros((B, H, W), device=intensity_images.device, dtype=intensity_images.dtype)

        for i in range(B):
            num_iter, nu, mu = acm_params[i]  # Extract hyperparameters for sample i
            print("ACM Hyperparameters: ", num_iter, nu, mu)
            # Ensure all operations remain in PyTorch
            final_masks[i] = acm_layer(
                intensity_images[i].squeeze(0), # (H, W)
                initial_segmentations[i].squeeze(0), # (H, W)
                num_iter, nu, mu
            ) # (H, W)

        return final_masks.unsqueeze(1)  # (B, 1, H, W), gradients tracked
