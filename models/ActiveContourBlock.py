import torch
import torch.nn as nn
import os
import sys

cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../acm/model_utils')))

from RunLSA import acm_layer

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

            num_iter = torch.tensor(10, dtype=torch.float32) # testing on low defined number of iterations

            # print("ACM Hyperparameters: ", num_iter, nu, mu)
            # # ACM Hyperparameters:  tensor(258., grad_fn=<UnbindBackward0>) tensor(6.6076, grad_fn=<UnbindBackward0>) tensor(0.6891, grad_fn=<UnbindBackward0>)
            # print("Making sure the num_iter acm hyper-parameter is keeping track of gradients: ", num_iter.requires_grad)

            # Ensure all operations remain in PyTorch
            final_masks[i] = acm_layer(
                intensity_images[i].squeeze(0), # (H, W)
                initial_segmentations[i].squeeze(0), # (H, W)
                num_iter, nu, mu
            ) # (H, W)

        return final_masks.unsqueeze(1)  # (B, 1, H, W), gradients tracked
