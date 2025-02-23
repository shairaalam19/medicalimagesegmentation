import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
import sys

from models.AttentionBlock import Attention, EdgeAttention
from models.EdgeSegmentation import get_edges
from models.ActiveContourBlock import ActiveContourLayer

class EdgeSegmentationCNN(nn.Module):
    def __init__(self, edge_attention=False, define_edges_before=False, define_edges_after=False, use_acm=False):
        super(EdgeSegmentationCNN, self).__init__()
        if edge_attention: 
            print("Using Edge Attention Block")
            self.attention = EdgeAttention
        else: 
            print("Using Base Attention Block")
            self.attention = Attention

        self.define_edges_before = define_edges_before
        self.define_edges_after = define_edges_after
        self.use_acm = use_acm
        self.acm = ActiveContourLayer()

        # Encoder: downsamples to extract features and reduces the spatial dimensions 
        self.encoder = nn.Sequential(
            # COARSE attention 
            # Linear transformation to input feature map compressing/expanding feature space 
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2), # downsampling

            self.attention(32, 32),  # Suppresses irrelevant / less important features by assigning lowe rattention weights 

            # REFINED attention
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Bottleneck: further processes features and focuses on important features (more attention)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            self.attention(128, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Output of bottleneck will be like (H_reduced, W_reduced, 64)
        self.acm_hyperparameter_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global spatial pooling to reduce to 1x1x64
            nn.Flatten(),  # Flatten the 1x1x64 tensor to 64
            nn.Linear(64, 32),  # Reduce to 32 neurons
            nn.ReLU(),
            nn.Linear(32, 3)  # Output 3 values corresponding to ACM hyperparameters - num_iter, nu, mu
        )

        # Decoder: upsamples features to reconstruct an edge-detected image 
        self.decoder = nn.Sequential(
            # Increases spatial resolution of feature map by 2 (upsampling from bottleneck to start reconstructing spatial structure of image)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(), # Introduces non-linearity to help learn complex patterns 
            self.attention(32, 32),  # Add attention block in the decoder
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid() # generates probabilities between 0 and 1
        )

    def forward(self, x):

        intensity_image = x

        if self.define_edges_before: 
            print("Defining edges before encoder")
            x = get_edges(x)

        acm_hyperparameters = None

        # Pass through the encoder-decoder architecture
        encoded = self.encoder(x)
        bottleneck_output = self.bottleneck(encoded)

        if(self.use_acm):
            acm_hyperparameters = self.acm_hyperparameter_generator(bottleneck_output)
            # Need to ensure that the first hyperparameter [num_iter] is an integer
            # acm_hyperparameters[:, 0] = torch.round(acm_hyperparameters[:, 0])
            # # Enforcing a range to focus on for num_iter
            # acm_hyperparameters[:, 0] = torch.clamp(acm_hyperparameters[:, 0], min=1, max=3000)
            # # Enforcing a range to focus on for nu, mu
            # acm_hyperparameters[:, 1:] = torch.clamp(acm_hyperparameters[:, 1:], min=0, max=100)

            # Ensure that the first hyperparameter [num_iter] is an integer
            acm_hyperparameters = torch.cat((
                torch.round(acm_hyperparameters[:, 0]).unsqueeze(1),  # Round and keep the first column
                acm_hyperparameters[:, 1:]  # Keep the rest of the columns unchanged
            ), dim=1)

            # Enforcing a range to focus on for num_iter
            acm_hyperparameters = torch.cat((
                torch.clamp(acm_hyperparameters[:, 0], min=1, max=3000).unsqueeze(1),  # Clamp the first column
                acm_hyperparameters[:, 1:]  # Keep the rest unchanged
            ), dim=1)

            # Enforcing a range to focus on for nu, mu
            acm_hyperparameters = torch.cat((
                acm_hyperparameters[:, 0].unsqueeze(1),  # Keep the first column unchanged
                torch.clamp(acm_hyperparameters[:, 1:], min=0, max=100)  # Clamp the rest of the columns
            ), dim=1)


        decoded = self.decoder(bottleneck_output)

        if self.define_edges_after: 
            print("Defining edges after decoder")
            decoded = get_edges(decoded)

        
        # print("Understanding forward function results: ")

        # print("Intensity Image", type(intensity_image), intensity_image.dtype, intensity_image.shape, intensity_image.min(), intensity_image.max())
        # # Intensity Image <class 'torch.Tensor'> torch.float32 torch.Size([4, 1, 1024, 1024]) tensor(0.) tensor(1.)

        # print("ACM Hyperparameters", type(acm_hyperparameters), acm_hyperparameters.dtype, acm_hyperparameters.shape, acm_hyperparameters.min(), acm_hyperparameters.max())
        # # ACM Hyperparameters <class 'torch.Tensor'> torch.float32 torch.Size([4, 3]) tensor(0.0699, grad_fn=<MinBackward1>) tensor(0.1567, grad_fn=<MaxBackward1>)

        # print("Probability Mask", type(decoded), decoded.dtype, decoded.shape, decoded.min(), decoded.max())
        # # Probability Mask <class 'torch.Tensor'> torch.float32 torch.Size([4, 1, 1024, 1024]) tensor(0.4191, grad_fn=<MinBackward1>) tensor(0.4278, grad_fn=<MaxBackward1>)

        if(not self.use_acm):
            output = decoded
        else:
            # send decoded throuh acm and then the output will be the final probability mask
            # TODO: call acm to get the output
            output = self.acm(intensity_image*255, decoded, acm_hyperparameters)
            print("ACM Result", type(output), output.dtype, output.shape, output.min(), output.max())

        print(output.requires_grad)  # Should be True if gradients are tracked
        #sys.exit()

        return output