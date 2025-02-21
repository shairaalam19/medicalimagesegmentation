import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim

from models.AttentionBlock import Attention, EdgeAttention
from models.EdgeSegmentation import get_edges

class EdgeSegmentationCNN(nn.Module):
    def __init__(self, edge_attention=False, define_edges_before=False, define_edges_after=False):
        super(EdgeSegmentationCNN, self).__init__()
        if edge_attention: 
            print("Using Edge Attention Block")
            self.attention = EdgeAttention
        else: 
            print("Using Base Attention Block")
            self.attention = Attention

        self.define_edges_before = define_edges_before
        self.define_edges_after = define_edges_after

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

        # Decoder: upsamples features to reconstruct an edge-detected image 
        self.decoder = nn.Sequential(
            # Increases spatial resolution of feature map by 2 (upsampling from bottleneck to start reconstructing spatial structure of image)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(), # Introduces non-linearity to help learn complex patterns 
            self.attention(32, 32),  # Add attention block in the decoder
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.define_edges_before: 
            print("Defining edges before encoder")
            x = get_edges(x)

        # Pass through the encoder-decoder architecture
        encoded = self.encoder(x)
        bottleneck_output = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck_output)

        if self.define_edges_after: 
            print("Defining edges after decoder")
            decoded = get_edges(decoded)
        
        return decoded