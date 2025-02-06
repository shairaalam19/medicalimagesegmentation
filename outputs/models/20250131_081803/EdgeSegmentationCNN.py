import torch
import torch.nn as nn
import torch.nn.functional as F

from models.AttentionBlock import AttentionBlock
from models.EdgeSegmentation import get_edges

class EdgeSegmentationCNN(nn.Module):
    def __init__(self):
        super(EdgeSegmentationCNN, self).__init__()
        # Encoder: downsamples to extract features and reduces the spatial dimensions 
        self.encoder = nn.Sequential(
            # COARSE attention 
            # Linear transformation to input feature map compressing/expanding feature space 
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2), # downsampling

            AttentionBlock(32, 32),  # Suppresses irrelevant / less important features by assigning lowe rattention weights 

            # REFINED attention
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Bottleneck: further processes features and focuses on important features (more attention)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            AttentionBlock(128, 128),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder: upsamples features to reconstruct an edge-detected image 
        self.decoder = nn.Sequential(
            # Increases spatial resolution of feature map by 2 (upsampling from bottleneck to start reconstructing spatial structure of image)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(), # Introduces non-linearity to help learn complex patterns 
            AttentionBlock(32, 32),  # Add attention block in the decoder
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        edges = get_edges(x)

        # Pass through the encoder-decoder architecture
        encoded = self.encoder(edges)
        bottleneck_output = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck_output)
        
        return decoded