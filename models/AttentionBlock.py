import torch
import torch.nn as nn
import torch.nn.functional as F

from models.EdgeSegmentation import get_edges

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1) # passes input feature map through a 1x1 convolution to compute intermediate attention weights 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1) # refines attention map with anotehr 1x1 convolution
        self.sigmoid = nn.Sigmoid() # sigmoid to normalize the attention values to a range between 0 and 1

    def forward(self, x):
        attention = self.conv1(x)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention # scales the input by its learned attention weights 

class EdgeAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.edge_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # Edge-specific convolution

        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()  # Attention weights in [0, 1]
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        # Extract edges from the multi-channel input
        edges = get_edges(x)  # Shape: [batch_size, in_channels, height, width]

        # Process the original features
        features = self.conv(x)
        features = self.relu(features)

        # Compute attention weights using edge information
        attention_weights = self.attention(edges)

        # Apply attention to the features
        output = features * attention_weights

        return output