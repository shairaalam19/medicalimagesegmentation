import torch
import torch.nn as nn
import torch.nn.functional as F

def get_edges(x): 
    edge_scale = 1.0

    # Apply the Roberts operator
    edges = apply_roberts_operator(x)
    
    # Amplify edges
    edges *= edge_scale

    # Apply thresholding to get binary edges (0 or 1)
    edges = threshold_edges(edges)

    # Fill inside the edges 
    edges = fill_inside_edges(edges)

    # Resize tensors to match dimensions
    edges = F.interpolate(edges, size=x.size()[-2:], mode="bilinear", align_corners=False)
    
    return edges

def apply_roberts_operator(x):
    # # Define Roberts filters for horizontal and vertical gradients
    # roberts_cross_v = torch.tensor([[1, 0], [0, -1]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)
    # roberts_cross_h = torch.tensor([[0, 1], [-1, 0]], dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(0)

    # # Apply convolution to detect edges
    # roberts_edges_v = F.conv2d(x, roberts_cross_v, padding=1)
    # roberts_edges_h = F.conv2d(x, roberts_cross_h, padding=1)

    # # Calculate the magnitude of the gradients (edge strength)
    # edges = torch.sqrt(roberts_edges_v**2 + roberts_edges_h**2)

    # return edges

    batch_size, channels, height, width = x.shape
    roberts_cross_v = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32).view(1, 1, 2, 2).to(x.device)
    roberts_cross_h = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32).view(1, 1, 2, 2).to(x.device)

    # Apply Roberts operator to each channel separately
    edges_v = torch.cat([F.conv2d(x[:, i:i+1, :, :], roberts_cross_v, padding=1) for i in range(channels)], dim=1)
    edges_h = torch.cat([F.conv2d(x[:, i:i+1, :, :], roberts_cross_h, padding=1) for i in range(channels)], dim=1)

    edges = torch.sqrt(edges_v ** 2 + edges_h ** 2)

    return edges

def threshold_edges(edges):
    # Dynamically compute the threshold based on the image's statistics
    mean_value = edges.mean()
    std_value = edges.std()

    # Threshold as a multiple of the standard deviation above the mean
    threshold_value = mean_value + 2.0 * std_value  # Higher facter focuses on prominent edges and lower factor captures finer details

    # Static Thresholding
    # threshold_value = edges.max() * 0.3  # Adjust this factor for more/less aggressive detection
    
    # Set edges inside the threshold to 1 (region of interest), else 0
    edges = (edges >= threshold_value).float()

    return edges

def fill_inside_edges(edges):
    # batch_size, channels, height, width = edges.shape
    # dilation_kernel = torch.ones((1, 1, 7, 7), device=edges.device, dtype=edges.dtype)
    # erosion_kernel = torch.ones((1, 1, 5, 5), device=edges.device, dtype=edges.dtype)

    # # Dilate the edges to close gaps
    # edges = F.conv2d(edges, dilation_kernel, padding=3)

    # # Fill the interior by detecting connected regions
    # edges = F.conv2d(edges, erosion_kernel, padding=2)

    # edges = (edges > 0).float()

    # return edges

    batch_size, channels, height, width = edges.shape
    
    # Define dilation and erosion kernels
    dilation_kernel = torch.ones((channels, 1, 7, 7), device=edges.device, dtype=edges.dtype)
    erosion_kernel = torch.ones((channels, 1, 5, 5), device=edges.device, dtype=edges.dtype)

    # Apply dilation to close gaps
    edges_dilated = torch.cat([
        F.conv2d(edges[:, i:i+1, :, :], dilation_kernel[i:i+1, :, :, :], padding=3) 
        for i in range(channels)
    ], dim=1)

    # Apply erosion to fill interiors
    edges_filled = torch.cat([
        F.conv2d(edges_dilated[:, i:i+1, :, :], erosion_kernel[i:i+1, :, :, :], padding=2) 
        for i in range(channels)
    ], dim=1)

    # Threshold to get binary mask (0 or 1)
    edges_filled = (edges_filled > 0).float()

    return edges_filled