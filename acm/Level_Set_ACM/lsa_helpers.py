import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

# ---- Helper functions for working with level set acms

# Displaying image
def displayImage(img, title, grayscale=False, save_dir=None):
    if (grayscale):
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.title(title)

    if save_dir:
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    plt.close()
    #plt.show()

# Cropping the image so that it is square
def crop_to_square(img):
    
    # Get the dimensions of the image
    width, height = img.size
    
    # Determine the size of the square crop
    crop_size = min(width, height)
    
    # Calculate the coordinates for cropping the image
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    
    # Crop the image to a square
    img_cropped = img.crop((left, top, right, bottom))
    
    return img_cropped

# helper to check image has only 0s and 1s
def is_binary_mask(arr):
    unique_elements = np.unique(arr)
    return np.array_equal(unique_elements, [0, 1]) or np.array_equal(unique_elements, [0]) or np.array_equal(unique_elements, [1])

# helper to create initial circular segmentation mask
def create_circular_mask(square_size, factor=1):
    # Create a grid of coordinates
    x, y = np.ogrid[:square_size, :square_size]
    
    # Calculate the center and radius of the circle
    center = square_size // 2
    radius = (square_size*factor) // 2
    
    # Create the mask based on the distance from the center
    mask = (x - center)**2 + (y - center)**2 <= radius**2
    
    # Convert the boolean mask to an integer mask (0s and 1s)
    binary_mask = mask.astype(int)
    
    return binary_mask

# Evalustion functions

def normalize_mask(mask):
    """
    Normalize a mask with values 0 and 256 to values 0 and 1.
    """
    return (mask > 0).astype(np.float32)

def iou_score(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) score between two binary masks.
    
    Parameters:
    mask1, mask2: numpy arrays
        Binary masks with values 0 or 1.
        
    Returns:
    float
        IoU score.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0  # Avoid division by zero
    iou = intersection / union
    return iou

# Higher dice scores are better
# Perfect dice scores are 1
def dice_score(mask1, mask2):
    """
    Calculate the Dice score between two binary masks using IoU score.
    
    Parameters:
    mask1, mask2: numpy arrays
        Binary masks with values 0 or 1.
        
    Returns:
    float
        Dice score.
    """
    iou = iou_score(mask1, mask2)
    dice = 2 * iou / (1 + iou) if iou > 0 else 0.0
    return dice

