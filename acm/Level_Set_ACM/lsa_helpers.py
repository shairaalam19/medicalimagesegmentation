import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from PIL import Image

# ---- Helper functions for working with level set acms

# Displaying image - to plot the segmentation of the contour
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

# helper to create initial circular segmentation mask given square image size
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

def create_circle_mask(square_size, center, radius):
    """
    Create a square binary mask with a circle of ones.

    Parameters:
    - square_size (int): Size of the square (square_size x square_size).
    - center (tuple): Coordinates of the circle center (x, y).
    - radius (float): Radius of the circle.

    Returns:
    - mask (numpy.ndarray): Binary mask as a 2D NumPy array.
    """
    # Create a grid of coordinates
    x = np.arange(square_size)
    y = np.arange(square_size)
    xx, yy = np.meshgrid(x, y)

    # Calculate the distance from the center for each point
    distance_from_center = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)

    # Create the mask: 1 inside and on the circle, 0 outside
    mask = (distance_from_center <= radius).astype(int)

    return mask

# Helper function for defining intial contour given center coordinates and radius
def CreateInitCircContour(c_x, c_y, radius):
    # Defining an initial circular contour around the object of interest
    s = np.linspace(0, 2 * np.pi, 400) # Defining angles between 0 and 2*pi
    r = c_x + radius * np.sin(s) # Defining x coordinates along the circle
    c = c_y + radius * np.cos(s) # Defining y coordinates along the circle
    init = np.array([r, c]).T # Each row represents a coordinate point (x,y)
    #print ("Shape of original contour: ", init.shape) # (400,2)
    return init

# Helper function to plot the contour on the image
def DisplayACMResult(img, init, snake):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray) # grayscale image (not gaussian blurred image)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3) # red dashed line for initial contour
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3) # blue line for fitted contour
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()

# Helper function to get center of an image and the largest possible radius
def GetDefaultInitContourParams(img):
    # Dimensions and parameters
    rows, cols = img.shape[:2]
    center_row, center_col = rows // 2, cols // 2
    radius = min(rows, cols) // 2  # Example radius
    return center_row, center_col, radius

def AnalyzeCoordinates(image_path):
    img = Image.open(image_path)

    # Display the image
    plt.imshow(img)
    plt.title("Hover over the image to get coordinates")
    coords = plt.ginput(1)  # Click to get one coordinate point
    print(f"Coordinates: {coords}")

# Evaluation functions

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

