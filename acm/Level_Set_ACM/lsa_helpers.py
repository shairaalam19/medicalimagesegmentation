import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage import img_as_ubyte
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter
import tensorflow as tf
import sys

# ---------------------------- Helper functions for working with level set acms -----------------------------------

# ---------- Visualization functions

# Very useful to save PIL images, np array images, rgb images, grayscale images.
# Very useful to save images, binary masks, probability masks, segmentations  
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

# Displaying initial contour (init) and fitted contour (snake) on the image
# I used this during the snakes implementation
def DisplayACMResultSnakes(img, init, snake):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray) # grayscale image (not gaussian blurred image)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3) # red dashed line for initial contour
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3) # blue line for fitted contour
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()

# Displaying initial contour and fitted contour given signed distance maps 
def displayACMResult(image, init_phi, phi, acm_dir):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')  # Display the image

    # Plot the initial contour (ϕ = 0 in init_phi)
    plt.contour(init_phi, levels=[0], colors='red', linewidths=2)

    # Plot the final contour (ϕ = 0 in phi)
    plt.contour(phi, levels=[0], colors='blue', linewidths=2)

    # Add labels and title
    plt.title("ACM Contour Evolution: Red-initial, Blue-final")
    plt.axis('off')

    save_path = os.path.join(acm_dir, "ACM Contour Evolution.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Show the plot
    #plt.show()

# Displaying a level set function
def displayLSF(image, lsf, acm_dir, suffix):
    plt.imshow(image, cmap='gray'),plt.xticks([]), plt.yticks([])
    plt.contour(lsf,[0],colors='r',linewidth=2)
    plt.draw()
    
    file_name =  "Contour" + suffix + ".png"
    save_path = os.path.join(acm_dir, file_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ---------- Image properties

# helper to check image has only 0s and 1s (integer or floating point)
def is_binary_mask(arr):
    unique_elements = np.unique(arr)
    #print('Unique elements during binary mask check: ', unique_elements)
    # Check if all unique elements are either 0 or 1 (integer or float)
    return np.all(np.isin(unique_elements, [0, 1, 0.0, 1.0]))

# helper function to check if image is a probability mask
def is_probability_mask(arr):
    if isinstance(arr, tf.Tensor):
        return arr.dtype.is_floating and tf.reduce_all((arr >= 0) & (arr <= 1))
    return np.issubdtype(arr.dtype, np.floating) and np.all((arr >= 0) & (arr <= 1))

# Cropping the image so that it is square
# This function is potentially not needed anymore
# If needed it could be a little buggy so double check.
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

# Brings up a view where you can hover around and see what pixel you  are on
# May be helpful to choose a good center and readius for initial contour
def AnalyzeCoordinates(image_path):
    img = Image.open(image_path)

    # Display the image
    plt.imshow(img)
    plt.title("Hover over the image to get coordinates")
    coords = plt.ginput(1)  # Click to get one coordinate point
    print(f"Coordinates: {coords}")

# Applying additive bias correction to grayscale image - attempt to improve intial contour robustness.
def apply_ABC(image, sigma=50):
    """
    Apply Additive Bias Correction (ABC) to remove intensity non-uniformity.

    Parameters:
    - image: Grayscale image (NumPy array)
    - sigma: Standard deviation for Gaussian filter (controls smoothing level)

    Returns:
    - corrected_image: Bias-corrected image
    - bias_field: Estimated bias field
    """

    # Convert to float for computation
    image = image.astype(np.float32)

    # Estimate the bias field using a Gaussian filter
    bias_field = gaussian_filter(image, sigma=sigma)

    # Avoid division by zero
    bias_field[bias_field == 0] = 1

    # Correct the image by dividing by the bias field
    corrected_image = image / bias_field

    # Normalize to [0, 255] for visualization
    corrected_image = (corrected_image - corrected_image.min()) / (corrected_image.max() - corrected_image.min()) * 255
    corrected_image = corrected_image.astype(np.uint8)

    # Returned image is also 0 and 255 but in uint8

    return corrected_image, bias_field

def DisplayABCResult(image, bias_field, corrected_image, save_dir=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title("Original Image")
    plt.subplot(1, 3, 2), plt.imshow(bias_field, cmap='jet'), plt.title("Estimated Bias Field")
    plt.subplot(1, 3, 3), plt.imshow(corrected_image, cmap='gray'), plt.title("Corrected Image")

    if save_dir:
        title = "ABC Result"
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
    plt.close()

def apply_blur_plus_clahe(image, sigma=1, clipLimit=2.0):
    """
    Applies Gaussian blurring followed by contrast limited adaptive histogram equalization (CLAHE) to a grayscale image.
    
    Parameters:
        image (np.ndarray): Input grayscale image.
    
    Returns:
        np.ndarray: Processed image.
    """

    # # Apply Gaussian blur
    # blurred = gaussian(image, sigma=sigma)
    
    # # Convert to uint8 (skimage's gaussian returns float image)
    # blurred = (blurred * 255).astype(np.uint8)
    
    # # Apply CLAHE
    # clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    # enhanced = clahe.apply(blurred)
    
    # return enhanced

    # Ensure image is float in range [0, 1]
    image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0, 1)

    # Apply Gaussian blur
    blurred = gaussian(image, sigma=sigma)

    # Convert to uint8 safely
    blurred_uint8 = img_as_ubyte(blurred)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred_uint8)

    # between 0 and 255 but integers
    
    return enhanced

def DisplayCLAHEResult(image, enhanced_image, save_dir=None):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title("Original Image")
    plt.subplot(1, 2, 2), plt.imshow(enhanced_image, cmap='gray'), plt.title("CLAHE Corrected Image")

    if save_dir:
        title = "CLAHE Result"
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
    plt.close()

def displayACMScores(iterations, dice_scores, iou_scores, acm_dir):
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, dice_scores, label='DICE Score', color='blue', marker='o')
    plt.plot(iterations, iou_scores, label='IoU Score', color='green', marker='x')

    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.title('DICE and IoU Scores vs ACM Iterations')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(acm_dir, 'acm_scores_plot.png'), dpi=300, bbox_inches='tight')

# ---------- Contour/Segmentation Initializations

# Helper function to get center of an image and the largest possible radius [The minimum between height and width]
def GetDefaultInitContourParams(shape):
    # Dimensions and parameters
    rows, cols = shape[:2]
    center_row, center_col = rows // 2, cols // 2
    radius = min(rows, cols) // 2  # Example radius
    return center_row, center_col, radius

# Helper function to create an initial segmentation in the form of a binary mask where initial detection is a circle
def create_circular_mask(shape, center, radius):
    """
    Create a rectangular binary mask with a circle of ones.

    Parameters:
    - shape (tuple): Shape of the rectangle (height, width).
    - center (tuple): Coordinates of the circle center (x, y).
    - radius (float): Radius of the circle.

    Returns:
    - mask (numpy.ndarray): Binary mask as a 2D NumPy array.
    """

    height, width = shape  # Extract the dimensions of the rectangle
    # Create a grid of coordinates
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y) # This is correct

    # Calculate the distance from the center for each point
    #distance_from_center = np.sqrt((xx - center[0])**2 + (yy - center[1])**2) #buggy version
    distance_from_center = np.sqrt((yy - center[0])**2 + (xx - center[1])**2)

    # Create the mask: 1 inside and on the circle, 0 outside
    mask = (distance_from_center <= radius).astype(int)

    return mask

# Helper function for defining intial contour given center coordinates and radius
# I used this in the snakes implementation
def CreateInitCircContour(c_x, c_y, radius):
    # Defining an initial circular contour around the object of interest
    s = np.linspace(0, 2 * np.pi, 400) # Defining angles between 0 and 2*pi
    r = c_x + radius * np.sin(s) # Defining x coordinates along the circle
    c = c_y + radius * np.cos(s) # Defining y coordinates along the circle
    init = np.array([r, c]).T # Each row represents a coordinate point (x,y)
    #print ("Shape of original contour: ", init.shape) # (400,2)
    return init


# ---------- Evaluation functions

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