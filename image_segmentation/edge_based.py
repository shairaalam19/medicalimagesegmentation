import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import io, color, morphology


def robert_operator(input_path, image_name, output_path): 
    # Read Input Image
    input_image_path = f'{input_path}/{image_name}'
    input_image = imread(input_image_path)

    # Create the output directory if it doesn't exist
    output_dir = f'{output_path}/{image_name}'
    if not os.path.exists(output_dir):
        print(f"Creating directory {output_dir}")
        os.makedirs(output_dir)

    # Displaying Input Image
    input_image = img_as_ubyte(input_image)
    plt.figure(), plt.imshow(input_image), plt.title('(1) Input Image')
    plt.savefig(os.path.join(output_dir, '1_input_image.png'))

    # Convert the truecolor RGB image to grayscale
    input_image = rgb2gray(input_image)

    # Convert the image to double (float in Python)
    input_image = img_as_float(input_image)

    # Pre-allocate the filtered_image matrix with zeros
    filtered_image = np.zeros(input_image.shape)

    # Robert Operator Mask
    Mx = np.array([[1, 0], [0, -1]])
    My = np.array([[0, 1], [-1, 0]])

    # Edge Detection Process
    for i in range(input_image.shape[0] - 1):
        for j in range(input_image.shape[1] - 1):
            # Gradient approximations
            Gx = np.sum(Mx * input_image[i:i+2, j:j+2])
            Gy = np.sum(My * input_image[i:i+2, j:j+2])
            # Calculate magnitude of vector
            filtered_image[i, j] = np.sqrt(Gx**2 + Gy**2)

    # Displaying Filtered Image
    filtered_image = img_as_ubyte(filtered_image)
    plt.figure(), plt.imshow(filtered_image, cmap='gray'), plt.title('(2) Filtered Image')
    plt.savefig(os.path.join(output_dir, '2_filtered_image.png'))

    # Define a threshold value
    threshold_value = 100  # varies between [0, 255]
    output_image = np.maximum(filtered_image, threshold_value)
    output_image[output_image == round(threshold_value)] = 0

    # Convert to binary image
    threshold = threshold_otsu(output_image)
    output_image = output_image > threshold

    # Displaying Output Image (Edge Detected Image)
    plt.figure(), plt.imshow(output_image, cmap='gray'), plt.title('(3) Edge Detected Image')
    plt.savefig(os.path.join(output_dir, '3_edge_detected_image.png'))

def sobel_operator(input_path, image_name, output_path):
    # Read Input Image
    input_image_path = f'{input_path}/{image_name}'
    input_image = imread(input_image_path)

    # Create the output directory if it doesn't exist
    output_dir = f'{output_path}/{image_name}'
    if not os.path.exists(output_dir):
        print(f"Creating directory {output_dir}")
        os.makedirs(output_dir)

    # Displaying Input Image
    # Making image 2D
    if len(input_image.shape) == 3:  # If the image has multiple channels (RGB), convert to grayscale
        input_image = color.rgb2gray(input_image)
    input_image = img_as_ubyte(input_image)
    plt.figure(), plt.imshow(input_image), plt.title('(1) Input Image')
    plt.savefig(os.path.join(output_dir, '1_input_image.png'))
    
    # Step 1: Canny Edge Detection
    edges = canny(input_image / 255.0)
    plt.figure(figsize=(4, 3))
    plt.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.title('(2) Canny detector')
    canny_output_path = os.path.join(output_dir, '2_canny_edges.png')
    plt.savefig(canny_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Step 2: Fill Holes
    filled_image = ndi.binary_fill_holes(edges)
    plt.figure(figsize=(4, 3))
    plt.imshow(filled_image, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.title('(3) Filling the holes')
    filled_output_path = os.path.join(output_dir, '3_filled_image.png')
    plt.savefig(filled_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Step 3: Remove Small Objects
    cleaned_image = morphology.remove_small_objects(filled_image, 21)
    plt.figure(figsize=(4, 3))
    plt.imshow(cleaned_image, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.title('(4) Removing small objects')
    cleaned_output_path = os.path.join(output_dir, '4_cleaned_image.png')
    plt.savefig(cleaned_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def apply_operators(input_path, output_path): 
    # Create output folders for Robert and Sobel if they don't exist
    robert_output_path = os.path.join(output_path, 'robert')
    sobel_output_path = os.path.join(output_path, 'sobel')
    os.makedirs(robert_output_path, exist_ok=True)
    os.makedirs(sobel_output_path, exist_ok=True)

    # Iterate over all files in the input path
    for input_image_name in os.listdir(input_path):
        input_image_path = os.path.join(input_path, input_image_name)
        if os.path.isfile(input_image_path):
            # Apply Robert operator
            robert_operator(input_path, input_image_name, f"{output_path}/robert")
            # Apply Sobel operator
            sobel_operator(input_path, input_image_name, f"{output_path}/sobel")

input_path = "../dataset/animals"
output_path = '../output/edge_based' 
apply_operators(input_path, output_path)