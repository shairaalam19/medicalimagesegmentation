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
from skimage.exposure import rescale_intensity

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

    # Step 1: Robert Operator Mask for horizontal and vertical gradients
    # Two 2x2 Robert operator masks (Mx and My) are defined to compute the horizontal and vertical gradients.
    Mx = np.array([[1, 0], [0, -1]]) # horizontal diagonal 
    My = np.array([[0, 1], [-1, 0]]) # vertical diagonal

    # Step 2: Apply filtering to image based on calculated gradients (diagonal gradient)
    # Edge Detection Process
    # This loop applies the Robert operator over each pixel (except the boundary). For each pixel, it extracts a 2x2 neighborhood and applies the Mx and My masks to calculate the gradients (Gx and Gy). The magnitude of the gradient is then stored in the filtered_image.
    for i in range(input_image.shape[0] - 1):
        for j in range(input_image.shape[1] - 1):
            # Gradient approximations
            Gx = np.sum(Mx * input_image[i:i+2, j:j+2]) # horizontal diagonal gradient 
            Gy = np.sum(My * input_image[i:i+2, j:j+2]) # vertical diagonal gradient 
            # Calculate magnitude of vector
            filtered_image[i, j] = np.sqrt(Gx**2 + Gy**2) # This gives the strength of the edge at that point, combining the diagonal gradients from both directions.

    # Displaying Filtered Image
    # The filtered image is rescaled to fit within the range [0, 1] and then converted back to 8-bit unsigned byte format for display. The filtered image (with edges detected) is shown and saved as 2_filtered_image.png.
    filtered_image = rescale_intensity(filtered_image, in_range=(-1, 1), out_range=(0, 1))
    filtered_image = img_as_ubyte(filtered_image)
    plt.figure(), plt.imshow(filtered_image, cmap='gray'), plt.title('(2) Filtered Image')
    plt.savefig(os.path.join(output_dir, '2_filtered_image.png'))

    # Step 3: Apply a threshold to the images for edges 
    # Define a threshold value
    # A threshold value (100 in this case) is applied to the filtered image, suppressing lower-intensity values (weak edges). Pixels with intensity equal to the threshold are set to 0.
    threshold_value = 100  # varies between [0, 255]
    output_image = np.maximum(filtered_image, threshold_value)
    output_image[output_image == round(threshold_value)] = 0

    # Convert to binary image
    # The Otsu thresholding method is used to convert the image into a binary form (edges = white, non-edges = black) for clearer edge detection.
    threshold = threshold_otsu(output_image)
    output_image = output_image > threshold

    # Displaying Output Image (Edge Detected Image)
    # Finally, the edge-detected image is displayed and saved as 3_edge_detected_image.png. This image shows the detected edges as a binary image.
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
    # Making image 2D and making it grayscale
    if len(input_image.shape) == 3:  # If the image has multiple channels, it is a colored image 
        input_image = color.rgb2gray(input_image)
    input_image = img_as_ubyte(input_image)
    # plt.figure(), plt.imshow(input_image), plt.title('(1) Input Image')
    # plt.savefig(os.path.join(output_dir, '1_input_image.png'))
    plt.figure(), plt.imshow(input_image, cmap='gray'), plt.title('(1) Input Image')
    plt.savefig(os.path.join(output_dir, '1_input_image.png'))
    
    # Step 1: Canny Edge Detection (3x3 convolution, vertical and horizontal gradient, smoothing)
    # includes multiple steps such as Gaussian smoothing, gradient calculation (using Sobel or a similar operator internally), and non-maximum suppression, followed by edge linking through hysteresis. While it computes gradients, it does so internally without explicitly defining the Sobel masks
    # Canny edge detection algorithm is applied to the grayscale image. It detects the edges in the image by finding areas with a sharp change in intensity.
    edges = canny(input_image / 255.0) # normalized by dividing by 255.0 (scaling pixel values to the range [0, 1])
    # plt.figure(figsize=(4, 3))
    # plt.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
    # plt.axis('off')
    # plt.title('(2) Canny detector')
    # canny_output_path = os.path.join(output_dir, '2_canny_edges.png')
    # plt.savefig(canny_output_path, bbox_inches='tight', pad_inches=0)
    # plt.close()

    plt.figure(), plt.imshow(edges, cmap='gray'), plt.title('(2) Canny detector')
    plt.savefig(os.path.join(output_dir, '2_canny_edges.png'))
    
    # Step 2: Fill Holes
    # This fills in gaps which might correspond to solid objects where internal parts are missed during edge detection
    filled_image = ndi.binary_fill_holes(edges) # Holes (areas within detected edges that are fully enclosed) in the binary edge image are filled 
    # plt.figure(figsize=(4, 3))
    # plt.imshow(filled_image, cmap=plt.cm.gray, interpolation='nearest')
    # plt.axis('off')
    # plt.title('(3) Filling the holes')
    # filled_output_path = os.path.join(output_dir, '3_filled_image.png')
    # plt.savefig(filled_output_path, bbox_inches='tight', pad_inches=0)
    # plt.close()

    plt.figure(), plt.imshow(filled_image, cmap='gray'), plt.title('(3) Filling the holes')
    plt.savefig(os.path.join(output_dir, '3_filled_image.png'))

    # Step 3: Remove Small Objects
    # Cleans up noise or irrelevant small regions that might have been detected as edges or filled areas 
    cleaned_image = morphology.remove_small_objects(filled_image, 21) # Small objects (connected components smaller than a specified size, in this case, 21 pixels) are removed from the image using
    # plt.figure(figsize=(4, 3))
    # plt.imshow(cleaned_image, cmap=plt.cm.gray, interpolation='nearest')
    # plt.axis('off')
    # plt.title('(4) Removing small objects')
    # cleaned_output_path = os.path.join(output_dir, '4_cleaned_image.png')
    # plt.savefig(cleaned_output_path, bbox_inches='tight', pad_inches=0)
    # plt.close()

    plt.figure(), plt.imshow(cleaned_image, cmap='gray'), plt.title('(4) Removing small objects')
    plt.savefig(os.path.join(output_dir, '4_cleaned_image.png'))

def multiple(input_path, output_path): 
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

def single(input_path, input_image_name, output_path): 
    # Create output folders for Robert and Sobel if they don't exist
    robert_output_path = os.path.join(output_path, 'robert')
    sobel_output_path = os.path.join(output_path, 'sobel')
    os.makedirs(robert_output_path, exist_ok=True)
    os.makedirs(sobel_output_path, exist_ok=True)

    # Apply Robert operator
    robert_operator(input_path, input_image_name, f"{output_path}/robert")
    # Apply Sobel operator
    sobel_operator(input_path, input_image_name, f"{output_path}/sobel")

input_path = "dataset/medical_images"
input_image_name = "bjorke_1.png"
output_path = 'output/edge_based' 
multiple(input_path, output_path)
# single(input_path, input_image_name, output_path)