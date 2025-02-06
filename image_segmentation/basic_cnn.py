import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from scipy.ndimage import convolve
import cv2

def conv2d(image, kernel):
    """Apply a 2D convolution to an image using the given kernel."""
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output_height, output_width = image_height - kernel_height + 1, image_width - kernel_width + 1
    output = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i + kernel_height, j:j + kernel_width] * kernel)
    return output

def relu(feature_map):
    return np.maximum(0, feature_map)

def max_pooling(image, pool_size=2, stride=2):
    """Apply max pooling to downsample the image."""
    output_height = (image.shape[0] - pool_size) // stride + 1
    output_width = (image.shape[1] - pool_size) // stride + 1
    output = np.zeros((output_height, output_width))
    for i in range(0, output_height * stride, stride):
        for j in range(0, output_width * stride, stride):
            output[i // stride, j // stride] = np.max(image[i:i + pool_size, j:j + pool_size])
    return output

def upsample(image, scale=2):
    """Upsample the image by a given scale factor using nearest-neighbor interpolation."""
    output_height = image.shape[0] * scale
    output_width = image.shape[1] * scale
    output = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = image[i // scale, j // scale]
    return output

# Roberts Operator
def roberts_operator(input_image):
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])

    edge_x = conv2d(input_image, roberts_x)
    edge_y = conv2d(input_image, roberts_y)
    
    return np.sqrt(edge_x**2 + edge_y**2)

# Prewitt Operator
def prewitt_operator(input_image):
    prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    edge_x = conv2d(input_image, prewitt_x)
    edge_y = conv2d(input_image, prewitt_y)
    
    return np.sqrt(edge_x**2 + edge_y**2)

# Resize function to make arrays the same size
def resize_image_to_match(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

# Define a more complex CNN model with multiple layers
def apply_complex_cnn(input_image):
    # Layer 1: First Convolution + ReLU
    kernel1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) # Sobel operator to detect vertical edges 
    conv_output1 = conv2d(input_image, kernel1)
    relu_output1 = relu(conv_output1)

    # Layer 2: Second Convolution + ReLU
    kernel2 = np.array([[0, 1, 0], [0, 1, 0], [-1, -1, 1]])
    conv_output2 = conv2d(relu_output1, kernel2)
    relu_output2 = relu(conv_output2)

    # Apply Roberts Operator
    roberts_edges = roberts_operator(input_image)

    # Apply Prewitt Operator
    prewitt_edges = prewitt_operator(input_image)

    # Resize edge detection outputs to match the CNN feature map size
    roberts_edges_resized = resize_image_to_match(roberts_edges, relu_output2.shape)
    prewitt_edges_resized = resize_image_to_match(prewitt_edges, relu_output2.shape)

    # Combine the outputs of convolution layers and edge detectors
    combined_output = relu_output2 + roberts_edges_resized + prewitt_edges_resized

    # Layer 3: Max Pooling
    pool_output = max_pooling(combined_output, pool_size=2, stride=2)

    # Layer 4: Third Convolution + ReLU (after pooling)
    kernel3 = np.array([[1, -1, 1], [0, 1, 0], [-1, 1, -1]])
    conv_output3 = conv2d(pool_output, kernel3)
    relu_output3 = relu(conv_output3)

    # Layer 5: Upsampling
    upsampled_output = upsample(relu_output3, scale=2)

    # Layer 6: Final Convolution + ReLU (to refine after upsampling)
    kernel4 = np.array([[-1, 0, 1], [1, -1, 1], [0, 1, -1]])
    conv_output4 = conv2d(upsampled_output, kernel4)
    relu_output4 = relu(conv_output4)

    return relu_output4

# Single Convolutional layer CNN
def apply_basic_cnn(input_image): 
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # Sobel Operator

    # Step 1: Convolution
    conv_output = conv2d(input_image, kernel)

    # Step 2: ReLU Activation
    relu_output = relu(conv_output)

    # Step 3: Pooling - Reduces spatial dimensions, which is useful for downscaling the image while retaining important features
    pool_output = max_pooling(relu_output, pool_size=2, stride=2)

    # Step 4: Upsampling - Increases spatial dimensions back to the original size after pooling, which is useful for tasks like image segmentation where the output needs to be the same size as the input.
    upsample_output = upsample(pool_output, scale=2)

    # Step 5: Second Convolution and ReLU 
    kernel2 = np.array([[0, 1, 0], [0, 1, 0], [-1, -1, 1]])  # Another example kernel
    conv_output2 = conv2d(upsample_output, kernel2)
    relu_output2 = relu(conv_output2)

    return relu_output2

# Apply CNN to each image and store output
def cnn_operator(input_path, image_name, output_path):
    # Read Input Image
    input_image_path = f'{input_path}/{image_name}'
    input_image = imread(input_image_path)

    # Create the output directory if it doesn't exist
    output_dir = f'{output_path}/{image_name}'
    if not os.path.exists(output_dir):
        print(f"Creating directory {output_dir}")
        os.makedirs(output_dir)

    # Convert image to grayscale if it's RGB
    if len(input_image.shape) == 3:
        input_image = rgb2gray(input_image)
    input_image = img_as_ubyte(input_image)

    # Apply the CNN model
    cnn_basic_output = apply_basic_cnn(input_image)
    cnn_complex_output = apply_complex_cnn(input_image)

    # Save the original and CNN output images
    plt.figure(), plt.imshow(input_image, cmap='gray'), plt.title('Input Image')
    plt.savefig(os.path.join(output_dir, '1_input_image.png'))

    plt.figure(), plt.imshow(cnn_basic_output, cmap='gray'), plt.title('CNN Output')
    plt.savefig(save_image_with_unique_name(output_dir, '2_cnn_basic_output'))

    plt.figure(), plt.imshow(cnn_complex_output, cmap='gray'), plt.title('CNN Output')
    plt.savefig(save_image_with_unique_name(output_dir, '3_cnn_complex_output'))

    plt.close('all')

def save_image_with_unique_name(output_dir, base_name):
        i = 1
        file_path = os.path.join(output_dir, f'{base_name}.png')
        while os.path.exists(file_path):
            file_path = os.path.join(output_dir, f'{base_name}_{i}.png')
            i += 1
        return file_path

# Pass the dataset path and where to store the output 
def multiple(input_path, output_path, subfolder):
    cnn_output_path = os.path.join(output_path, subfolder)
    os.makedirs(cnn_output_path, exist_ok=True)

    # Iterate over all files in the input path
    for input_image_name in os.listdir(input_path):
        input_image_path = os.path.join(input_path, input_image_name)
        if os.path.isfile(input_image_path):
            # Apply CNN operator
            cnn_operator(input_path, input_image_name, cnn_output_path)

input_path = "dataset/medical_images"
output_path = "output/cnn"
subfolder = "U-Net"
multiple(input_path, output_path, subfolder)
