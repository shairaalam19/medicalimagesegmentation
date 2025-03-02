# original snakes model (active contour model)

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.io import imread
import sys

# Helper function for defining intial contour given center coordinates and radius
def CreateInitCircContour(c_x, c_y, radius):
    # Defining an initial circular contour around the object of interest
    s = np.linspace(0, 2 * np.pi, 400) # Defining angles between 0 and 2*pi
    r = c_x + radius * np.sin(s) # Defining x coordinates along the circle
    c = c_y + radius * np.cos(s) # Defining y coordinates along the circle
    init = np.array([r, c]).T # Each row represents a coordinate point (x,y)
    #print ("Shape of original contour: ", init.shape) # (400,2)
    return init

def DisplayACMResult(img, init, snake):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray) # grayscale image (not gaussian blurred image)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3) # red dashed line for initial contour
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3) # blue line for fitted contour
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    plt.show()

def GetDefaultInitContourParams(img):
    # Dimensions and parameters
    rows, cols = img.shape[:2]
    center_row, center_col = rows // 2, cols // 2
    radius = min(rows, cols) // 2  # Example radius
    return center_row, center_col, radius

# -----------------------Image 1: astronaut

# img = data.astronaut()
# img = rgb2gray(img) # converting to grayscale makes it easier for contour detection

# init = CreateInitCircContour(100,220,100)

# # Variable snake will store the final contour
# snake = active_contour(
#     gaussian(img, sigma=3, preserve_range=False), # smoothing image with a gaussian filter - applying gaussian blurring - greater sigma, greater blur
#     init, # starting contour
#     alpha=0.015, # controls tension
#     beta=10, # controls smoothness
#     gamma=0.001, # controls step-size
#     # The default parameters w_line=0, w_edge=1 will make the curve search towards edges, such as the boundaries of the face.
# )
# #print ("Shape of final contour: ", snake.shape) # (400, 2) - points of the original contour are iteratively updates (hence same shape)

# DisplayACMResult(img, init, snake)

# ------------------Image 2: Cat

# img = imread("../../data/animals/cat.jpg")
# img = rgb2gray(img)  # Convert to grayscale

# center_row, center_col, radius = GetDefaultInitContourParams(img)
# init = CreateInitCircContour(center_row, center_col, radius)

# snake = active_contour(
#     gaussian(img, sigma=3, preserve_range=False), # smoothing image with a gaussian filter - applying gaussian blurring - greater sigma, greater blur
#     init, # starting contour
#     alpha=0.015, # controls tension
#     beta=10, # controls smoothness
#     gamma=0.001, # controls step-size
#     # The default parameters w_line=0, w_edge=1 will make the curve search towards edges.
# )

# DisplayACMResult(img, init, snake)

# ------------------Image 3: Dog

img = imread("../../data/animals/dog.jpeg")
img = rgb2gray(img)  # Convert to grayscale

center_row, center_col, radius = GetDefaultInitContourParams(img)
init = CreateInitCircContour(center_row, center_col, radius)

snake = active_contour(
    gaussian(img, sigma=3, preserve_range=False), # smoothing image with a gaussian filter - applying gaussian blurring - greater sigma, greater blur
    init, # starting contour
    alpha=0.001, # controls tension
    beta=1, # controls smoothness
    gamma=0.001, # controls step-size
    #max_num_iter=5000,
    # The default parameters w_line=0, w_edge=1 will make the curve search towards edges.
)

DisplayACMResult(img, init, snake)