import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
#from pylab import*

# This code implements the Chan-Vese active contour model for image segmentation. 
# It evolves the initial level set contour to segment an object based on region-based intensity differences.

# Reading and Preprocessing the Image
Image = cv2.imread('dataset/medical/cccc.jpg',1)  # color mode 1: BGR mode
#Image = cv2.imread('dataset/animals/cat.jpg',1)
#Image = cv2.imread('dataset/animals/dog.jpeg',1)
image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY) # converts image from BGR color space to grayscale
img=np.array(image,dtype=np.float64) # Converts the grayscale image into a NumPy array with float64 data type for precise calculations

# Initializing the level set function

# Create an initial level set function (LSF) as a matrix filled with 1.
IniLSF = np.ones((img.shape[0],img.shape[1]),img.dtype)
# Set a square region of the LSF (rows 30 to 80 and columns 30 to 80) to -1.
# TODO: make this step less hard-coded
IniLSF[30:80,30:80]= -1 
# Invert the LSF values (1 becomes -1 and vice versa).
IniLSF=-IniLSF

# Preparing the Image for Display

# Convert the BGR image to RGB format for proper display in matplotlib.
Image = cv2.cvtColor(Image,cv2.COLOR_BGR2RGB)
# Display the RGB image without axis ticks
plt.figure(1),plt.imshow(Image),plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
# Overlay the contour of the initial LSF (zero level) on the image in blue with a line width of 2.
plt.contour(IniLSF,[0],color = 'b',linewidth=2)
# Render and display the plot without blocking further execution.
#plt.draw(),plt.show(block=False)
plt.draw(),plt.show()

# A function to apply a mathematical operation element-wise on a matrix (input) based on the string (str).
# Computes either the arctangent or square root of each pixel value.
def mat_math (input,str):
    output=input
    # TODO: in the loops, change img to input for more clarity
    #for i in range(img.shape[0]):
    #    for j in range(img.shape[1]):
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if str=="atan":
                output[i,j] = math.atan(input[i,j]) 
            if str=="sqrt":
                output[i,j] = math.sqrt(input[i,j])
    return output 

# Chan-Vese Algorithm Implementation
def CV (LSF, img, mu, nu, epison,step):

    # Background
    # The zero-level contour of the Level Set Function (LSF) refers to the set of points where the LSF equals zero.
    # the LSF is a scalar field (a function defined over the image domain), and the zero-level contour represents a curve or boundary embedded within this field.

    # Computes the Dirac delta function approximation.
    # This is for localizing computations along the zero-level contour of the level set function (LSF).
    # Ensures that updates in the evolution equation only affect the zero-level set.
    Drc = (epison / math.pi) / (epison*epison+ LSF*LSF)

    # Computes the Heaviside function approximation.
    # used to distinguish between points inside and outside the curve
    # basically segmenting image into two regions
    # inside contour expected to have values less than 0
    # outsdie contour expected to have values greater than 0
    Hea = 0.5*(1 + (2 / math.pi)*mat_math(LSF/epison,"atan"))
    # Now each element transitions smoothly between 0 and 1.

    # computes the gradient of LSF in the x and y directions
    Iy, Ix = np.gradient(LSF)

    #Computes the gradient magnitude and normalizes it to get the normals Nx and Ny
    s = mat_math(Ix*Ix+Iy*Iy,"sqrt") 
    Nx = Ix / (s+0.000001) 
    Ny = Iy / (s+0.000001)
    # Nx, Ny represent the unit normal vector to the contour.
    # Define the direction of movement for the zero-level contour during its evolution

    # Computes the second derivatives of the LSF.
    # Calculates the curvature (cur) of the of the zero-level contour
    # The curvature term smoothens the contour, preventing it from forming sharp edges.
    Mxx,Nxx = np.gradient(Nx)
    Nyy,Myy = np.gradient(Ny)
    cur = Nxx + Nyy

    # Calculates the length term
    # length regularization term proportional to the curvature
    Length = nu*Drc*cur

    # Computes the Laplacian and the penalty term.
    Lap = cv2.Laplacian(LSF,-1) 
    Penalty = mu*(Lap - cur)
    # A penalty term to keep the LSF smooth and close to a signed distance function.
    # The penalty term regularizes the shape of the LSF
    # This enforces uniformity in the evolution and prevents irregular deformations.

    # Calculates the means inside (C1) and outside (C2) the contour.
    s1=Hea*img 
    s2=(1-Hea)*img 
    s3=1-Hea 
    C1 = s1.sum()/ Hea.sum() 
    C2 = s2.sum()/ s3.sum() 

    # Computing the Chan-Vese energy term
    CVterm = Drc*(-1 * (img - C1)*(img - C1) + 1 * (img - C2)*(img - C2))
    # Updates the LSF based on how well the regions fit the respective intensity models

    # Updating the LSF
    LSF = LSF + step*(Length + Penalty + CVterm)
    return LSF


# Running the Algorithm

# Sets parameters for the Chan-Vese algorithm and initializes the LSF
mu = 1 # weight of the penalty term
nu = 0.003 * 255 * 255 # weight of the length term in CV energy function
epison = 1 
step = 0.1 # wieght of update to LSF
LSF=IniLSF

# Iteratively updates the LSF using the Chan-Vese algorithm.
num = 20 # TODO: update number of energy minimization iterations as needed
for i in range(1,num):
    LSF = CV(LSF, img, mu, nu, epison,step) 
    if i % 1 == 0: # TODO: update frequency of display as needed
        # overlays the updated contour of LSF (red) on the original image
        plt.imshow(Image),plt.xticks([]), plt.yticks([])  
        plt.contour(LSF,[0],colors='r',linewidth=2) 
        plt.draw(), plt.show()
        #plt.show(block=False)
        #plt.pause(0.01)
