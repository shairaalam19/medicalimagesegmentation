import numpy as np
import cv2
import math
#from pylab import*

# code source: https://www.kaggle.com/code/naim99/active-contour-model-python

# This code implements the Chan-Vese active contour model for image segmentation. 
# It evolves the initial level set contour to segment an object based on region-based intensity differences.

# A function to apply a mathematical operation element-wise on a matrix (input) based on the string (str).
# Computes either the arctangent or square root of each pixel value.
# TODO: potentially optimize this
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
def CV (LSF, img, mu, nu, eps, step):

    # Background:
    # The zero-level contour of the Level Set Function (LSF) refers to the set of points where the LSF equals zero.
    # The LSF is a scalar field (a function defined over the image domain), and the zero-level contour represents a curve or boundary embedded within this field.
    # LSF has the same shape as the image
    # It's like the signed distance map (like phi in DALS), but more like a signed map.

    # Computes the Dirac delta function approximation.
    # This is for localizing computations along the zero-level contour of the level set function (LSF).
    # Ensures that updates in the evolution equation only affect the zero-level set.
    Drc = (eps / math.pi) / (eps*eps+ LSF*LSF)

    # Computes the Heaviside function approximation.
    # used to distinguish between points inside and outside the curve
    # basically segmenting image into two regions
    # inside contour expected to have values less than 0
    # outsdie contour expected to have values greater than 0
    Hea = 0.5*(1 + (2 / math.pi)*mat_math(LSF/eps,"atan"))
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