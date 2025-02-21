import os
import sys

from models.LevelSetACM import *

def acm_layer(intensity_image, initial_segmentation, num_iter, nu, mu):

    # Note: image, ground truth, and initial segmentation all have the same shape.

    # --- From initial segmentation get lambdas and initial phi (they would also have the same shape)

    # lambdas
    map_lambda1, map_lambda2 = get_lambda_maps(initial_segmentation)

    # initial phi
    initial_phi = get_initial_phi(initial_segmentation)
    # phi (signed distance map) is zero on the contour and signed inside and outside.

    # --- Initialize parameter elems to active_contour_layer function
    elems = (intensity_image, initial_phi, map_lambda1, map_lambda2)
    input_image_size_x = intensity_image.shape[1]
    input_image_size_y = intensity_image.shape[0]

    print('Shapes of inputs to active contour layer: ', intensity_image.shape, initial_phi.shape, map_lambda1.shape, map_lambda2.shape)

    # --- Call active_contour_layer function and get the final seg output
    final_binary_segmentation, final_phi, final_probability_mask = active_contour_layer(elems=elems, input_image_size=input_image_size_x, input_image_size_2=input_image_size_y, 
                                                        nu=nu, mu=mu, iter_limit = num_iter)
    
    # return final probability mask
    return final_probability_mask
    