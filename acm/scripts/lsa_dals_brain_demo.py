import numpy as np
import os
import sys

cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))

import level_set_acm as lsa
import lsa_helpers as lsah

# --- Loading input image, ground truth
image_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_input.npy'))
gt_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_label.npy'))

image = np.load(image_path)
gt = lsah.normalize_mask(np.load(gt_path))

#acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Demo_brain/trivial_initial_contour'))
acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Demo_brain/intermediate_initial_contour'))

# --- Confirming properties

# image - 2d array, grayscale, 0 to 255
print('Shape of the image: ', image.shape) # (256, 256)
print('min and max intensity values of image: ', np.min(image), np.max(image)) # 0.0, 255.0
lsah.displayImage(image, "Image", True, save_dir=acm_dir)

# image analysis
#lsah.AnalyzeCoordinates(os.path.abspath(os.path.join(acm_dir, 'Image.png')))

# ground truth - 2d array, binary mask
print('Shape of the ground truth: ', gt.shape) # (256, 256)
lsah.displayImage(gt, "Ground Truth", True, save_dir=acm_dir)
print('Unique values in ground truth: ', np.unique(gt)) # [0. 1.]

# initial segmentation
#initial_probability_mask = lsah.create_circular_mask(image.shape[0], factor=0.5)
initial_probability_mask = lsah.create_circle_mask(image.shape[0], (165, 85), 40)
print('Type of initial probability mask: ', type(initial_probability_mask)) #numpy.ndarray
print('Shape of the initial probability mask: ', initial_probability_mask.shape)
print('binary mask check for initial probability mask: ', lsah.is_binary_mask(initial_probability_mask))
lsah.displayImage(initial_probability_mask, 'Initial Probability Mask', True, acm_dir)

# --- Priniting initial dice score
init_mask = np.round(initial_probability_mask)
dice_score = lsah.dice_score(init_mask, gt)
print('Initial Dice score: ', dice_score)

# --- From initial segmentation get lambdas and initial phi

# lambdas
map_lambda1, map_lambda2 = lsa.get_lambda_maps(initial_probability_mask)
print('Shape of map lambda 1: ', map_lambda1.shape) # (256, 256)
print('Type of map lambda 1: ', type(map_lambda1), map_lambda1.dtype) # <class 'tensorflow.python.framework.ops.EagerTensor'> <dtype: 'float32'>
print('Min max values of map lambda 1: ', np.min(map_lambda1), np.max(map_lambda1)) # 1.6487212 7.3128667
print('Shape of map lambda 2: ', map_lambda2.shape) # (256, 256)
print('Type of map lambda 2: ', type(map_lambda2), map_lambda2.dtype) # <class 'tensorflow.python.framework.ops.EagerTensor'> <dtype: 'float32'>
print('Min max values of map lambda 2: ', np.min(map_lambda2), np.max(map_lambda2)) # 1.6530212 7.389056

# initial phi
initial_phi = lsa.get_initial_phi(initial_probability_mask)
print('Shape of initial phi: ', initial_phi.shape) # (256, 256)
print('Type of initial phi: ', type(initial_phi), initial_phi.dtype) # <class 'numpy.ndarray'> float32
print('Min max values of initial phi: ', np.min(initial_phi), np.max(initial_phi)) # -28.42534 158.90248

# --- Initialize parameter elems to active_contour_layer function
elems = (image, initial_phi, map_lambda1, map_lambda2)
input_image_size = image.shape[0]
print('Input image size: ', input_image_size) # 256
iter_lim = 1200

# --- Call active_contour_layer function and get the final phi output
final_seg = lsa.active_contour_layer(elems=elems, input_image_size=input_image_size, iter_limit = iter_lim, acm_dir=acm_dir, freq=100, gt=gt) 
print('Type of final segmentation mask: ', type(final_seg)) # <class 'tensorflow.python.framework.ops.EagerTensor'>
print('Shape of final segmentation mask: ', final_seg.shape) # (256, 256)
print('unique values in final segmentation: ', np.unique(final_seg)) # [0. 1.]
lsah.displayImage(final_seg, 'Final Segmentation Mask', True, acm_dir)
dice_score = lsah.dice_score(final_seg, gt)
print('Final Dice score: ', dice_score)