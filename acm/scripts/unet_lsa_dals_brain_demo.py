import numpy as np
import os
import sys

cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))

import level_set_acm as lsa
import lsa_helpers as lsah

make_rectangle = True

# --- Loading input image, ground truth, and initial segmentation
image_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_input.npy'))
gt_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_label.npy'))
init_seg_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_initseg.npy'))

image = np.load(image_path)
gt = lsah.normalize_mask(np.load(gt_path))
init_seg = np.load(init_seg_path)
init_seg = init_seg[0, :, :, 0]

acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Demo_brain/DALS/unet_initial_contour'))

if(make_rectangle):
    acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Demo_brain/DALS/unet_initial_contour_non_square'))
    # adding two columns at the end with all zero values to make non-square
    zero_columns = np.zeros((image.shape[0], 2))
    image = np.hstack((image, zero_columns))
    gt = np.hstack((gt, zero_columns))
    init_seg = np.hstack((init_seg, zero_columns))


# --- Confirming properties

# image - 2d array, grayscale, 0 to 255
print('Shape of the image: ', image.shape) # (256, 256)
print('min and max intensity values of image: ', np.min(image), np.max(image)) # 0.0, 255.0
lsah.displayImage(image, "Image", True, save_dir=acm_dir)

# ground truth - 2d array, binary mask
print('Shape of the ground truth: ', gt.shape) # (256, 256)
lsah.displayImage(gt, "Ground Truth", True, save_dir=acm_dir)
print('Unique values in ground truth: ', np.unique(gt)) # [0. 1.]

# initial segmentation - 2d array
print('Shape of the initial segmentation: ', init_seg.shape) # (256, 256)
lsah.displayImage(init_seg, "Initial Segmentation", True, save_dir=acm_dir)
print('unique values in initial segmentation: ', np.unique(init_seg)) # probabilities between 0 and 1

# --- Priniting initial dice score
init_mask = np.round(init_seg)
dice_score = lsah.dice_score(init_mask, gt)
print('Initial Dice score: ', dice_score)

# --- From initial segmentation get lambdas and initial phi

# lambdas
map_lambda1, map_lambda2 = lsa.get_lambda_maps(init_seg)
print('Shape of map lambda 1: ', map_lambda1.shape) # (256, 256)
print('Type of map lambda 1: ', type(map_lambda1), map_lambda1.dtype) # <class 'tensorflow.python.framework.ops.EagerTensor'> <dtype: 'float32'>
print('Min max values of map lambda 1: ', np.min(map_lambda1), np.max(map_lambda1)) # 1.6487212 7.3128667
print('Shape of map lambda 2: ', map_lambda2.shape) # (256, 256)
print('Type of map lambda 2: ', type(map_lambda2), map_lambda2.dtype) # <class 'tensorflow.python.framework.ops.EagerTensor'> <dtype: 'float32'>
print('Min max values of map lambda 2: ', np.min(map_lambda2), np.max(map_lambda2)) # 1.6530212 7.389056

# initial phi
initial_phi = lsa.get_initial_phi(init_seg)
print('Shape of initial phi: ', initial_phi.shape) # (256, 256)
print('Type of initial phi: ', type(initial_phi), initial_phi.dtype) # <class 'numpy.ndarray'> float32
print('Min max values of initial phi: ', np.min(initial_phi), np.max(initial_phi)) # -28.42534 158.90248

# --- Initialize parameter elems to active_contour_layer function
elems = (image, initial_phi, map_lambda1, map_lambda2)
input_image_size_x = image.shape[1]
input_image_size_y = image.shape[0]
print('Input image size x: ', input_image_size_x)
print('Input image size y: ', input_image_size_y)
iter_lim = 600

# --- Call active_contour_layer function and get the final phi output
final_seg = lsa.active_contour_layer(elems=elems, input_image_size=input_image_size_x, input_image_size_2=input_image_size_y, iter_limit = iter_lim, acm_dir=acm_dir, freq=50, gt=gt) 
print('Type of final segmentation mask: ', type(final_seg)) # <class 'tensorflow.python.framework.ops.EagerTensor'>
print('Shape of final segmentation mask: ', final_seg.shape) # (256, 256)
print('unique values in final segmentation: ', np.unique(final_seg)) # [0. 1.]
lsah.displayImage(final_seg, 'Final Segmentation Mask', True, acm_dir)
dice_score = lsah.dice_score(final_seg, gt)
print('Final Dice score: ', dice_score)