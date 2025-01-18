import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add the parent directory to the system path
cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))
save_dir = os.path.abspath(os.path.join(cwd, '../Results'))

# Imports
import level_set_acm as lsa
import lsa_helpers as lsah


# --- PART 1: Define the initial image and ground truth

image_path = os.path.abspath(os.path.join(cwd, '../../dataset/chase_db1/Image_01L.jpg'))
img = Image.open(image_path)
square_img = lsah.crop_to_square(img)

gt_path = os.path.abspath(os.path.join(cwd, '../../dataset/chase_db1/Image_01L_1stHO.png'))
gt_img = Image.open(gt_path)
gt_img_sq = lsah.crop_to_square(gt_img)

acm_dir = os.path.abspath(os.path.join(save_dir, 'Image_01L'))

# to use from now: square_img and gt_img_sq
final_img = square_img
final_gt = gt_img_sq
# confirming the properties of these images
print('Type of image and ground truth', type(final_img), type(final_gt)) # PIL.Image.Image
print('Shape of the image: ', final_img.size) # (960, 960)
print('Shape of the ground truth: ', final_gt.size) # (960, 960)
print('min and max intensity values of image: ', np.min(final_img), np.max(final_img)) # 0, 255
print('binary mask check for ground truth: ', lsah.is_binary_mask(final_gt)) # True
lsah.displayImage(final_img, "Image", save_dir=acm_dir)
lsah.displayImage(final_gt, "Ground Truth", save_dir=acm_dir)

# --- PART 2: Define the initial probability mask
initial_probability_mask = lsah.create_circular_mask(final_img.size[0])
print('Type of initial probability mask: ', type(initial_probability_mask)) #numpy.ndarray
print('Shape of the initial probability mask: ', initial_probability_mask.shape)
print('binary mask check for initial probability mask: ', lsah.is_binary_mask(initial_probability_mask))
lsah.displayImage(initial_probability_mask, 'Initial Probability Mask', True, acm_dir)

# --- PART 3: Deriving initial phi and lambda maps from initial probability mask

# img
f_img = final_img.convert('L')
img_intesity = np.array(f_img)
print('Type of intensity image', type(img_intesity)) # numpy.ndarray
print('Shape of the intensity image: ', img_intesity.shape) # (960, 960)
print('min and max intensity values of image: ', np.min(img_intesity), np.max(img_intesity)) # 0 236
lsah.displayImage(img_intesity, 'Intensity Image', True, acm_dir)

# lambdas
map_lambda1, map_lambda2 = lsa.get_lambda_maps(initial_probability_mask)
print('Shape of map lambda 1: ', map_lambda1.shape) # (960, 960)
print('Shape of map lambda 2: ', map_lambda2.shape) # (960, 960)

# initial phi
initial_phi = lsa.get_initial_phi(initial_probability_mask)
print('Shape of initial phi: ', initial_phi.shape) # (960, 960)

# --- Initialize parameter elems to active_contour_layer function
elems = (img_intesity, initial_phi, map_lambda1, map_lambda2)
input_image_size = final_img.size[0]
print('Input image size: ', input_image_size) # 960
iter_lim = 20

# --- Call active_contour_layer function and get the final phi output
final_seg_mask = lsa.active_contour_layer(elems=elems, input_image_size=input_image_size, nu=0.2, mu=0.1, iter_limit = iter_lim, acm_dir=acm_dir, freq=10) 
print('Type of final segmentation mask: ', type(final_seg_mask)) # <class 'tensorflow.python.framework.ops.EagerTensor'>
print('Shape of final segmentation mask: ', final_seg_mask.shape) # (960, 960)
print('binary mask check for final segmentation mask: ', lsah.is_binary_mask(final_seg_mask)) # True
lsah.displayImage(final_seg_mask, 'Final Segmentation Mask', True, acm_dir)


# --- Compute DICE loss between the phi output and ground truth