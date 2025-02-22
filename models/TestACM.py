# DALS DEMO testing either torch/tf Level set acm

import numpy as np
import os
import sys
import math
import matplotlib.pyplot as plt
import torch

cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../acm/Level_Set_ACM')))

import lsa_helpers as lsah
import LevelSetACM_tf as lsa
use_torch = False
# import LevelSetACM_torch as lsa
# use_torch = True

# ----- Original Brain Demo
print('Testing Model ACM for Original Brain Demo')
print()

image_path = os.path.abspath(os.path.join(cwd, '../acm/dals_demo_brain/img1_input.npy'))
gt_path = os.path.abspath(os.path.join(cwd, '../acm/dals_demo_brain/img1_label.npy'))
init_seg_path = os.path.abspath(os.path.join(cwd, '../acm/dals_demo_brain/img1_initseg.npy'))

img = np.load(image_path)
gt = lsah.normalize_mask(np.load(gt_path))

# normalize image between 0 and 1
#img = img/255.0

print('Shape of the image: ', img.shape)
print('min and max intensity values of image: ', np.min(img), np.max(img))

print('Shape of the ground truth: ', gt.shape)
print('binary mask check for ground truth: ', lsah.is_binary_mask(gt))

init_s = np.load(init_seg_path)
init_s = init_s[0, :, :, 0]

print('Shape of the initial segmentation: ', init_s.shape)
is_prob_mask = lsah.is_binary_mask(init_s) or lsah.is_probability_mask(init_s)
print('Probability mask check for initial segmentation: ', is_prob_mask) # probabilities between 0 and 1

# plt.imshow(img, cmap='gray')
# plt.show()

# plt.imshow(gt, cmap='gray')
# plt.show()

# plt.imshow(init_s, cmap='gray')
# plt.show()

if use_torch:
    # convert image, gt, and init_s into torch tensors
    img = torch.tensor(img)
    gt = torch.tensor(gt)
    init_s = torch.tensor(init_s)
    print(torch.min(img), torch.max(img))
    print(torch.min(gt), torch.max(gt))
    print(torch.min(init_s), torch.max(init_s))

# lambdas
map_lambda1, map_lambda2 = lsa.get_lambda_maps(init_s)
# print('Shape of map lambda 1: ', map_lambda1.shape)
# print('Type of map lambda 1: ', type(map_lambda1), map_lambda1.dtype) # <class 'tensorflow.python.framework.ops.EagerTensor'> <dtype: 'float32'>
# print('Min max values of map lambda 1: ', np.min(map_lambda1), np.max(map_lambda1))
# print('Shape of map lambda 2: ', map_lambda2.shape)
# print('Type of map lambda 2: ', type(map_lambda2), map_lambda2.dtype) # <class 'tensorflow.python.framework.ops.EagerTensor'> <dtype: 'float32'>
# print('Min max values of map lambda 2: ', np.min(map_lambda2), np.max(map_lambda2))

# initial phi
initial_phi = lsa.get_initial_phi(init_s)
# print('Shape of initial phi: ', initial_phi.shape)
# print('Type of initial phi: ', type(initial_phi), initial_phi.dtype) # <class 'numpy.ndarray'> float32
# print('Min max values of initial phi: ', np.min(initial_phi), np.max(initial_phi))
# # phi (signed distance map) is zero on the contour and signed inside and outside.

# --- Initialize parameter elems to active_contour_layer function
elems = (img, initial_phi, map_lambda1, map_lambda2)
input_image_size_x = img.shape[1]
input_image_size_y = img.shape[0]
print('Input image size x: ', input_image_size_x)
print('Input image size y: ', input_image_size_y)

if use_torch:
    nu = torch.tensor(5.0)
    mu = torch.tensor(0.2)
    iter_limit = torch.tensor(600)
else:
    nu = 5.0
    mu = 0.2
    iter_limit = 600

# --- Call active_contour_layer function and get the final seg output
final_seg, final_phi, final_prob_mask = lsa.active_contour_layer(elems=elems, input_image_size=input_image_size_x, input_image_size_2=input_image_size_y, 
                                        nu=nu, mu=mu, iter_limit = iter_limit) 
print('Type of final segmentation mask: ', type(final_seg)) # <class 'tensorflow.python.framework.ops.EagerTensor'>
print('Shape of final segmentation mask: ', final_seg.shape)
print('binary mask check for final segmentation: ', lsah.is_binary_mask(final_seg))

# Note: initial dice score - 0.9039
dice_score = lsah.dice_score(final_seg, gt) # Tf - 0.9735, torch - 0.9629
iou_score = lsah.iou_score(final_seg, gt)
print('Final Dice score: ', dice_score)
print('Final IOU score: ', iou_score)

plt.imshow(final_seg, cmap='gray')
plt.show()