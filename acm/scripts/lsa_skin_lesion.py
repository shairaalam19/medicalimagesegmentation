import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# Update system path
cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))

import level_set_acm as lsa
import lsa_helpers as lsah

# Assumption: ISIC dataset is right outside the medicalimagesegmentation folder 

image_path = os.path.abspath(os.path.join(cwd, '../../../ISIC/Train_data/ISIC_0000000.jpg'))
gt_path = os.path.abspath(os.path.join(cwd, '../../../ISIC/Train_gt/ISIC_0000000_Segmentation.png'))
acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Skin_Lesion'))

img = Image.open(image_path)
sq_img = img.resize((min(img.size), min(img.size)))
gt = Image.open(gt_path)
sq_gt = gt.resize((min(gt.size), min(gt.size)))

f_img = sq_img.convert('L') # converting to grayscale
img_intensity = np.array(f_img) # converting to array

f_gt = lsah.normalize_mask(np.array(sq_gt))

print('Shape of image: ', sq_img.size)
lsah.displayImage(sq_img, "Image", save_dir=acm_dir)
print('Shape of ground truth: ', f_gt.shape)
print('binary mask check for ground truth: ', lsah.is_binary_mask(f_gt)) # True
lsah.displayImage(f_gt, "Ground Truth", True, save_dir=acm_dir)
lsah.displayImage(img_intensity, 'Intensity Image', True, acm_dir)

# probability mask
initial_probability_mask = lsah.create_circular_mask(f_img.size[0], factor=0.5)
print('binary mask check for initial probability mask: ', lsah.is_binary_mask(initial_probability_mask))
lsah.displayImage(initial_probability_mask, 'Initial Probability Mask', True, acm_dir)

dice_score = lsah.dice_score(initial_probability_mask, f_gt)
print('Initial Dice score: ', dice_score)

map_lambda1, map_lambda2 = lsa.get_lambda_maps(initial_probability_mask)
initial_phi = lsa.get_initial_phi(initial_probability_mask)
elems = (img_intensity, initial_phi, map_lambda1, map_lambda2)
input_image_size = img_intensity.shape[0]
iter_lim = 100

final_seg_mask = lsa.active_contour_layer(elems=elems, input_image_size=input_image_size, iter_limit = iter_lim, acm_dir=acm_dir, nu=100, mu=1, freq=20, gt=f_gt)
print('binary mask check for final segmentation mask: ', lsah.is_binary_mask(final_seg_mask))
lsah.displayImage(final_seg_mask, 'Final Segmentation Mask', True, acm_dir)

dice_score = lsah.dice_score(final_seg_mask, f_gt)
print('Final Dice score: ', dice_score)