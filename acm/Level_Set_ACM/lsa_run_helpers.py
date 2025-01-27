import numpy as np
import os
import sys
from PIL import Image
from scipy.ndimage import gaussian_filter
import level_set_acm as lsa
import lsa_helpers as lsah

# Defines helper functions that take care of entire level set acm pipeline:
#   image/gt/init seg processing, constructing inputs to lsa, running lsa, logs, results, saving outputs

# Helper function that given an image, ground truth, image segmentation and acm properties, 
#   runs the acm and generates and displays all results.
def run_lsa(image, ground_truth, init_seg=None, acm_dir=None, iter_lim=300, save_freq=50, nu = 5.0, mu = 0.2):

    if (acm_dir):
        if (not os.path.exists(acm_dir)):
            print("The acm directory %s is not valid" % acm_dir)
            sys.exit()

    # Finalizing the image
    if isinstance(image, str):
        if (not os.path.exists(image)):
            print("The image path %s is not valid" % image)
            sys.exit()
        if(image.endswith('.npy')):
            img = np.load(image)
        else:
            # It's a path to an image file
            pil_img = Image.open(image)
            if (acm_dir):
                lsah.displayImage(pil_img, "Color Image", save_dir=acm_dir)
            img = np.array(pil_img.convert('L')) # converting to grayscale and an array.
            # potentially gaussian filtering can be applied here if needed.
    elif (isinstance(image, np.ndarray)):
        img = image
    else:
        print("Image type currently not supported")
        sys.exit()

    print('Shape of the image: ', img.shape)
    print('min and max intensity values of image: ', np.min(img), np.max(img))
    
    # Finalizing the ground truth
    if isinstance(ground_truth, str):
        if (not os.path.exists(ground_truth)):
            print("The ground truth path %s is not valid" % ground_truth)
            sys.exit()
        if(ground_truth.endswith('.npy')):
            gt = lsah.normalize_mask(np.load(ground_truth))
        else:
            # It's a path to a ground truth file
            pil_gt = Image.open(ground_truth)
            gt = lsah.normalize_mask(np.array(pil_gt))
    elif (isinstance(ground_truth, np.ndarray)):
        gt = ground_truth
    else:
        print("Ground truth type currently not supported")
        sys.exit()

    print('Shape of the ground truth: ', gt.shape)
    print('binary mask check for ground truth: ', lsah.is_binary_mask(gt))

    # Finalizing the initial image segmentation
    if(init_seg is None):
        # default: create an initial circular mask from the center
        center_row, center_col, radius_max = lsah.GetDefaultInitContourParams(img.shape)
        init_seg = lsah.create_circular_mask(img.shape, (center_row, center_col), radius_max // 2)

    if (isinstance(init_seg, str)):
        if (not os.path.exists(init_seg)):
            print("The initial segmentation path %s is not valid" % init_seg)
            sys.exit()
        if(init_seg.endswith('.npy')):
            init_s = np.load(init_seg)
            init_s = init_s[0, :, :, 0] # TODO: make sure this is valid for the data ultimately used
        else:
            # It's a path to an image segmentation file
            print("Image segmentation type currently not supported")
            sys.exit()
    elif (isinstance(init_seg, np.ndarray)):
        init_s=init_seg
    else:
        # Not a string or numpy array
        print("Image segmentation type currently not supported")
        sys.exit()

    print('Shape of the initial segmentation: ', init_s.shape)
    is_prob_mask = lsah.is_binary_mask(init_s) or lsah.is_probability_mask(init_s)
    print('Probability mask check for initial segmentation: ', is_prob_mask) # probabilities between 0 and 1

    # Note: image, ground truth, and initial segmentation all have the same shape.

    # --- Priniting initial dice score
    init_mask = np.round(init_s)
    init_dice_score = lsah.dice_score(init_mask, gt)
    init_iou_score = lsah.iou_score(init_mask, gt)
    print('Initial Dice score: ', init_dice_score)
    print('Initial IOU: ', init_iou_score)

    # Displays of everything before acm
    if (acm_dir):
        lsah.displayImage(img, "Image", True, save_dir=acm_dir)
        lsah.displayImage(gt, "Ground Truth", True, save_dir=acm_dir)
        init_s_title = 'Initial Segmentation' + ' - ' + 'DICE:{0:0.3f}'.format(init_dice_score) + ' - ' + 'IOU:{0:0.3f}'.format(init_iou_score)
        lsah.displayImage(init_s, init_s_title, True, save_dir=acm_dir)

    # ------- Note: AT THIS POINT IMAGE, GROUND TRUTH, AND INITIAL SEGMENTATION SHOULD BE IN A STATE FOR ACM PROCESSING

    # --- From initial segmentation get lambdas and initial phi (they would also have the same shape)

    # lambdas
    map_lambda1, map_lambda2 = lsa.get_lambda_maps(init_s)
    print('Shape of map lambda 1: ', map_lambda1.shape)
    print('Type of map lambda 1: ', type(map_lambda1), map_lambda1.dtype) # <class 'tensorflow.python.framework.ops.EagerTensor'> <dtype: 'float32'>
    print('Min max values of map lambda 1: ', np.min(map_lambda1), np.max(map_lambda1))
    print('Shape of map lambda 2: ', map_lambda2.shape)
    print('Type of map lambda 2: ', type(map_lambda2), map_lambda2.dtype) # <class 'tensorflow.python.framework.ops.EagerTensor'> <dtype: 'float32'>
    print('Min max values of map lambda 2: ', np.min(map_lambda2), np.max(map_lambda2))

    # initial phi
    initial_phi = lsa.get_initial_phi(init_s)
    print('Shape of initial phi: ', initial_phi.shape)
    print('Type of initial phi: ', type(initial_phi), initial_phi.dtype) # <class 'numpy.ndarray'> float32
    print('Min max values of initial phi: ', np.min(initial_phi), np.max(initial_phi))
    # phi (signed distance map) is zero on the contour and signed inside and outside.

    # --- Initialize parameter elems to active_contour_layer function
    elems = (img, initial_phi, map_lambda1, map_lambda2)
    input_image_size_x = img.shape[1]
    input_image_size_y = img.shape[0]
    print('Input image size x: ', input_image_size_x)
    print('Input image size y: ', input_image_size_y)

    # --- Call active_contour_layer function and get the final seg output
    final_seg, final_phi = lsa.active_contour_layer(elems=elems, input_image_size=input_image_size_x, input_image_size_2=input_image_size_y, 
                                         nu=nu, mu=mu, iter_limit = iter_lim, acm_dir=acm_dir, freq=save_freq, gt=gt) 
    print('Type of final segmentation mask: ', type(final_seg)) # <class 'tensorflow.python.framework.ops.EagerTensor'>
    print('Shape of final segmentation mask: ', final_seg.shape)
    print('binary mask check for final segmentation: ', lsah.is_binary_mask(final_seg))
    
    dice_score = lsah.dice_score(final_seg, gt)
    iou_score = lsah.iou_score(final_seg, gt)
    print('Final Dice score: ', dice_score)
    print('Final IOU score: ', iou_score)

    # Display of final result
    if(acm_dir):
        # final segmentation
        final_s_title = 'Final Segmentation' + ' - ' + 'DICE:{0:0.3f}'.format(dice_score) + ' - ' + 'IOU:{0:0.3f}'.format(iou_score)
        lsah.displayImage(final_seg, final_s_title, True, acm_dir)
        # final contours
        lsah.displayACMResult(img, initial_phi, final_phi, acm_dir)

    # Will be returning a dictionary of information in case needed
    result = {}
    result['image'] = img
    result['gt'] = gt
    result['init_seg'] = init_s
    result['final_seg_mask'] = final_seg
    result['initial dice score'] = init_dice_score
    result['final dice score'] = dice_score
    result['initial iou score'] = init_iou_score
    result['final iou score'] = iou_score

    return result
