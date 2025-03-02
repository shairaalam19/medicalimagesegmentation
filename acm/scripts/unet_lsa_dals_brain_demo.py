import numpy as np
import os
import sys
import math

cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))

import lsa_helpers as lsah
import lsa_run_helpers as lsarh

# ----- Original Brain Demo
print('Original Brain Demo')
print()

image_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_input.npy'))
gt_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_label.npy'))
init_seg_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_initseg.npy'))

acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/brain/unet_init_contour'))
result = lsarh.run_lsa(image_path, gt_path, init_seg=init_seg_path, acm_dir=acm_dir, iter_lim=600)

#acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/brain/unet_init_contour/abc'))
#result = lsarh.run_lsa(image_path, gt_path, init_seg=init_seg_path, acm_dir=acm_dir, abc=True, iter_lim=600)

# ----- Testing non-square input version [expect identical results]
if(False):
    print()
    print()
    print('Non-square Input Version of Original Brain Demo')

    acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/brain/non_square_version'))
    # adding two columns at the end with all zero values to make non-square
    zero_columns = np.zeros((result['image'].shape[0], 2))
    image = np.hstack((result['image'], zero_columns))
    gt = np.hstack((result['gt'], zero_columns))
    init_seg = np.hstack((result['init_seg'], zero_columns))

    result_ns = lsarh.run_lsa(image, gt, init_seg=init_seg, acm_dir=acm_dir, iter_lim=600)

    print()
    print('Testing for identical results')
    assert(math.isclose(result['final dice score'], result_ns['final dice score']))
    assert(math.isclose(result['final iou score'], result_ns['final iou score']))