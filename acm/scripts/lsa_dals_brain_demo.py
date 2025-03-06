import numpy as np
import os
import sys

cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))

import lsa_helpers as lsah
import lsa_run_helpers as lsarh

# ----- DEMO BRAIN (very trivial initial contour)
if(True):
    print('BRAIN DEMO - trivial initial contour')
    print()

    image_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_input.npy'))
    gt_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_label.npy'))

    # acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/brain/trivial_initial_contour/gray'))
    # lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, iter_lim=600)

    # acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/brain/trivial_initial_contour/abc'))
    # lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, abc=True, iter_lim=600)

    acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/brain/trivial_initial_contour/clahe'))
    lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, clahe=True, iter_lim=600)

if(False):
    # ----- DEMO BRAIN (intermediate initial contour)
    print()
    print()
    print('BRAIN DEMO - intermediate initial contour')

    image_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_input.npy'))
    gt_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_label.npy'))

    img = np.load(image_path)
    initial_probability_mask = lsah.create_circular_mask(img.shape, (90, 170), 40)

    # acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/brain/intermediate_initial_contour/gray'))
    # lsarh.run_lsa(image_path, gt_path, init_seg=initial_probability_mask, acm_dir=acm_dir, iter_lim=600)

    # acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/brain/intermediate_initial_contour/abc'))
    # lsarh.run_lsa(image_path, gt_path, init_seg=initial_probability_mask, acm_dir=acm_dir, abc=True, iter_lim=600)

    acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/brain/intermediate_initial_contour/clahe'))
    lsarh.run_lsa(image_path, gt_path, init_seg=initial_probability_mask, acm_dir=acm_dir, clahe=True, iter_lim=600)