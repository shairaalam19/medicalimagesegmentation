import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# Update system path
cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))

import lsa_run_helpers as lsarh
 
image_path = os.path.abspath(os.path.join(cwd, '../../data/ISIC/Train_data/ISIC_0000000.jpg'))
gt_path = os.path.abspath(os.path.join(cwd, '../../data/ISIC/Train_gt/ISIC_0000000_Segmentation.png'))

# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/Skin/gray'))
# lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, iter_lim=600)

# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/Skin/abc'))
# lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, abc=True, iter_lim=600)

# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/Skin/clahe'))
# lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, clahe=True, iter_lim=600)

# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/Skin/nu_0_mu_0'))
# lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, iter_lim=600, nu=0, mu=0)

acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/Skin/nu_10_mu_1'))
lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, iter_lim=600, nu=10, mu=1)

# ------
# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/Skin/new_pm'))
# lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, iter_lim=600) # Now this wont give same result
# # check a comment near default init_seg in run_lsa - uncommenting a line of code there would give same results for above line
# # basically tested less strict binary mask.
# -----