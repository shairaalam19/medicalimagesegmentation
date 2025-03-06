import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Update system path
cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))

# Imports
import lsa_run_helpers as lsarh

# Paths
image_path = os.path.abspath(os.path.join(cwd, '../../data/chase_db1/Image_01L.jpg'))
gt_path = os.path.abspath(os.path.join(cwd, '../../data/chase_db1/Image_01L_1stHO.png'))


# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/Retina/gray'))
# lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, iter_lim=600)

# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/Retina/abc'))
# lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, abc=True, iter_lim=600)

acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/Retina/clahe'))
lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, clahe=True, iter_lim=600)

