import os
import sys
import numpy as np

# Update system path
cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))

import lsa_run_helpers as lsarh

# Paths
image_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_input.npy'))
gt_path = os.path.abspath(os.path.join(cwd, '../dals_demo_brain/img1_label.npy'))

arr = np.load(image_path)
rows, cols = arr.shape
#print(rows, cols) # 256, 256

center_row = 90
center_col = 170
init_lsf = np.ones((rows,cols), np.float64)
row_min = center_row - 40
row_max = center_row + 40
col_min = center_col - 40
col_max = center_col + 40
init_lsf[row_min:row_max,col_min:col_max]= -1


# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Chan_Vese_LSA/Brain/abc/'))
# lsarh.run_cv(image_path, gt_path, init_lsf=init_lsf , acm_dir=acm_dir, abc=True, iter_lim=100, save_freq=20)

acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Chan_Vese_LSA/Brain/clahe/'))
lsarh.run_cv(image_path, gt_path, init_lsf=init_lsf , acm_dir=acm_dir, clahe=True, iter_lim=100, save_freq=20)