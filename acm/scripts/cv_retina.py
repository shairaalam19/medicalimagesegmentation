import os
import sys
import numpy as np

# Update system path
cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))

import lsa_run_helpers as lsarh

# Paths
image_path = os.path.abspath(os.path.join(cwd, '../../data/chase_db1/Image_01L.jpg')) # (960, 999)
gt_path = os.path.abspath(os.path.join(cwd, '../../data/chase_db1/Image_01L_1stHO.png'))

acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Chan_Vese_LSA/Retina/abc/new_lsf'))

center_row = 800
center_col = 400
init_lsf = np.ones((960,999), np.float64)
row_min = center_row - 25
row_max = center_row + 25
col_min = center_col - 25
col_max = center_col + 25
init_lsf[row_min:row_max,col_min:col_max]= -1

# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Chan_Vese_LSA/Retina/trivial_lsf/gray'))
# lsarh.run_cv(image_path, gt_path, acm_dir=acm_dir, iter_lim=150)

# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Chan_Vese_LSA/Retina/trivial_lsf/abc'))
# lsarh.run_cv(image_path, gt_path, acm_dir=acm_dir, abc=True, iter_lim=150)

# acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Chan_Vese_LSA/Retina/trivial_lsf/clahe'))
# lsarh.run_cv(image_path, gt_path, acm_dir=acm_dir, clahe=True, iter_lim=150)

acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Chan_Vese_LSA/Retina/focussed_lsf/gray'))
lsarh.run_cv(image_path, gt_path, init_lsf=init_lsf , acm_dir=acm_dir, save_freq=2)

acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Chan_Vese_LSA/Retina/focussed_lsf/abc'))
lsarh.run_cv(image_path, gt_path, init_lsf=init_lsf , acm_dir=acm_dir, abc=True, save_freq=2, nu=0, mu=0)

acm_dir = os.path.abspath(os.path.join(cwd, '../Results/Chan_Vese_LSA/Retina/focussed_lsf/clahe'))
lsarh.run_cv(image_path, gt_path, init_lsf=init_lsf , acm_dir=acm_dir, clahe=True, save_freq=2, nu=0, mu=0)
