import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# Update system path
cwd = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../Level_Set_ACM')))

import lsa_run_helpers as lsarh

# Assumption: ISIC dataset is right outside the medicalimagesegmentation folder 
image_path = os.path.abspath(os.path.join(cwd, '../../../ISIC/Train_data/ISIC_0000000.jpg'))
gt_path = os.path.abspath(os.path.join(cwd, '../../../ISIC/Train_gt/ISIC_0000000_Segmentation.png'))
acm_dir = os.path.abspath(os.path.join(cwd, '../Results/DALS_LSA/Skin'))

lsarh.run_lsa(image_path, gt_path, acm_dir=acm_dir, iter_lim=600)