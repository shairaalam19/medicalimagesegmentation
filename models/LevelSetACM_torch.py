# This is the final ACM reference we will use throughout the project.
# Level Set ACM as implemented in DALS.

# Important hyperparameters:
# 1. nu (controls length) - balloon force - (I think controls the length of the contour)
# 2. mu (controls smoothness) - regularization parameter that balances the trade-off between closely fitting the data and maintaining a smooth contour. Higher mu - more smoothing
# 3. number of iterations
# 4. initial contour (location/shape/size)

# imports
from scipy.ndimage import distance_transform_edt
import numpy as np
import torch
import torch.nn.functional as F
import sys
#print("TensorFlow version:", tf.__version__) # 2.18.0

narrow_band_width = 1
f_size = 15

def re_init_phi(phi, dt, input_image_size_x, input_image_size_y):
    # Shift operations using torch.roll
    D_left_shift = torch.roll(phi, shifts=-1, dims=1).float()
    D_right_shift = torch.roll(phi, shifts=1, dims=1).float()
    D_up_shift = torch.roll(phi, shifts=-1, dims=0).float()
    D_down_shift = torch.roll(phi, shifts=1, dims=0).float()

    # Compute differences
    bp = D_left_shift - phi
    cp = phi - D_down_shift
    dp = D_up_shift - phi
    ap = phi - D_right_shift

    # Clone to preserve original values
    an = ap.clone()
    bn = bp.clone()
    cn = cp.clone()
    dn = dp.clone()

    # Clipping values
    # max_val = 10 ** 38
    # min_val = -10 ** 38
    max_val = torch.finfo(phi.dtype).max
    min_val = torch.finfo(phi.dtype).min

    ap = torch.clamp(ap, 0, max_val)
    bp = torch.clamp(bp, 0, max_val)
    cp = torch.clamp(cp, 0, max_val)
    dp = torch.clamp(dp, 0, max_val)

    an = torch.clamp(an, min_val, 0)
    bn = torch.clamp(bn, min_val, 0)
    cn = torch.clamp(cn, min_val, 0)
    dn = torch.clamp(dn, min_val, 0)

    # Find positive and negative regions
    area_pos = torch.nonzero(phi > 0, as_tuple=False)
    area_neg = torch.nonzero(phi < 0, as_tuple=False)

    # if area_pos.numel() == 0 or area_neg.numel() == 0:
    #     return phi

    pos_y, pos_x = area_pos[:, 0], area_pos[:, 1]
    neg_y, neg_x = area_neg[:, 0], area_neg[:, 1]

    # Compute updates
    tmp1 = torch.max(torch.stack([ap[pos_y, pos_x]**2, bn[pos_y, pos_x]**2]), dim=0)[0]
    tmp1 += torch.max(torch.stack([cp[pos_y, pos_x]**2, dn[pos_y, pos_x]**2]), dim=0)[0]
    update1 = torch.sqrt(torch.abs(tmp1) + 2.220446049250313e-16) - 1
    #update1 = torch.sqrt(torch.abs(tmp1)) - 1

    tmp2 = torch.max(torch.stack([an[neg_y, neg_x]**2, bp[neg_y, neg_x]**2]), dim=0)[0]
    tmp2 += torch.max(torch.stack([cn[neg_y, neg_x]**2, dp[neg_y, neg_x]**2]), dim=0)[0]
    #print('Value of tmp 2: ', tmp2)
    update2 = torch.sqrt(torch.abs(tmp2) + 2.220446049250313e-16) - 1 # torch.abs() has a gradient discontinuity at 0
    #update2 = torch.sqrt(torch.abs(tmp2)) - 1

    # Create indices and updates
    indices1 = torch.stack([pos_y, pos_x], dim=1)
    indices2 = torch.stack([neg_y, neg_x], dim=1)
    indices_final = torch.cat([indices1, indices2], dim=0)
    update_final = torch.cat([update1, update2], dim=0)

    # Scatter updates
    dD = torch.zeros((input_image_size_y, input_image_size_x), dtype=phi.dtype, device=phi.device)
    dD[indices_final[:, 0], indices_final[:, 1]] = update_final

    # Compute sign function
    S = phi / (phi**2 + 1)
    
    # Update phi
    phi = phi - (dt * S * dD)

    return phi

def get_curvature(phi, x, y):
    phi_shape = phi.shape
    dim_y, dim_x = phi_shape[0], phi_shape[1]
    
    x = x.to(dtype=torch.int32)
    y = y.to(dtype=torch.int32)

    y_plus = torch.clamp(y + 1, max=dim_y - 1)
    y_minus = torch.clamp(y - 1, min=0)
    x_plus = torch.clamp(x + 1, max=dim_x - 1)
    x_minus = torch.clamp(x - 1, min=0)

    d_phi_dx = phi[y, x_plus] - phi[y, x_minus]
    d_phi_dx_2 = d_phi_dx ** 2
    d_phi_dy = phi[y_plus, x] - phi[y_minus, x]
    d_phi_dy_2 = d_phi_dy ** 2

    d_phi_dxx = phi[y, x_plus] + phi[y, x_minus] - 2 * phi[y, x]
    d_phi_dyy = phi[y_plus, x] + phi[y_minus, x] - 2 * phi[y, x]

    d_phi_dxy = 0.25 * (-phi[y_minus, x_minus] - phi[y_plus, x_plus] + phi[y_minus, x_plus] + phi[y_plus, x_minus])

    tmp_1 = d_phi_dx_2 * d_phi_dyy + d_phi_dy_2 * d_phi_dxx - 2 * (d_phi_dx * d_phi_dy * d_phi_dxy)
    tmp_2 = (d_phi_dx_2 + d_phi_dy_2) ** 1.5 + 2.220446049250313e-16
    tmp_3 = (d_phi_dx_2 + d_phi_dy_2 + 2.220446049250313e-16) ** 0.5
    tmp_4 = tmp_1 / tmp_2
    curvature = tmp_3 * tmp_4
    mean_grad = (d_phi_dx_2 + d_phi_dy_2 + 2.220446049250313e-16) ** 0.5

    return curvature, mean_grad

def get_intensity(image, masked_phi, filter_patch_size=5):

    # print('image')
    # print(image)
    # print()

    # print('masked_phi')
    # print(masked_phi)
    # print()

    u_1 = F.avg_pool2d(image * masked_phi, kernel_size=filter_patch_size, stride=1, padding=filter_patch_size // 2)

    # print('u_1')
    # print(u_1)
    # print()

    u_2 = F.avg_pool2d(masked_phi, kernel_size=filter_patch_size, stride=1, padding=filter_patch_size // 2)

    u_2_prime = 1 - (u_2 > 0).float() + (u_2 < 0).float()
    #u_2 = u_2 + u_2_prime + 1e-16
    u_2 = u_2 + u_2_prime + 2.220446049250313e-16

    return u_1 / u_2

# def get_intensity(image, masked_phi, filter_patch_size=5):

#     # print('image')
#     # print(image)
#     # print()

#     # print('masked_phi')
#     # print(masked_phi)
#     # print()


#     pad_total = filter_patch_size - 1
#     pad_left = pad_total // 2
#     pad_right = pad_total - pad_left
#     pad_top = pad_left
#     pad_bottom = pad_right

#     # Apply explicit padding before pooling
#     padded_input = F.pad(image * masked_phi, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

#     # Apply average pooling
#     u_1 = F.avg_pool2d(padded_input, kernel_size=filter_patch_size, stride=1)

#     # print('u_1')
#     # print(u_1)
#     # print()

#     u_2 = F.avg_pool2d(masked_phi, kernel_size=filter_patch_size, stride=1, padding=filter_patch_size // 2)

#     u_2_prime = 1 - (u_2 > 0).float() + (u_2 < 0).float()
#     #u_2 = u_2 + u_2_prime + 1e-16
#     u_2 = u_2 + u_2_prime + 2.220446049250313e-16

#     return u_1 / u_2

def active_contour_layer(elems, input_image_size, input_image_size_2=None, nu=5.0, mu=0.2, iter_limit=300):
    img = elems[0]  # input image - 2D tensor
    init_phi = elems[1]  # initial distance map - 2D tensor
    map_lambda1_acl = elems[2]  # weight map inside contour - 2D tensor
    map_lambda2_acl = elems[3]  # weight map outside contour - 2D tensor
    
    input_image_size_x = input_image_size
    input_image_size_y = input_image_size_2 if input_image_size_2 is not None else input_image_size

    def _body(i, phi_level):
        print()
        print()
        print('Level Set ACM Iteration: ', int((i+1).numpy()))
        #print(phi_level)

        # --- Identify the pixels within the narrow band
        band_index = torch.logical_and(phi_level <= narrow_band_width, phi_level >= -narrow_band_width)
        band = torch.nonzero(band_index, as_tuple=False)
        band_y, band_x = band[:, 0], band[:, 1]  # Separate x and y coordinates
        print('Shape of bands: ', band_index.shape, band.shape, band_y.shape, band_x.shape)
        #print(band)

        # Reshape distance map and image into 4D tensors
        phi_4d = phi_level.unsqueeze(0).unsqueeze(-1)  # Shape: (1, H, W, 1)
        image = img.unsqueeze(0).unsqueeze(-1)  # Shape: (1, H, W, 1)
        print('Phi_4d and image shapes: ', phi_4d.shape, image.shape)

        # Compute new band indices
        band_index_2 = torch.logical_and(phi_4d <= narrow_band_width, phi_4d >= -narrow_band_width)
        band_2 = torch.nonzero(band_index_2, as_tuple=False)
        print('Band index 2 and band 2 shapes: ', band_index_2.shape, band_2.shape)

        # Compute intensities inside and outside level set
        u_inner = get_intensity(image, (phi_4d <= 0).float()[0], filter_patch_size=f_size)
        u_outer = get_intensity(image, (phi_4d > 0).float()[0], filter_patch_size=f_size)
        print('u_inner and u_outer shapes: ', u_inner.shape, u_outer.shape)
        print("Checking if get intensity is returning any nan: ", torch.isnan(u_inner).any(), torch.isnan(u_outer).any())
        # print(u_inner)
        # print(u_outer)
        # sys.exit()

        # Gather mean intensities for the narrow band
        mean_intensities_inner = u_inner[band_2[:, 0], band_2[:, 1], band_2[:, 2], band_2[:, 3]]
        mean_intensities_outer = u_outer[band_2[:, 0], band_2[:, 1], band_2[:, 2], band_2[:, 3]]
        print('Shape of mean intensities', mean_intensities_inner.shape, mean_intensities_outer.shape)
        print("Checking if mean intensities are nan: ", torch.isnan(mean_intensities_inner).any(), torch.isnan(mean_intensities_outer).any())
        # print(mean_intensities_inner)
        # print(mean_intensities_outer)

        # Gather lambda1 and lambda2 values in the narrow band
        lambda1 = map_lambda1_acl[band[:, 0], band[:, 1]]
        lambda2 = map_lambda2_acl[band[:, 0], band[:, 1]]
        print('Shape of lambdas: ', lambda1.shape, lambda2.shape)

        # Compute curvature and gradient regularization
        curvature, mean_grad = get_curvature(phi_level, band_x, band_y)
        kappa = curvature * mean_grad
        print('Shape of curvature terms: ', curvature.shape, mean_grad.shape, kappa.shape)
        print("Checking curvature, mean_grad, kappa has nans: ", torch.isnan(curvature).any(), torch.isnan(mean_grad).any(), torch.isnan(kappa).any())
        # print(curvature)
        # print(mean_grad)
        # print(kappa)

        # Compute energy terms
        term1 = lambda1.float() * (img[band[:, 0], band[:, 1]].float() - mean_intensities_inner) ** 2
        term2 = lambda2.float() * (img[band[:, 0], band[:, 1]].float() - mean_intensities_outer) ** 2
        # Compute force and normalize
        force = -nu + term1 - term2
        if force.numel() > 0: force = force / (torch.max(torch.abs(force)) + 2.220446049250313e-16)
        print('Term and force shapes: ', term1.shape, term2.shape, force.shape)
        print("Checking term1, term2, force has nans: ", torch.isnan(term1).any(), torch.isnan(term2).any(), torch.isnan(force).any())

        # Compute update step
        d_phi_dt = force + mu * kappa.float()
        dt = 0.45 / (torch.max(torch.abs(d_phi_dt)) + 2.220446049250313e-16)
        d_phi = dt * d_phi_dt
        print('Shape of gradients: ', d_phi_dt.shape, dt.shape, d_phi.shape)
        print('Checking if gradients have any nans: ', torch.isnan(d_phi_dt).any(), torch.isnan(dt).any(), torch.isnan(d_phi).any())

        # --- Apply the update to the level set function ϕ
        phi_update = torch.zeros_like(phi_level)
        phi_update[band[:, 0], band[:, 1]] = d_phi
        print('Phi Update Shape', phi_update.shape) # 1024, 1024
        print('Checking if phi update has any nans: ', torch.isnan(phi_update).any())

        phi_level = phi_level + phi_update  # Update phi_level in the narrow band
        print('Extreme values in phi_level: ', torch.max(phi_level), torch.min(phi_level))
        # Reinitialize phi for numerical stability
        phi_level = re_init_phi(phi_level, 0.5, input_image_size_x, input_image_size_y)
        print('Phi level Shape', phi_level.shape)
        print('Checking if re-initialized phi has any nans: ', torch.isnan(phi_level).any())
        #print(phi_level)

        return i + 1, phi_level

    i = torch.tensor(0, dtype=torch.int32)
    phi = init_phi
    while i < iter_limit.long():
        i, phi = _body(i, phi)
    
    phi_dis_map = phi
    final_prob_mask = (1 - torch.sigmoid(phi)).float()
    phi = torch.round(final_prob_mask)
    
    return phi, phi_dis_map, final_prob_mask

def my_func(mask):
    epsilon = 0
    # Helper function for distance transform using SciPy
    def bwdist(im):
        im = im.detach().cpu().numpy()  # Convert tensor to NumPy array for SciPy
        return distance_transform_edt(np.logical_not(im))
    bw = mask
    signed_dist = bwdist(bw) - bwdist(1 - bw)
    # Convert back to PyTorch tensor, but retain the original tensor's device
    # We must ensure that the new tensor is on the same device as the original mask.
    d = torch.tensor(signed_dist, dtype=torch.float32, device=mask.device)

    # Now ensure the new tensor is part of the computation graph, so we need to enable gradients on it.
    # d.requires_grad_()  # Re-enable gradient tracking for the tensor
    #print(d.shape, d[0][0])

    d += epsilon

    if(isinstance(d,torch.Tensor)):
        while torch.count_nonzero(d < 0) < 5:
            d -= 1
    else:
        while np.count_nonzero(d < 0) < 5:
            d -= 1
    
    return d

def get_lambda_maps(out_seg):
    map_lambda1 = torch.exp((2.0 - out_seg) / (1.0 + out_seg))
    map_lambda2 = torch.exp((1.0 + out_seg) / (2.0 - out_seg))
    return map_lambda1, map_lambda2

def get_initial_phi(out_seg):
    binary_seg = torch.round(out_seg)
    dt_trans = my_func(binary_seg) 
    return dt_trans