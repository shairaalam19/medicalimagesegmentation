# This is the final ACM reference we will use throughout the project.
# Level Set ACM as implemented in DALS.

# Hyperparameters: 
# 1. initial contour (location/shape/size).
# 2. nu - balloon force - (I think controls the length of the contour)
# 3. mu - regularization parameter that balances the trade-off between closely fitting the data and maintaining a smooth contour. Higher mu - more smoothing
# 4. number of iterations
# 5. narrow band width 
# 6. filter patch size (In fast lookup)
# 7. Size of square representing small region (non-fast-lookup version)

# Important hyperparameters to tune for now:
# 1. nu (controls length)
# 2. mu (controls smoothness)
# 3. number of iterations

# imports
import tensorflow as tf
from scipy.ndimage import distance_transform_edt
import numpy as np
#print("TensorFlow version:", tf.__version__) # 2.18.0

import lsa_helpers as lsah

# initializing some parameters to default values used in the algorithm
#input_image_size = 512 # height and width of the input image
# So I think img, phi, lambda maps are all square with the above dimension 
narrow_band_width = 1
fast_lookup = True
#nu = 5.0
#mu = 0.2
f_size = 15
#iter_limit = 300

# The function re_init_phi aims to reinitialize the level-set function phi 
# to maintain its signed distance property, which is crucial for the stability 
# and accuracy of level-set methods in image segmentation.
def re_init_phi(phi, dt, input_image_size_x, input_image_size_y):
    # D_left_shift = tf.cast(tf.manip.roll(phi, -1, axis=1), dtype='float32')
    # D_right_shift = tf.cast(tf.manip.roll(phi, 1, axis=1), dtype='float32')
    # D_up_shift = tf.cast(tf.manip.roll(phi, -1, axis=0), dtype='float32')
    # D_down_shift = tf.cast(tf.manip.roll(phi, 1, axis=0), dtype='float32')

    D_left_shift = tf.cast(tf.roll(phi, -1, axis=1), dtype='float32')
    D_right_shift = tf.cast(tf.roll(phi, 1, axis=1), dtype='float32')
    D_up_shift = tf.cast(tf.roll(phi, -1, axis=0), dtype='float32')
    D_down_shift = tf.cast(tf.roll(phi, 1, axis=0), dtype='float32')
    bp = D_left_shift - phi
    cp = phi - D_down_shift
    dp = D_up_shift - phi
    ap = phi - D_right_shift
    an = tf.identity(ap)
    bn = tf.identity(bp)
    cn = tf.identity(cp)
    dn = tf.identity(dp)
    ap = tf.clip_by_value(ap, 0, 10 ^ 38)
    bp = tf.clip_by_value(bp, 0, 10 ^ 38)
    cp = tf.clip_by_value(cp, 0, 10 ^ 38)
    dp = tf.clip_by_value(dp, 0, 10 ^ 38)
    an = tf.clip_by_value(an, -10 ^ 38, 0)
    bn = tf.clip_by_value(bn, -10 ^ 38, 0)
    cn = tf.clip_by_value(cn, -10 ^ 38, 0)
    dn = tf.clip_by_value(dn, -10 ^ 38, 0)
    area_pos = tf.where(phi > 0)
    area_neg = tf.where(phi < 0)
    pos_y = area_pos[:, 0]
    pos_x = area_pos[:, 1]
    neg_y = area_neg[:, 0]
    neg_x = area_neg[:, 1]
    tmp1 = tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [ap, bn]], axis=0)
    tmp1 += tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [cp, dn]], axis=0)
    update1 = tf.sqrt(tf.abs(tmp1)) - 1
    indices1 = tf.stack([pos_y, pos_x], 1)
    tmp2 = tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [an, bp]], axis=0)
    tmp2 += tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [cn, dp]], axis=0)
    update2 = tf.sqrt(tf.abs(tmp2)) - 1
    indices2 = tf.stack([neg_y, neg_x], 1)
    indices_final = tf.concat([indices1, indices2], 0)
    update_final = tf.concat([update1, update2], 0)
    dD = tf.scatter_nd(indices_final, update_final, shape=[input_image_size_y, input_image_size_x])
    S = tf.divide(phi, tf.square(phi) + 1)
    phi = phi - tf.multiply(dt * S, dD)

    return phi

def get_curvature(phi, x, y):
    phi_shape = tf.shape(phi)
    dim_y = phi_shape[0]
    dim_x = phi_shape[1]
    x = tf.cast(x, dtype="int32")
    y = tf.cast(y, dtype="int32")
    y_plus = tf.cast(y + 1, dtype="int32")
    y_minus = tf.cast(y - 1, dtype="int32")
    x_plus = tf.cast(x + 1, dtype="int32")
    x_minus = tf.cast(x - 1, dtype="int32")
    y_plus = tf.minimum(tf.cast(y_plus, dtype="int32"), tf.cast(dim_y - 1, dtype="int32"))
    x_plus = tf.minimum(tf.cast(x_plus, dtype="int32"), tf.cast(dim_x - 1, dtype="int32"))
    y_minus = tf.maximum(y_minus, 0)
    x_minus = tf.maximum(x_minus, 0)
    d_phi_dx = tf.gather_nd(phi, tf.stack([y, x_plus], 1)) - tf.gather_nd(phi, tf.stack([y, x_minus], 1))
    d_phi_dx_2 = tf.square(d_phi_dx)
    d_phi_dy = tf.gather_nd(phi, tf.stack([y_plus, x], 1)) - tf.gather_nd(phi, tf.stack([y_minus, x], 1))
    d_phi_dy_2 = tf.square(d_phi_dy)
    d_phi_dxx = tf.gather_nd(phi, tf.stack([y, x_plus], 1)) + tf.gather_nd(phi, tf.stack([y, x_minus], 1)) - \
                2 * tf.gather_nd(phi, tf.stack([y, x], 1))
    d_phi_dyy = tf.gather_nd(phi, tf.stack([y_plus, x], 1)) + tf.gather_nd(phi, tf.stack([y_minus, x], 1)) - \
                2 * tf.gather_nd(phi, tf.stack([y, x], 1))
    d_phi_dxy = 0.25 * (- tf.gather_nd(phi, tf.stack([y_minus, x_minus], 1)) - tf.gather_nd(phi, tf.stack(
        [y_plus, x_plus], 1)) + tf.gather_nd(phi, tf.stack([y_minus, x_plus], 1)) + tf.gather_nd(phi, tf.stack(
        [y_plus, x_minus], 1)))
    tmp_1 = tf.multiply(d_phi_dx_2, d_phi_dyy) + tf.multiply(d_phi_dy_2, d_phi_dxx) - \
            2 * tf.multiply(tf.multiply(d_phi_dx, d_phi_dy), d_phi_dxy)
    tmp_2 = tf.add(tf.pow(d_phi_dx_2 + d_phi_dy_2, 1.5), 2.220446049250313e-16)
    tmp_3 = tf.pow(d_phi_dx_2 + d_phi_dy_2, 0.5)
    tmp_4 = tf.divide(tmp_1, tmp_2)
    curvature = tf.multiply(tmp_3, tmp_4)
    mean_grad = tf.pow(d_phi_dx_2 + d_phi_dy_2, 0.5)

    return curvature, mean_grad


def get_intensity(image, masked_phi, filter_patch_size=5):
    #u_1 = tf.layers.average_pooling2d(tf.multiply(image, masked_phi), [filter_patch_size, filter_patch_size], 1,padding='SAME')
    #u_2 = tf.layers.average_pooling2d(masked_phi, [filter_patch_size, filter_patch_size], 1, padding='SAME')
    u_1 = tf.keras.layers.AveragePooling2D(pool_size=(filter_patch_size, filter_patch_size), strides=1, padding='SAME')(tf.multiply(image, masked_phi))
    u_2 = tf.keras.layers.AveragePooling2D(pool_size=(filter_patch_size, filter_patch_size), strides=1, padding='SAME')(masked_phi)
    u_2_prime = 1 - tf.cast((u_2 > 0), dtype='float32') + tf.cast((u_2 < 0), dtype='float32')
    u_2 = u_2 + u_2_prime + 2.220446049250313e-16

    return tf.divide(u_1, u_2)

# implements a level set-based active contour method
# elems: input dictionary containing all inputs to the level-set acm
# The function iteratively updates a level set function (phi), representing the boundary of an object in an image, based on energy minimization principles.
# The returned phi is a binary mask that can directly be compared with ground truth mask using dice function.
def active_contour_layer(elems, input_image_size, input_image_size_2 = None, nu = 5.0, mu = 0.2, iter_limit = 300, acm_dir=None, freq=None, gt=None):
    img = elems[0] # input image (I think intensity values) - 2D matrix [pixel values -]
    init_phi = elems[1] # The initial distance map, which defines the starting contour - 2D
    map_lambda1_acl = elems[2] # Weight map influencing the region-based energy terms - inside contour - 2D
    map_lambda2_acl = elems[3] # Weight map influencing the region-based energy terms - outside contour - 2D
    wind_coef = 3 # Determines the size of the local window around each point for intensity computations. (potential hyperparameter)
    zero_tensor = tf.constant(0, shape=[], dtype="int32") # Represents zero for bounds checking in TensorFlow.

    input_image_size_x = input_image_size
    input_image_size_y = input_image_size
    if (input_image_size_2 is not None):
        input_image_size_y = input_image_size_2

    # Each iteration does one phi update [Represents one iteration of level-set ACM]
    def _body(i, phi_level):
        print('Level Set ACM Iteration: ', int((i+1).numpy()))
        # --- Identify the pixels within the narrow band (a region around the zero level set of ϕ)
        # band_index: A boolean tensor indicating whether each pixel in ϕ lies within the narrow band defined by narrow_band_width.
        band_index = tf.reduce_all([phi_level <= narrow_band_width, phi_level >= -narrow_band_width], axis=0)
        # The coordinates of the pixels within this narrow band.
        band = tf.where(band_index)
        # Separate the x and y coordinates of the narrow band pixels
        band_y = band[:, 0]
        band_x = band[:, 1]
        # getting the total number of pixels in the narrow band
        shape_y = tf.shape(band_y)
        num_band_pixel = shape_y[0]
        # window_radii_x and window_radii_y: Define a fixed window size around each pixel for local calculations.
        window_radii_x = tf.ones(num_band_pixel) * wind_coef
        window_radii_y = tf.ones(num_band_pixel) * wind_coef

        # Computes mean intensities within local regions for pixels in the narrow band.
        # For a pixel in the narrow band, calculates the mean intensity of pixels inside and outside the level set in a local window.
        # Details:
        #   a) Extracts a local window around the current pixel.
        #   b) Calculates the mean intensity for pixels inside (inner) and outside (outer) the level set (ϕ≤0 and ϕ>0, respectively).
        #   c) Updates the running mean intensities (mean_intensities_outer, mean_intensities_inner)
        def body_intensity(j, mean_intensities_outer, mean_intensities_inner):
            xnew = tf.cast(band_x[j], dtype="float32")
            ynew = tf.cast(band_y[j], dtype="float32")
            window_radius_x = tf.cast(window_radii_x[j], dtype="float32")
            window_radius_y = tf.cast(window_radii_y[j], dtype="float32")
            local_window_x_min = tf.cast(tf.floor(xnew - window_radius_x), dtype="int32")
            local_window_x_max = tf.cast(tf.floor(xnew + window_radius_x), dtype="int32")
            local_window_y_min = tf.cast(tf.floor(ynew - window_radius_y), dtype="int32")
            local_window_y_max = tf.cast(tf.floor(ynew + window_radius_y), dtype="int32")
            local_window_x_min = tf.maximum(zero_tensor, local_window_x_min)
            local_window_y_min = tf.maximum(zero_tensor, local_window_y_min)
            local_window_x_max = tf.minimum(tf.cast(input_image_size_x - 1, dtype="int32"), local_window_x_max)
            local_window_y_max = tf.minimum(tf.cast(input_image_size_y - 1, dtype="int32"), local_window_y_max)
            local_image = img[local_window_y_min: local_window_y_max + 1,local_window_x_min: local_window_x_max + 1]
            local_phi = phi_level[local_window_y_min: local_window_y_max + 1,local_window_x_min: local_window_x_max + 1]
            inner = tf.where(local_phi <= 0)
            area_inner = tf.cast(tf.shape(inner)[0], dtype='float32')
            outer = tf.where(local_phi > 0)
            area_outer = tf.cast(tf.shape(outer)[0], dtype='float32')
            image_loc_inner = tf.gather_nd(local_image, inner)
            image_loc_outer = tf.gather_nd(local_image, outer)
            mean_intensity_inner = tf.cast(tf.divide(tf.reduce_sum(image_loc_inner), area_inner), dtype='float32')
            mean_intensity_outer = tf.cast(tf.divide(tf.reduce_sum(image_loc_outer), area_outer), dtype='float32')
            mean_intensities_inner = tf.concat(axis=0, values=[mean_intensities_inner[:j], [mean_intensity_inner]])
            mean_intensities_outer = tf.concat(axis=0, values=[mean_intensities_outer[:j], [mean_intensity_outer]])

            return (j + 1, mean_intensities_outer, mean_intensities_inner)

        # Compute the mean intensities using either a fast lookup or a more detailed iterative approach.
        # The code by default goes into fast_lookup
        if fast_lookup:
            # Reshaping distance map and image into 4 dimensions
            phi_4d = phi_level[tf.newaxis, :, :, tf.newaxis]
            image = img[tf.newaxis, :, :, tf.newaxis]
            # Computing the new band indices around the contour
            band_index_2 = tf.reduce_all([phi_4d <= narrow_band_width, phi_4d >= -narrow_band_width], axis=0)
            band_2 = tf.where(band_index_2)
            # phi_4d <= 0 and phi_4d > 0 create masks for the inner and outer regions relative to the zero level-set.
            # tf.cast(..., dtype='float32') converts these boolean masks into float tensors (0.0 for False, 1.0 for True).
            # get_intensity computes the average intensity in a local region of the image 
            u_inner = get_intensity(image, tf.cast((([phi_4d <= 0])), dtype='float32')[0], filter_patch_size=f_size)
            u_outer = get_intensity(image, tf.cast((([phi_4d > 0])), dtype='float32')[0], filter_patch_size=f_size)
            # tf.gather_nd retrieves the values of u_inner and u_outer at the indices specified by band_2
            # These operations collect the computed mean intensities for the narrow band pixels, producing arrays of mean intensities 
            # for the inner and outer regions.
            mean_intensities_inner = tf.gather_nd(u_inner, band_2)
            mean_intensities_outer = tf.gather_nd(u_outer, band_2)

        else:
            # Uses a loop to compute mean intensities for each pixel in the narrow band individually.
            mean_intensities_inner = tf.constant([0], dtype='float32')
            mean_intensities_outer = tf.constant([0], dtype='float32')
            j = tf.constant(0, dtype=tf.int32)
            _, mean_intensities_outer, mean_intensities_inner = tf.while_loop(
                lambda j, mean_intensities_outer, mean_intensities_inner:
                j < num_band_pixel, body_intensity, loop_vars=[j, mean_intensities_outer, mean_intensities_inner],
                shape_invariants=[j.get_shape(), tf.TensorShape([None]), tf.TensorShape([None])])

        # --- Compute the update for the level set function ϕ
        lambda1 = tf.gather_nd(map_lambda1_acl, [band]) # gathering the lambda 1 values in the narrow band
        lambda2 = tf.gather_nd(map_lambda2_acl, [band]) # gathering the lambda 2 values in the narrow band
        # Computes the curvature and mean gradient at the band locations of phi_level. 
        # Curvature is related to the shape of the level set, and the mean gradient helps in regularizing the contour.
        # Curvature and mean_grad are computed for each corresponding (x, y) coordinate, 
        #   so their size will match the size of the input coordinate arrays x and y.
        curvature, mean_grad = get_curvature(phi_level, band_x, band_y) 
        # Multiplies the curvature by the mean gradient to get the combined effect of curvature regularization at each point in the narrow band.
        kappa = tf.multiply(curvature, mean_grad)
        # Computes the first term of the energy, which represents the inner region. 
        # It squares the difference between the pixel values (at band indices) and mean_intensities_inner, then scales it by lambda1.
        # term1 = tf.multiply(tf.cast(lambda1, dtype='float32'),tf.square(tf.gather_nd(img, [band]) - mean_intensities_inner))
        term1 = tf.multiply(tf.cast(lambda1, dtype='float32'),tf.square(tf.cast(tf.gather_nd(img, [band]), dtype='float32') - mean_intensities_inner))
        # Similar to term1, but for the outer region. It scales the squared difference between the pixel values and mean_intensities_outer by lambda2.
        # term2 = tf.multiply(tf.cast(lambda2, dtype='float32'),tf.square(tf.gather_nd(img, [band]) - mean_intensities_outer))
        term2 = tf.multiply(tf.cast(lambda2, dtype='float32'),tf.square(tf.cast(tf.gather_nd(img, [band]), dtype='float32') - mean_intensities_outer))
        # Computes the force applied to the level set function by combining the constant -nu (usually a balloon force) with term1 and term2. 
        # This force dictates the movement of the contour based on the image features.
        # Again same length as number of bands in the narrow band width
        force = -nu + term1 - term2
        # Normalizes the force by dividing it by its maximum absolute value to prevent instability in the updates.
        force /= (tf.reduce_max(tf.abs(force)))
        # Calculates the rate of change of phi by adding the force and the product of mu and kappa (which incorporates the curvature term).
        d_phi_dt = tf.cast(force, dtype="float32") + tf.cast(mu * kappa, dtype="float32")
        # Computes a time step dt for the update, ensuring it is scaled properly to maintain stability in the evolution of phi.
        dt = .45 / (tf.reduce_max(tf.abs(d_phi_dt)) + 2.220446049250313e-16)
        # Calculates the actual change d_phi to apply to phi by multiplying the rate of change by the time step.
        d_phi = dt * d_phi_dt
        # --- Apply the calculated update to the level set function ϕ
        # Assigns d_phi to update_narrow_band to be applied to the narrow band of the level set function.
        update_narrow_band = d_phi
        # Updates phi_level by adding update_narrow_band at the indices specified by band. This operation selectively updates only the points in the narrow band.
        phi_level = phi_level + tf.scatter_nd([band], tf.cast(update_narrow_band, dtype='float32'),shape=[input_image_size_y, input_image_size_x])
        # Re-initialize ϕ to ensure numerical stability.
        # Reinitializes phi_level using re_init_phi to maintain its signed distance property and ensure numerical stability in subsequent iterations.
        # General note on signed distance property of phi:
        #   A signed distance function phi (x,y) represents the distance from a point (x,y) to the closest point on a contour (or interface), 
        #       with a sign indicating whether the point is inside or outside the contour:
        phi_level = re_init_phi(phi_level, 0.5, input_image_size_x, input_image_size_y)

        if(acm_dir):
            if((i+1)%freq == 0):
                phi_img = tf.round(tf.cast((1 - tf.nn.sigmoid(phi_level)), dtype=tf.float32))
                dice_score = lsah.dice_score(phi_img, gt)
                iou_score = lsah.iou_score(phi_img, gt)
                img_title = 'Mask ' + str(int((i+1).numpy())) + ' - ' + 'DICE:{0:0.3f}'.format(dice_score) + ' - ' + 'IOU:{0:0.3f}'.format(iou_score)
                print('intermediate dice score: ', dice_score)
                print('intermediate iou score: ', iou_score)
                lsah.displayImage(phi_img, img_title, True, acm_dir)

        return (i + 1, phi_level)

    i = tf.constant(0, dtype=tf.int32) # Defining constant 0
    phi = init_phi # setting initial phi for the while loop
    _, phi = tf.while_loop(lambda i, phi: i < iter_limit, _body, loop_vars=[i, phi]) # loop over body with iter_num iterations to iteratively update phi
    phi_dis_map = phi
    # Now phi is the final distance map after all the level set acm iterations
    # Sigmoid layer:
    phi = tf.round(tf.cast((1 - tf.nn.sigmoid(phi)), dtype=tf.float32))
    # This line transforms the final level set function phi into a binary mask representation suitable for segmentation. 
    # It leverages the sigmoid function to convert the values of phi into a smooth range of probabilities and then thresholds these probabilities to 
    # produce a binary mask. After applying sigmoid, casting, and rounding, the phi becomes a binary mask where each element is either 0 or 1.

    #return phi,init_phi, map_lambda1_acl, map_lambda2_acl # Later the code just ends up using the final phi returned.
    return phi, phi_dis_map

def my_func(mask):
    epsilon = 0
    def bwdist(im): return distance_transform_edt(np.logical_not(im))
    bw = mask
    signed_dist = bwdist(bw) - bwdist(1 - bw)
    d = signed_dist.astype(np.float32)
    d += epsilon
    while np.count_nonzero(d < 0) < 5:
        d -= 1

    return d

def get_lambda_maps(out_seg):
    map_lambda1 = tf.exp(tf.divide(tf.subtract(2.0,out_seg),tf.add(1.0,out_seg)))
    map_lambda2 = tf.exp(tf.divide(tf.add(1.0, out_seg), tf.subtract(2.0, out_seg)))
    return map_lambda1, map_lambda2

def get_initial_phi(out_seg):
    binary_seg = tf.round(out_seg)
    dt_trans = my_func(binary_seg)
    return dt_trans