# import statements

import tensorflow as tf
print("TensorFlow version:", tf.__version__)



# initializing some parameters to default values used in the algorithm
input_image_size = 512 # height and width of the input image
# So I think img, phi, lambda maps are all square with the above dimension 
narrow_band_width = 1
fast_lookup = True
nu = 5.0
mu = 0.2
f_size = 15
iter_limit = 300

# ========= ACM =========
def re_init_phi(phi, dt):
    D_left_shift = tf.cast(tf.manip.roll(phi, -1, axis=2), dtype='float32')
    D_right_shift = tf.cast(tf.manip.roll(phi, 1, axis=2), dtype='float32')
    D_up_shift = tf.cast(tf.manip.roll(phi, -1, axis=1), dtype='float32')
    D_down_shift = tf.cast(tf.manip.roll(phi, 1, axis=1), dtype='float32')
    D_in_shift = tf.cast(tf.manip.roll(phi, -1, axis=0), dtype='float32')
    D_out_shift = tf.cast(tf.manip.roll(phi, 1, axis=0), dtype='float32')
    bp = D_left_shift - phi
    cp = phi - D_down_shift
    dp = D_up_shift - phi
    ap = phi - D_right_shift
    gp = phi - D_out_shift
    fp = D_in_shift - phi
    an = tf.identity(ap)
    bn = tf.identity(bp)
    cn = tf.identity(cp)
    dn = tf.identity(dp)
    gn = tf.identity(gp)
    fn = tf.identity(fp)
    ap = tf.clip_by_value(ap, 0, 10 ^ 38)
    bp = tf.clip_by_value(bp, 0, 10 ^ 38)
    cp = tf.clip_by_value(cp, 0, 10 ^ 38)
    dp = tf.clip_by_value(dp, 0, 10 ^ 38)
    gp = tf.clip_by_value(gp, 0, 10 ^ 38)
    fp = tf.clip_by_value(fp, 0, 10 ^ 38)
    an = tf.clip_by_value(an, -10 ^ 38, 0)
    bn = tf.clip_by_value(bn, -10 ^ 38, 0)
    cn = tf.clip_by_value(cn, -10 ^ 38, 0)
    dn = tf.clip_by_value(dn, -10 ^ 38, 0)
    gn = tf.clip_by_value(gn, -10 ^ 38, 0)
    fn = tf.clip_by_value(fn, -10 ^ 38, 0)
    area_pos = tf.where(phi > 0)
    area_neg = tf.where(phi < 0)
    pos_y = area_pos[:, 1]
    pos_x = area_pos[:, 2]
    pos_z = area_pos[:, 0]
    neg_y = area_neg[:, 1]
    neg_x = area_neg[:, 2]
    neg_z = area_neg[:, 0]
    tmp1 = tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [ap, bn]], axis=0)
    tmp1 += tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [cp, dn]], axis=0)
    tmp1 += tf.reduce_max([tf.square(tf.gather_nd(t, area_pos)) for t in [gp, fn]], axis=0)
    update1 = tf.sqrt(tf.abs(tmp1)) - 1
    indices1 = tf.stack([pos_y, pos_x], 1)
    tmp2 = tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [an, bp]], axis=0)
    tmp2 += tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [cn, dp]], axis=0)
    tmp2 += tf.reduce_max([tf.square(tf.gather_nd(t, area_neg)) for t in [gn, df]], axis=0)
    update2 = tf.sqrt(tf.abs(tmp2)) - 1
    indices2 = tf.stack([neg_y, neg_x], 1)
    indices_final = tf.concat([indices1, indices2], 0)
    update_final = tf.concat([update1, update2], 0)
    dD = tf.scatter_nd(indices_final, update_final, shape=[input_image_size, input_image_size, input_image_z])
    S = tf.divide(phi, tf.square(phi) + 1)
    phi = phi - tf.multiply(dt * S, dD)

    return phi

def get_curvature(phi, x, y, z):
    phi_shape = tf.shape(phi)
    dim_y = phi_shape[1]
    dim_x = phi_shape[2]
    dim_z = phi_shape[0]
    x = tf.cast(x, dtype="int32")
    y = tf.cast(y, dtype="int32")
    z = tf.cast(z, dtype="int32")
    y_plus = tf.cast(y + 1, dtype="int32")
    y_minus = tf.cast(y - 1, dtype="int32")
    x_plus = tf.cast(x + 1, dtype="int32")
    x_minus = tf.cast(x - 1, dtype="int32")
    z_plus = tf.cast(z + 1, dtype="int32")
    z_minus = tf.cast(z - 1, dtype="int32")
    y_plus = tf.minimum(tf.cast(y_plus, dtype="int32"), tf.cast(dim_y - 1, dtype="int32"))
    x_plus = tf.minimum(tf.cast(x_plus, dtype="int32"), tf.cast(dim_x - 1, dtype="int32"))
    z_plus = tf.minimum(tf.cast(z_plus, dtype="int32"), tf.cast(dim_z - 1, dtype="int32"))
    y_minus = tf.maximum(y_minus, 0)
    x_minus = tf.maximum(x_minus, 0)
    z_minus = tf.maximum(z_minus, 0)
    d_phi_dx = tf.gather_nd(phi, tf.stack([z , y, x_plus], 1)) - tf.gather_nd(phi, tf.stack([z , y, x_minus], 1))
    d_phi_dx_2 = tf.square(d_phi_dx)
    d_phi_dy = tf.gather_nd(phi, tf.stack([z, y_plus, x], 1)) - tf.gather_nd(phi, tf.stack([z, y_minus, x], 1))
    d_phi_dy_2 = tf.square(d_phi_dy)
    d_phi_dz = tf.gather_nd(phi, tf.stack([z_plus, y, x], 1)) - tf.gather_nd(phi, tf.stack([z_minus, y, x], 1))
    d_phi_dz_2 = tf.square(d_phi_dz)
    d_phi_dxx = tf.gather_nd(phi, tf.stack([z, y, x_plus], 1)) + tf.gather_nd(phi, tf.stack([z, y, x_minus], 1)) - \
                2 * tf.gather_nd(phi, tf.stack([z, y, x], 1))
    d_phi_dyy = tf.gather_nd(phi, tf.stack([z, y_plus, x], 1)) + tf.gather_nd(phi, tf.stack([z, y_minus, x], 1)) - \
                2 * tf.gather_nd(phi, tf.stack([z, y, x], 1))
    d_phi_dzz = tf.gather_nd(phi, tf.stack([z_plus, y, x], 1)) + tf.gather_nd(phi, tf.stack([z_minus, y, x], 1)) - \
                2 * tf.gather_nd(phi, tf.stack([z, y, x], 1))
    d_phi_dxy = 0.25 * (- tf.gather_nd(phi, tf.stack([z, y_minus, x_minus], 1)) - tf.gather_nd(phi, tf.stack(
        [z, y_plus, x_plus], 1)) + tf.gather_nd(phi, tf.stack([z, y_minus, x_plus], 1)) + tf.gather_nd(phi, tf.stack(
        [z, y_plus, x_minus], 1)))
    d_phi_dxz = 0.25 * (- tf.gather_nd(phi, tf.stack([z_minus, y, x_minus], 1)) - tf.gather_nd(phi, tf.stack(
        [z_plus, y, x_plus], 1)) + tf.gather_nd(phi, tf.stack([z_minus, y, x_plus], 1)) + tf.gather_nd(phi, tf.stack(
        [z_plus, y, x_minus], 1)))
    d_phi_dyz = 0.25 * (- tf.gather_nd(phi, tf.stack([z_minus, y_minus, x], 1)) - tf.gather_nd(phi, tf.stack(
        [z_plus, y_plus, x], 1)) + tf.gather_nd(phi, tf.stack([z_plus, y_minus, x], 1)) + tf.gather_nd(phi, tf.stack(
        [z_minus, y_plus, x], 1)))
    d_phi_dxyz = 0.25 * (- tf.gather_nd(phi, tf.stack([z_minus, y_minus, x_minus], 1)) - tf.gather_nd(phi, tf.stack(
        [z_plus, y_plus, x_plus], 1)) + tf.gather_nd(phi, tf.stack([z_plus, y_minus, x_minus], 1)) + tf.gather_nd(phi, tf.stack(
        [z_minus, y_plus, x_minus], 1))+ tf.gather_nd(phi, tf.stack([z_plus, y_minus, x_plus], 1)) + tf.gather_nd(phi, tf.stack([z_minus, y_minus, x_plus], 1)))
    tmp_1 = tf.multiply(d_phi_dx_2, d_phi_dyy) + tf.multiply(d_phi_dx_2, d_phi_dzz) + tf.multiply(d_phi_dy_2, d_phi_dxx) + tf.multiply(d_phi_dy_2, d_phi_dzz) + tf.multiply(d_phi_dz_2, d_phi_dxx) + tf.multiply(d_phi_dz_2, d_phi_dyy) - \
            2 * tf.multiply(tf.multiply(tf.multiply(d_phi_dx, d_phi_dy), d_phi_dz), d_phi_dxyz)
    tmp_1 = 0
    tmp_2 = tf.add(tf.pow(d_phi_dx_2 + d_phi_dy_2 + d_phi_dz_2, 1.5), 2.220446049250313e-16)
    tmp_3 = tf.pow(d_phi_dx_2 + d_phi_dy_2 + d_phi_dz_2, 0.5)
    tmp_4 = tf.divide(tmp_1, tmp_2)
    curvature = tf.multiply(tmp_3, tmp_4)
    mean_grad = tf.pow(d_phi_dx_2 + d_phi_dy_2, 0.5)

    return curvature, mean_grad


def get_intensity(image, masked_phi, filter_patch_size=5, filter_depth_size=1):
    u_1 = tf.layers.average_pooling2d(tf.multiply(image, masked_phi), [filter_patch_size, filter_patch_size, filter_depth_size], 1,padding='SAME')
    u_2 = tf.layers.average_pooling2d(masked_phi, [filter_patch_size, filter_patch_size, filter_depth_size], 1, padding='SAME')
    u_2_prime = 1 - tf.cast((u_2 > 0), dtype='float32') + tf.cast((u_2 < 0), dtype='float32')
    u_2 = u_2 + u_2_prime + 2.220446049250313e-16

    return tf.divide(u_1, u_2)


def active_contour_layer(elems):
    img = elems[0]
    init_phi = elems[1]
    map_lambda1_acl = elems[2]
    map_lambda2_acl = elems[3]
    wind_coef = 3
    zero_tensor = tf.constant(0, shape=[], dtype="int32")
    def _body(i, phi_level):
        band_index = tf.reduce_all([phi_level <= narrow_band_width, phi_level >= -narrow_band_width], axis=0)
        band = tf.where(band_index)
        band_z = band[:, 0]
        band_y = band[:, 1]
        band_x = band[:, 2]
        shape_y = tf.shape(band_y)
        shape_z = tf.shape(band_z)
        num_band_pixel = shape_y[0]
        num_band_pixel_z = shape_z[0]
        window_radii_x = tf.ones(num_band_pixel) * wind_coef
        window_radii_y = tf.ones(num_band_pixel) * wind_coef
        window_radii_z = tf.ones(num_band_pixel) * wind_coef

        def body_intensity(j, mean_intensities_outer, mean_intensities_inner):
            xnew = tf.cast(band_x[j], dtype="float32")
            ynew = tf.cast(band_y[j], dtype="float32")
            znew = tf.cast(band_z[j], dtype="float32")
            window_radius_x = tf.cast(window_radii_x[j], dtype="float32")
            window_radius_y = tf.cast(window_radii_y[j], dtype="float32")
            window_radius_z = tf.cast(window_radii_z[j], dtype="float32")
            local_window_x_min = tf.cast(tf.floor(xnew - window_radius_x), dtype="int32")
            local_window_x_max = tf.cast(tf.floor(xnew + window_radius_x), dtype="int32")
            local_window_y_min = tf.cast(tf.floor(ynew - window_radius_y), dtype="int32")
            local_window_y_max = tf.cast(tf.floor(ynew + window_radius_y), dtype="int32")
            local_window_z_min = tf.cast(tf.floor(znew - window_radius_z), dtype="int32")
            local_window_z_max = tf.cast(tf.floor(znew + window_radius_z), dtype="int32")
            local_window_x_min = tf.maximum(zero_tensor, local_window_x_min)
            local_window_y_min = tf.maximum(zero_tensor, local_window_y_min)
            local_window_z_min = tf.maximum(zero_tensor, local_window_z_min)
            local_window_x_max = tf.minimum(tf.cast(input_image_size - 1, dtype="int32"), local_window_x_max)
            local_window_y_max = tf.minimum(tf.cast(input_image_size - 1, dtype="int32"), local_window_y_max)
            local_window_z_max = tf.minimum(tf.cast(input_image_z - 1, dtype="int32"), local_window_z_max)
            local_image = img[local_window_y_min: local_window_y_max + 1,local_window_x_min: local_window_x_max + 1, local_window_z_min: local_window_z_max + 1]
            local_phi = phi_level[local_window_y_min: local_window_y_max + 1,local_window_x_min: local_window_x_max + 1, local_window_z_min: local_window_z_max + 1]
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

        if fast_lookup:
            phi_4d = phi_level[tf.newaxis, :, :, :, tf.newaxis]
            image = img[tf.newaxis, :, :, :, tf.newaxis]
            band_index_2 = tf.reduce_all([phi_4d <= narrow_band_width, phi_4d >= -narrow_band_width], axis=0)
            band_2 = tf.where(band_index_2)
            u_inner = get_intensity(image, tf.cast((([phi_4d <= 0])), dtype='float32')[0], filter_patch_size=f_size)
            u_outer = get_intensity(image, tf.cast((([phi_4d > 0])), dtype='float32')[0], filter_patch_size=f_size)
            mean_intensities_inner = tf.gather_nd(u_inner, band_2)
            mean_intensities_outer = tf.gather_nd(u_outer, band_2)

        else:
            mean_intensities_inner = tf.constant([0], dtype='float32')
            mean_intensities_outer = tf.constant([0], dtype='float32')
            j = tf.constant(0, dtype=tf.int32)
            _, mean_intensities_outer, mean_intensities_inner = tf.while_loop(
                lambda j, mean_intensities_outer, mean_intensities_inner:
                j < num_band_pixel, body_intensity, loop_vars=[j, mean_intensities_outer, mean_intensities_inner],
                shape_invariants=[j.get_shape(), tf.TensorShape([None]), tf.TensorShape([None])])

        lambda1 = tf.gather_nd(map_lambda1_acl, [band])
        lambda2 = tf.gather_nd(map_lambda2_acl, [band])
        curvature, mean_grad = get_curvature(phi_level, band_x, band_y, band_z)
        kappa = tf.multiply(curvature, mean_grad)
        term1 = tf.multiply(tf.cast(lambda1, dtype='float32'),tf.square(tf.gather_nd(img, [band]) - mean_intensities_inner))
        term2 = tf.multiply(tf.cast(lambda2, dtype='float32'),tf.square(tf.gather_nd(img, [band]) - mean_intensities_outer))
        force = -nu + term1 - term2
        force /= (tf.reduce_max(tf.abs(force)))
        d_phi_dt = tf.cast(force, dtype="float32") + tf.cast(mu * kappa, dtype="float32")
        dt = .45 / (tf.reduce_max(tf.abs(d_phi_dt)) + 2.220446049250313e-16)
        d_phi = dt * d_phi_dt
        update_narrow_band = d_phi
        phi_level = phi_level + tf.scatter_nd([band], tf.cast(update_narrow_band, dtype='float32'),shape=[input_image_size, input_image_size, input_image_z])
        phi_level = re_init_phi(phi_level, 0.5)

        return (i + 1, phi_level)

    i = tf.constant(0, dtype=tf.int32)
    phi = init_phi
    _, phi = tf.while_loop(lambda i, phi: i < iter_limit, _body, loop_vars=[i, phi])
    phi = tf.round(tf.cast((1 - tf.nn.sigmoid(phi)), dtype=tf.float32))

    return phi,init_phi, map_lambda1_acl, map_lambda2_acl

# =======================