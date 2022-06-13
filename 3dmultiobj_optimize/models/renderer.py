import numpy as np
import math
import tensorflow as tf
from utils.shared_funcs import *
from utils.tf_funcs import gaussian_blur


def get_rotation_mat(alphas):
    """

    :param alphas:  (BS, )
    :return:        rotation matrix, (BS, 3, 3)
    """

    alphas = math.pi * alphas
    mats = tf.stack([tf.stack([tf.cos(alphas), -tf.sin(alphas), tf.zeros_like(alphas)], axis=1),
                     tf.stack([tf.sin(alphas), tf.cos(alphas), tf.zeros_like(alphas)], axis=1),
                     tf.stack([tf.zeros_like(alphas), tf.zeros_like(alphas), tf.ones_like(alphas)], axis=1)],
                    axis=1)

    return mats


class DiffRenderer(tf.keras.Model):

    def __init__(self, deepsdf, cnfg, img_size_new=None):
        """
        Differentiable Renderer.
        :param cnfg:    renderer configuration dictionary
        """
        super(DiffRenderer, self).__init__()

        if img_size_new is not None:
            self.img_size = img_size_new
        else:
            self.img_size = cnfg['img_size']
        self.img_size_org = cnfg['img_size']
        self.d_steps = cnfg['d_steps']
        self.d_max = cnfg['d_max']

        K = [[70.*(self.img_size/64), 0., self.img_size/2.-0.5],
             [0., 70*(self.img_size/64), self.img_size/2.-0.5],
             [0., 0., 1.]]
        self.K_inv = tf.convert_to_tensor(np.linalg.inv(K), dtype=tf.float32, name='K_inv')    # (3, 3)
        self.R_t = tf.convert_to_tensor(np.transpose(cnfg['R']), dtype=tf.float32, name='R_t')         # (3, 3)
        self.t = tf.convert_to_tensor(np.expand_dims(cnfg['t'], -1), dtype=tf.float32, name='t')       # (3, 1)

        self._deepsdf = deepsdf
        self.n_pnts_block = self._deepsdf.n_pnts

    def build(self, input_shapes):
        pass

    def call(self, input):
        """
        input (dict):
            - [0] 'z_shape':    (BS, D_sh)
            - [1] 'z_extr':     (BS, D_extr=5)
        """

        z_shape = input[0]
        z_extr = input[1]

        res = self.render(z_shape, z_extr=z_extr)

        return res

    def set_new_camera_pose(self, t, rot):

        R = get_rotation_matrix(rot)
        R, t = get_blender_cam(R, t)

        self.R_t = tf.convert_to_tensor(np.transpose(R), dtype=tf.float32, name='R_t')  # (3, 3)
        self.t = tf.convert_to_tensor(np.expand_dims(t, -1), dtype=tf.float32, name='t')  # (3, 1)

    @staticmethod
    def get_img_coords(l):
        """
        Generate standard 2D coordinates + depth values.
        :param      l:              [(start, stop, num) for _ in [X, Y, Z]]  -> lin_space_list
        :return:    img_coords,     (1, H*W*D_steps, 2)
                    Z,              (1, H, W, D_steps)
        """
        X, Y, Z = tf.meshgrid(tf.linspace(l[0][0], l[0][1], l[0][2]),
                              tf.linspace(l[1][0], l[1][1], l[1][2]),
                              tf.linspace(l[2][0], l[2][1], l[2][2]))   # each: (H, W, D_steps)
        n_pnts = l[0][2] * l[1][2] * l[2][2]
        X = tf.reshape(X, [n_pnts, 1])                                  # (H*W*D_steps, 1)
        Y = tf.reshape(Y, [n_pnts, 1])
        img_coords = tf.concat([X, Y], axis=-1)
        img_coords = tf.expand_dims(img_coords, axis=0)
        Z = tf.expand_dims(Z, axis=0)

        return img_coords, Z

    @staticmethod
    def get_obj_trans(z_ext):
        """
        Get world -> object transformation matrices(from [scale, t_x, t_y, t_z, alpha_z]).
        :param z_ext:   (BS'=BS*N_img, 5) [scale, t_x, t_y, t_z, alpha_z]
        :return:        s_obj, t_obj, R_obj
        """
        s_obj = (1. / z_ext[:, 0])
        t_obj = tf.expand_dims(z_ext[:, 1:4], axis=-1)
        R_obj = get_rotation_mat(z_ext[:, 4])

        s_obj = tf.reshape(tf.convert_to_tensor(s_obj, dtype=tf.float32), (-1, 1, 1, 1), name='s_obj')  # (BS, 1, 1, 1)
        t_obj = tf.expand_dims(tf.convert_to_tensor(t_obj, dtype=tf.float32, name='t_obj'), axis=1)     # (BS, 1, 3, 1)
        R_obj = tf.convert_to_tensor(R_obj, dtype=tf.float32, name='R_obj')                             # (BS, 3, 3)

        return s_obj, t_obj, R_obj

    @staticmethod
    def pred_depth_from_samplegrid(sdfvalues, Z_coords):
        """
        Obtain zero-intersection from sample-grid with sdf values.
        :param sdfvalues:   (BS, H*W, D_steps, 1) or (BS, H, W, D_steps, 1)
        :param Z_coords:    (BS, H*W, D_steps) or (BS, H, W, D_steps)
        :return:            depth pred, occ:  (BS, H*W, 1) or (BS, H, W, 1)
        """

        # Find zero-crossing
        sdfvalues_out = tf.math.greater(sdfvalues, 0.)
        pnts_out = tf.where(sdfvalues_out,
                            tf.ones_like(sdfvalues_out, dtype=tf.float16),
                            -1.*tf.ones_like(sdfvalues_out, dtype=tf.float16))

        pnts_out1 = pnts_out[..., :-1, :]
        pnts_out2 = pnts_out[..., 1:, :]

        pnts_zero_cross = (pnts_out1-pnts_out2)
        pnts_zero_cross = tf.cast(tf.equal(pnts_zero_cross, 2), tf.float32)
        occ = tf.reduce_max(pnts_zero_cross, axis=-2)

        def helper(zero_cross_list):
            zero_cross = tf.concat(zero_cross_list, axis=-2)  # (BS, H, W, N_steps, 1)
            tmp = 100*(tf.ones_like(zero_cross)-zero_cross)
            depth = tf.reduce_min((tf.expand_dims(Z_coords, axis=-1) * zero_cross) + tmp, axis=-2)
            sdf = tf.reduce_sum(sdfvalues * zero_cross, axis=-2)
            return sdf, depth

        sdfvalues_1, depth_1 = helper([pnts_zero_cross, tf.zeros_like(pnts_zero_cross)[..., 0:1, :]])
        sdfvalues_2, depth_2 = helper([tf.zeros_like(pnts_zero_cross)[..., 0:1, :], pnts_zero_cross])

        eps = 0.000001
        depth_pred = occ*(depth_1 - (sdfvalues_1 / (sdfvalues_2 - sdfvalues_1 - eps)) * (depth_2 - depth_1))    # (BS, H*W, 1)

        return depth_pred, occ

    def trans_pnts(self, coords_3d, depth, obj_mats):
        """
        3D coords (transformation)
            -(image coordinates with respective depth)
        :param coords_3d:   (BS, N_pnts = H*W*D_steps, 3)
        :param depth:       (BS, N_pnts = H*W*D_steps, 1)
        :param obj_mats:    [s_obj, t_obj, R_obj]
        :return:            (BS, H*W*D_steps, 3)
        """

        #   - world coordinates (r_u,v)
        X = tf.expand_dims(coords_3d, -1)
        X_shape = tf.shape(X)
        X = tf.einsum('ij,ajk->aik', self.K_inv, tf.reshape(X, (-1, 3, 1)))
        X = tf.einsum('ij,ajk->aik', self.R_t, tf.reshape(depth, (-1, 1, 1)) * X - self.t)
        X = tf.reshape(X, X_shape)

        #   - object coordinates
        s_obj, t_obj, R_obj = obj_mats
        X = s_obj * tf.einsum('bij,bajk->baik', R_obj, (X - t_obj))

        X = tf.squeeze(X, axis=-1)                                              # (BS, H*W*D_steps, 3)

        return X

    def get_sdf_values(self, X, z_shape):
        """
        Compute sdf values for a set of points X based on latent z_shape.
        :param X:           (BS, N_pnts, 3)
        :param z_shape:     (BS, D_shape)
        :return:            sdf values, (BS, N_pnts, 1)
        """

        z_shape = tf.tile(tf.expand_dims(z_shape, axis=1), (1, X.shape[1], 1))
        sdfvalues = self._deepsdf([X], latent=z_shape)['sdf']

        return sdfvalues

    def call(self, z_shape, z_extr=None):
        """
        Render object based on shape and extinsic latent.
        Only points within a bounding box are considered.
        Returns 2D depth and occlusion map as well as 3D coordinates of observable surface.
        :param z_shape:     (BS'=BS*N_img, D)
        :param z_extr:      (BS'=BS*N_img, 5) [scale, t_x, t_y, t_z, alpha_z]
        :return:
        """
        BS = z_shape.shape[0]

        # --------------------------------
        # Object Transformations

        if z_extr is None:
            print('Renderer, no extrinsic coordinates')
            z_extr = tf.concat([tf.ones((BS, 1)), tf.zeros((BS, 3)), tf.zeros((BS, 1))], axis=1)
        s_obj, t_obj, R_obj = self.get_obj_trans(z_extr)

        # --------------------------------
        # Object Bounding Box

        X_bb = tf.constant([[[1., 1., 1.], [1., 1., -1.], [1., -1., 1.], [1., -1., -1.],
                             [-1., 1., 1.], [-1., 1., -1.], [-1., -1., 1.], [-1., -1., -1.],
                             [0., 0., 0.]]])
        X_bb = tf.expand_dims(tf.tile(X_bb, (BS, 1, 1)), -1)                                # (BS, 8+1, 3, 1)
        s_obj_inv = tf.ones_like(s_obj) / s_obj
        R_obj_inv = tf.linalg.inv(R_obj)
        R_cam_inv = tf.linalg.inv(self.R_t)
        K = tf.linalg.inv(self.K_inv)

        X_bb = tf.einsum('bij,bajk->baik', R_obj_inv, (s_obj_inv * X_bb))
        X_bb = X_bb + t_obj
        X_bb = tf.einsum('ij,bajk->baik', R_cam_inv, X_bb) + self.t
        X_bb = tf.einsum('ij,bajk->baik', K, X_bb)
        X_bb_depth = X_bb[:, :, 2:3]
        X_bb = X_bb / X_bb_depth
        X_bb_depth = tf.reduce_mean(X_bb_depth, axis=1, keepdims=True)

        cntr_coord = tf.squeeze(X_bb, axis=-1)[:, 8, 0:2]
        X_bb = tf.cast(tf.squeeze(X_bb, axis=-1)[:, :8, 0:2], tf.int32)
        X_bb = tf.clip_by_value(X_bb, 0, self.img_size)
        X_bb_up = tf.reduce_max(X_bb + tf.ones_like(X_bb), axis=1, keepdims=True)    # (BS, 1, 2)
        X_bb_up = tf.math.maximum(tf.math.minimum(X_bb_up, self.img_size*tf.ones_like(X_bb_up)), tf.zeros_like(X_bb_up))
        X_bb_low = tf.reduce_min(X_bb, axis=1, keepdims=True)
        X_bb_low = tf.math.minimum(tf.math.maximum(X_bb_low, tf.zeros_like(X_bb_low)), self.img_size*tf.ones_like(X_bb_up))

        X_bb_max_range = tf.reduce_max(X_bb_up - X_bb_low, axis=0, keepdims=True)  # (1, 1, 2)
        X_bb_max_range = tf.clip_by_value(X_bb_max_range, 0, self.img_size)

        off = X_bb_low + X_bb_max_range - self.img_size
        off = tf.clip_by_value(off, 0, self.img_size)
        X_bb_low = X_bb_low - off

        # --------------------------------
        # Generate 3D point coordinates w.r.t. object coordinate system
        lin_space_list = [(0., tf.cast(X_bb_max_range[0, 0, 0]-1, tf.float32), X_bb_max_range[0, 0, 0]),
                          (0., tf.cast(X_bb_max_range[0, 0, 1]-1, tf.float32), X_bb_max_range[0, 0, 1]),
                          (-self.d_max, self.d_max, self.d_steps)]
        coords_2d, Z = self.get_img_coords(lin_space_list)
        coords_2d = tf.tile(coords_2d, (BS, 1, 1))                          # (BS, H*W*D_steps, 2)
        coords_2d_new_x = coords_2d[:, :, 0] + tf.cast(X_bb_low[:, :, 0], tf.float32)
        coords_2d_new_y = coords_2d[:, :, 1] + tf.cast(X_bb_low[:, :, 1], tf.float32)
        coords_2d = tf.stack([coords_2d_new_x, coords_2d_new_y], axis=-1)
        Z = tf.tile(Z, (BS, 1, 1, 1))
        n_pnts = tf.reduce_prod(X_bb_max_range, axis=2)[0, 0]
        Z = tf.reshape(Z, (BS, n_pnts, self.d_steps, 1))
        Z = Z * s_obj_inv + X_bb_depth
        n_pnts = tf.reduce_prod(X_bb_max_range, axis=2)[0, 0]
        depth = tf.reshape(Z, (BS, n_pnts * self.d_steps, 1))                                  # (BS, H'*W'*D_steps, 1)

        coords_3d = tf.concat([coords_2d, tf.ones_like(depth)], axis=-1)
        X = self.trans_pnts(coords_3d, depth, [s_obj, t_obj, R_obj])            # (BS, H'*W'*D_steps, 3)

        # --------------------------------
        # Infer occlusion map from sdf

        # - extract sdf values
        sdfvalues = self.get_sdf_values(X, z_shape)
        sdfvalues = tf.reshape(sdfvalues, (BS, n_pnts, self.d_steps, 1))

        # - occlusion prediction (-> Direct Shape)
        sigma = 50.     # higher value yields sharper
        sdfvalues_exp = tf.exp(sigma*sdfvalues)
        mask_bb = 1. - tf.reduce_prod(sdfvalues_exp/(sdfvalues_exp+1.), axis=-2)            # (BS, N_bb, 1)

        # --------------------------------
        # insert bounding box (mask) content into entire image

        indices_b = tf.tile(tf.expand_dims(tf.linspace(0., tf.cast(BS-1, tf.float32), BS), axis=-1), (1, n_pnts*self.d_steps))
        indices = tf.stack([indices_b, coords_2d[:, :, 1], coords_2d[:, :, 0]], axis=-1)
        indices = tf.reshape(tf.cast(indices, tf.int32), (-1, self.d_steps, 3))[:, 0]    # inds of bb

        mask_bb = tf.reshape(mask_bb, (BS*n_pnts, ))
        mask_soft = tf.scatter_nd(indices, mask_bb, [BS, self.img_size, self.img_size])    # bb msk into image

        # --------------------------------
        # Depth Optimization

        # # alternative min-outside depth
        # sdfvalues_out = tf.where(tf.math.greater(sdfvalues, 0.),
        #                          sdfvalues, 20.*tf.ones_like(sdfvalues, dtype=tf.float32))
        # depth_ind = tf.math.argmin(sdfvalues_out, axis=-2)    # (BS, n_pnts, 1)
        # lin_space_list = [(0., tf.cast(BS-1, tf.float32), BS), (0., tf.cast(n_pnts-1, tf.float32), n_pnts), (0., 0., 1)]
        # depth_ind_coords, _ = self.get_img_coords(lin_space_list)
        # depth_ind = tf.reshape(tf.cast(depth_ind, tf.int32), (BS*n_pnts, 1))
        # depth_ind = tf.concat([tf.cast(depth_ind_coords[0], tf.int32), depth_ind], axis=-1)
        # depth_pred_soft = tf.gather_nd(Z, depth_ind)
        # depth_pred_soft = tf.reshape(tf.cast(depth_pred_soft, dtype=tf.float32), (BS*n_pnts, ))
        #
        # depth_pred_soft = tf.scatter_nd(indices, depth_pred_soft, [BS, self.img_size, self.img_size])  # bb depth into image
        # depth_pred_soft = mask_soft * depth_pred_soft

        # zero-intersection depth
        depth_pred, occ = self.pred_depth_from_samplegrid(sdfvalues, tf.squeeze(Z, -1))
        depth_pred = tf.reshape(depth_pred, (BS*n_pnts, ))
        depth_pred = tf.scatter_nd(indices, depth_pred, [BS, self.img_size, self.img_size])
        occ = tf.reshape(occ, (BS*n_pnts, ))
        occ = tf.scatter_nd(indices, occ, [BS, self.img_size, self.img_size])

        # # -- smooth depth, no 2
        # kernelsize = 3*2+1  # 16*2+1
        # sigma = 1.  #0.5
        # eps = 0.00001
        # mask_gauss = gaussian_blur(tf.expand_dims(occ, axis=-1), kernel_size=kernelsize, sigma=sigma)
        # mask_gauss_inv = tf.divide(tf.ones_like(mask_gauss), mask_gauss+eps)
        # depth_gauss = gaussian_blur(tf.expand_dims(depth_pred, axis=-1), kernel_size=kernelsize, sigma=sigma)
        # depth_soft = (mask_gauss_inv*depth_gauss)[..., 0]

        # depth_pred = depth_soft  # TODO: usefull?
        # depth_pred = depth_pred + (tf.ones_like(occ) - occ) * depth_pred_soft

        # --------------------------------
        # X_silhouette

        lin_space_list = [(0., self.img_size, self.img_size), (0., self.img_size, self.img_size), (0., 0., 1)]
        coords_2d, _ = self.get_img_coords(lin_space_list)
        coords_2d = tf.tile(coords_2d, (BS, 1, 1))
        d = tf.reshape(depth_pred, (BS, self.img_size * self.img_size, 1))

        coords_3d = tf.concat([coords_2d, tf.ones_like(d)], axis=2)
        X_silhouette = self.trans_pnts(coords_3d, d, [s_obj, t_obj, R_obj])

        return {'depth': depth_pred,                        # (BS', H, W, 1)
                'points_3d_silhouette': X_silhouette,       # (BS', H*W, 3)
                'occlusion': occ,                           # (BS', H, W)
                'occlusion_soft': mask_soft,                # (BS', H, W)
                'cntr_coord': cntr_coord                    # (BS', 2)
                }
