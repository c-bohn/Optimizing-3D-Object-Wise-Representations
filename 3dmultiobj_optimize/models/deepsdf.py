import tensorflow as tf
from tensorflow.keras import layers, losses
import tensorflow.keras.backend as K


class DeepSDF(tf.keras.Model):

    def __init__(self, cnfg, name="deep_sdf"):
        """
        DeepSDF-VAE Model.
        :param cnfg:    model configuration dictionary
        """
        super(DeepSDF, self).__init__(name=name)

        self.n_latents = cnfg['n_latents']
        self.dim_latent = cnfg['dim_latent']
        self.dim_out = cnfg['dim_out']
        self.n_pnts = cnfg['n_off_surface_samples'] + cnfg['n_on_surface_samples']

        # Decoder
        self.layers_fc = []
        for i, filters in enumerate(cnfg['dec_fc_layer']):
            self.layers_fc.append(layers.Conv1D(filters, kernel_size=(1,), strides=1, activation='relu'))
        self.layer_out = layers.Conv1D(self.dim_out, kernel_size=(1,), strides=1, activation='tanh')

        self.latents = tf.Variable(initial_value=tf.random.truncated_normal([self.n_latents,
                                                                             self.dim_latent],
                                                                            mean=0.0, stddev=0.01), name='latent')

    def get_input(self, batch_data):
        scene_ids = batch_data[0]
        rgb_gt = batch_data[1]
        pnts_coord = batch_data[2][:, :, :3]
        pnts_sdf = batch_data[2][:, :, 3:]
        slice_gt = batch_data[3]

        return {
            'scene_ids': scene_ids,         # (1)
            'rgb_gt': rgb_gt,               # (N, W, H, 3)
            'pnts_coord': pnts_coord,       # (N_pnts, 3)
            'pnts_sdf': pnts_sdf,           # (N_pnts, 1)
            'slice_gt': slice_gt,           # (W, H, 3),
        }

    def get_latent(self, obj_id):
        """
        Indexes latents wrt current scene.
        :param obj_id:      (BS, 1)
        :return:            (BS, 1, D_sh)
        """
        latent = K.gather(self.latents, obj_id)

        return latent

    def call(self, inputs, latent=None):

        input_coords = inputs[0]        # (BS, N_pnts, 3)

        if latent is None:              # (BS, N_pnts, D)
            input_obj_id = inputs[1]    # (BS, 1)
            latent = self.get_latent(input_obj_id)
            latent = layers.Lambda(K.tile, arguments={'n': (1, self.n_pnts, 1)}, name='tile')(latent)
        net = layers.Concatenate(axis=-1)([latent, input_coords])

        for i in range(len(self.layers_fc)):
            net = self.layers_fc[i](net)

        sdf = self.layer_out(net)

        return {
            'sdf': sdf,
            'z_shape_mean': latent
        }

    def get_loss(self, output_data, gt_data, params):
        """

        :param pred_sdf:    (BS, N_pnts, D)
        :param gt_sdf:      (BS, N_pnts, D)
        :param params:      {'w_s-reg', 'clamp_dist': ()}
        :return:            losses, {'loss_total', 'l_likelihood', 'l_s-reg': ()}
        """

        pred_sdf = output_data['sdf']
        gt_sdf = gt_data['pnts_sdf']

        with tf.name_scope('loss_likelihood'):
            # Likelihood loss: E[log P(X|z)]
            # - L1
            l_sdf = K.mean(losses.mean_absolute_error(gt_sdf, pred_sdf))
            # - clamped L1 (DeepSDF)
            clamp_dist = params['clamp_dist']
            clamped_input = tf.clip_by_value(gt_sdf, -clamp_dist, clamp_dist)
            clamped_out = tf.clip_by_value(pred_sdf, -clamp_dist, clamp_dist)
            l_sdf_clamp = K.mean(losses.mean_absolute_error(clamped_out, clamped_input))

        with tf.name_scope('loss_s-reg'):
            # - DeepSDF, 1/sigma^2 * ||z||^2_2
            # ('zero-mean multivariate Gaussian with a spherical covariance sigma^2I')
            l_s_reg = tf.reduce_mean(tf.reduce_sum(K.square(self.latents), axis=-1))

        loss = l_sdf + params['w_z-reg'] * l_s_reg

        return {
            'loss_total': loss,
            'l_sdf': l_sdf,
            'l_sdf-clamp': l_sdf_clamp,
            'l_s-reg': params['w_z-reg'] * l_s_reg,
        }


# ---------------------------------------------------------------------------------------------------------------------


def get_model(cnfg_model, ext):
    print('Load Model: DeepSDF-'+ext)
    if ext == 'org':
        model = DeepSDF(cnfg_model)
    else:
        print('MOSNet, Model Variation {} not defined.'.format(ext))
        exit(1)

    return model
