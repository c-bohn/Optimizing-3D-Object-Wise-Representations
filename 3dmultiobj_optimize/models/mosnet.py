import tensorflow as tf
from tensorflow.keras import layers, losses, activations
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import numpy as np
import copy

from math import pi
from termcolor import colored
from itertools import permutations

import models.deepsdf as deepsdf
import models.renderer as renderer
from utils.tf_funcs import gaussian_blur, create_normal_img

EXTR_DIM = 5


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


class EncoderNet(tf.keras.Model):

    def __init__(self, dim_in, dim_out, conv_filters, mlp_units, var_mode=False):

        super(EncoderNet, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.conv_filters = conv_filters
        self.mlp_units = mlp_units
        self.var_mode = var_mode

        # Network weights/ biases
        self.conv_layers = []
        self.maxpool_layers = []
        for i, f in enumerate(self.conv_filters):
            self.conv_layers.append(layers.Conv2D(f, kernel_size=(3, 3), strides=(1, 1), activation='relu',
                                                  name='conv_' + str(i)))
            self.maxpool_layers.append(layers.MaxPooling2D(pool_size=(2, 2), name='max_pool_' + str(i)))
        self.fc_layers = []
        for i, h in enumerate(self.mlp_units):
            self.fc_layers.append(layers.Dense(h, activation='relu', name='dense_' + str(i)))

        # TODO: Batch normalization, Drop Out

        self.layer_out_mean = layers.Dense(self.dim_out, name='z_mean')
        if self.var_mode:
            self.layer_out_var = layers.Dense(self.dim_out, name='z_var')

    def call(self, inputs, training=False):

        net = inputs[0]  # (BS, N_img, W, H, C)

        shape_in = net.shape
        shape_out = shape_in[:-3] + [self.dim_out]
        net = tf.reshape(net, (-1, shape_in[-3], shape_in[-2], shape_in[-1]))

        for i in range(len(self.conv_layers)):
            net = self.conv_layers[i](net)
            net = self.maxpool_layers[i](net)
            # TODO: BatchNormalization? Dropout?

        net = layers.Flatten()(net)
        for i in range(len(self.fc_layers)):
            net = self.fc_layers[i](net)

        z_mean = self.layer_out_mean(net)
        z_mean = tf.reshape(z_mean, shape_out)

        if self.var_mode:
            z_var = self.layer_out_var(net)
            z_var = tf.reshape(z_var, shape_out)

            return {
                'z_mean': z_mean,
                'z_var': z_var
            }
        else:
            return {
                'z_mean': z_mean,
            }


class RefinerNet(EncoderNet):

    def __init__(self, dim_in, dim_out, conv_filters, mlp_units, var_mode=False):
        #Except for the different number of input channels dim_in and the latents computed by an EncoderNet, this is almost the same as the EncoderNet

        super(RefinerNet, self).__init__(dim_in, dim_out, conv_filters, mlp_units, var_mode)
        self.latents_combination_layer = layers.Dense(self.dim_out, name='latents_combination')

    def call(self, inputs, latents_in, training=False):

        net = inputs[0]  # (BS, N_img, W, H, C)

        shape_in = net.shape
        shape_out = shape_in[:-3] + [self.dim_out]
        net = tf.reshape(net, (-1, shape_in[-3], shape_in[-2], shape_in[-1]))

        latents_in = tf.reshape(latents_in, [shape_out[0], shape_out[-1]])

        for i in range(len(self.conv_layers)):
            net = self.conv_layers[i](net)
            net = self.maxpool_layers[i](net)
            # TODO: BatchNormalization? Dropout?

        net = layers.Flatten()(net)
        for i in range(len(self.fc_layers)):
            net = self.fc_layers[i](net)

        z_mean = self.layer_out_mean(net)   #shape=[BS, dim_latent]

        # Latents combination layer: Here the model decides which parts of the latents to take from the encoder and which from the refiner
        #TODO This should probably be fixed to the refiner's output during training, and only then be combined with the shortcut
        # connection from the encoder; This is to force the refiner to learn its task and not to simply copy the encoder output
        #z_mean = self.latents_combination_layer(tf.concat([z_mean, latents_in], axis=-1))
        #z_mean = latents_in

        z_mean = tf.reshape(z_mean, shape_out) #shape=[BS, 1, dim_latent]

        if self.var_mode:
            z_var = self.layer_out_var(net)
            z_var = tf.reshape(z_var, shape_out)

            return {
                'z_mean': z_mean,
                'z_var': z_var
            }
        else:
            return {
                'z_mean': z_mean,
            }



class ConvNet(tf.keras.Model):

    def __init__(self, dim_in, dim_out, conv_filters):

        super(ConvNet, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.conv_filters = conv_filters

        # Network weights/ biases
        self.conv_layers = []
        for i, f in enumerate(self.conv_filters):
            self.conv_layers.append(layers.Conv2D(f, kernel_size=(3, 3), strides=(1, 1),
                                                  activation='relu', name='conv_' + str(i)))
        self.conv_layers.append(layers.Conv2D(self.dim_out, kernel_size=(3, 3), strides=(1, 1), name='conv_out'))

    def call(self, inputs, training=False):

        net = inputs  # (BS, N_img, W, H, C)

        shape_in = net.shape
        shape_out = shape_in[:-1] + [self.dim_out]
        net = tf.reshape(net, (-1, shape_in[-3], shape_in[-2], shape_in[-1]))
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

        for i in range(len(self.conv_layers)):
            net = tf.pad(net, paddings, "SYMMETRIC")
            net = self.conv_layers[i](net)

        net = tf.reshape(net, shape_out)

        return {
            'output': net
        }


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


class MultiObj3DNet(tf.keras.Model):

    # ----------------------------------------------------------------------
    # --- Initialization

    def __init__(self, cnfg):
        """
        MOSNet Model.
        :param cnfg:    overall configuration dictionary
        """
        super(MultiObj3DNet, self).__init__()

        self.n_slots = cnfg['model']['mosnet']['n_slots']
        self.n_imgs = cnfg['model']['mosnet']['n_imgs']
        self.img_size = cnfg['model']['mosnet']['img_size']
        if cnfg['model']['mosnet']['anti_aliasing']:
            print("Anti-aliasing = True")
            self.img_size_render = self.img_size*2
        else:
            print("Anti-aliasing = False")
            self.img_size_render = self.img_size
        dim_latent_split = cnfg['model']['mosnet']['dim_latent_split']
        self.dim_shape = dim_latent_split[0]
        self.dim_app = dim_latent_split[1]
        self.dim_extr = dim_latent_split[2]

        self.dim_feat = cnfg['model']['deepcolor']['dim_out']  # for basic model, this is 3 (RGB value)
        self.dim_enc_obj_in = 7
        self.dim_latent_obj = self.dim_shape + self.dim_app + self.dim_extr
        self.dim_enc_bg_in = 3
        self.dim_latent_bg = cnfg['model']['deepcolor']['dim_out']

        self.size_range = cnfg['model']['mosnet']['size_range']
        self.gauss_kernel = cnfg['model']['mosnet']['gauss_kernel']

        self.depth_max = cnfg['renderer']['depth_max']

        self.output_types = ['obj_rgb_pred', 'obj_depth_pred', 'obj_msk_pred', 'obj_cntr', 'z_extr']

        if 'contrast_factor' in cnfg['model']['mosnet']:
            self.contrast_factor = cnfg['model']['mosnet']['contrast_factor']
        else:
            cnfg['model']['mosnet']['contrast_factor'] = 1.
            self.contrast_factor = 1.

        self._bg = {}
        with tf.name_scope("enc_bg"):
            self.encoder_bg = EncoderNet(dim_in=self.dim_enc_bg_in, dim_out=self.dim_latent_bg,
                                         conv_filters=[32, 32], mlp_units=[16])

        with tf.name_scope("enc_objects"):
            self.enc_convs = cnfg['model']['mosnet']['enc_convs']
            self.enc_fc = cnfg['model']['mosnet']['enc_fc']
            self.encoder_obj = EncoderNet(dim_in=self.dim_enc_obj_in, dim_out=self.dim_latent_obj,
                                          conv_filters=self.enc_convs, mlp_units=self.enc_fc)

        with tf.name_scope("dec_sdf"):
            self.decoder_sdf = deepsdf.DeepSDF(cnfg['model']['deepsdf'], name='dec_sdf')
            self.decoder_sdf.trainable = False

        with tf.name_scope("dec_rgb"):
            self.decoder_rgb = deepsdf.DeepSDF(cnfg['model']['deepcolor'], name='dec_rgb')
            self.decoder_rgb.latents = tf.stop_gradient(self.decoder_rgb.latents)

        with tf.name_scope("renderer"):
            self.renderer = renderer.DiffRenderer(self.decoder_sdf, cnfg['renderer'], img_size_new=self.img_size_render)

        self.silhouette_mask_normal_thresh = 3. #set to 5. for the silhouette only, set to 3. to also get the edges of cubes
        self.silhouette_mask_depth_thresh = 5.

    def get_input(self, batch_data):
        """
        Extract list of tensors to correct input format and return dict
        :param batch_data:  list from input tensors from data generator
        :return:            dict of input tensors
        """

        scene_ids = batch_data[0]
        rgb_in = batch_data[1]
        rgb_gt = batch_data[2]

        depth_gt = batch_data[3]
        mask_gt = batch_data[4]

        z_extr_gt = batch_data[5]

        return {
            'scene_ids': scene_ids,  # (BS, 1)
            'rgb_in': rgb_in,  # (BS, N, W, H, 3)
            'rgb_gt': rgb_gt,  # (BS, N, W, H, 3)
            'depth_gt': depth_gt,  # (BS, N, H, W)
            'msk_gt': mask_gt,  # (BS, S, N, H, W)
            'z_extr_gt': z_extr_gt,  # (BS, S, N, D_extr)
        }

    def set_gt_bg(self, img=None, depth=None):

        if img is not None:
            self._bg['img'] = img

        if depth is not None:
            self._bg['depth'] = depth

    def call(self, inputs, training=False, manipulation=None):

        rgb_in = inputs[0]  # (BS, N, W, H, 3)

        # TODO: normalize

        # res_obj = self.predict_objs(input_imgs, inputs['gt_z_extr'], inputs['scene_ids'])  # TODO: extr as input
        res_obj = self.predict_objs(rgb_in, training=training)
        bg_img = self.predict_bg(rgb_in)

        if manipulation is not None:
            print(colored('[TODO] MultiObj3DNet.call, manipulation of objects', 'yellow'))
        #     z_obj_all = res_obj['z_obj_all']
        #     z_obj_all = self.manipulate_obj(z_obj_all, manipulate)  # manipulate objects (for evaluation)
        #     res_obj_manipulated = self.render_objs(z_obj_all)
        #     for k, v in res_obj_manipulated.items():
        #         res_obj[k] = v

        combined_res = self.depth_ordering(res_obj, bg_img)
        res = res_obj
        res['bg_img'] = bg_img
        for k, v in combined_res.items():
            res[k] = v

        res = self.crit_obj_intersection(res)
        res = self.crit_ground(res)

        # from utils import viz
        # rgb_pred = (np.transpose((res['rgb_pred'])[0, ...], (1, 0, 2, 3))).reshape((64, 64, 3))
        # viz.show_image(rgb_pred, 'test_rgb_pred.png')

        if self.img_size_render != self.img_size:
            res['rgb_pred_full_size'] = copy.deepcopy(res['rgb_pred'])
            res['depth_pred_full_size'] = copy.deepcopy(res['depth_pred'])
            rgb_pred = self.downsample_render(res['rgb_pred'], antialiasing=True)
            depth_pred = self.downsample_render(res['depth_pred'], antialiasing=False)
            res['rgb_pred'] = rgb_pred
            res['depth_pred'] = depth_pred

        return res

    # # ----------------------------------------------------------------------
    # # --- Scene Encoding

    # This wrapper is needed in the render-and-compare optimization to provide a common interface for MultiObj3DNet and MultiObj3DNet_w_refiner
    def gen_latents_single_slot(self, input, rgb_in, res, msk_list, diffrgb_list, training=False):
        return self.encoder_obj([input], training=training)

    def predict_objs(self, rgb_in, training, gt_extr=None, scene_ids=None):
        """

        :param rgb_in:      (BS, N_imgs, H, W, 3)
        :return:            {'img_rec': (BS, N_img, H, W, 3),
                             'depth': (BS, N_img, H, W, 1),
                             'obj_rec': (BS, N_slots, N_img, H, W, 3),
                             'occlusion', 'occlusion_soft': (BS, N_slots, N_img, H, W, 1),
                             'cntr_coords': (BS, N_obj, N_img, 2),
                             'z_ext': (BS, N_obj, N_img, D_ext),
                             'z_mean', 'z_stdvar': (BS, N_obj, N_img, D_shape)
                             }
        """

        with tf.name_scope('render_obj'):
            BS = tf.shape(rgb_in)[0]

            res = {k: [] for k in self.output_types}
            diffrgb_list = [rgb_in]
            msk_list = [tf.expand_dims(tf.zeros_like(rgb_in), axis=1)[..., :1]]
            # att_list = []
            z_obj_list = []

            for s in range(self.n_slots):
                with tf.name_scope('obj_' + str(s)):

                    # Get Object Encoding

                    # # - encoder input
                    msk_all = tf.concat(msk_list, axis=1)
                    msk_all = tf.reduce_max(msk_all, axis=1)
                    # att_list.append(msk_rendered)

                    input = tf.concat([rgb_in, diffrgb_list[-1], msk_all], axis=-1)
                    # input = tf.reshape(input, (BS * self.n_imgs, self.img_size, self.img_size, self.dim_enc_obj_in))

                    # - deterministic latent
                    z_obj_cur = self.encoder_obj([input], training=training)['z_mean']
                    z_obj_list.append(z_obj_cur)

                    # Get Object Rendering
                    obj_res = self.render_single_obj(z_obj_cur)
                    # # TODO: extr / scene_ids as input
                    # print('[WARNING] MOSNet: Usage of GT extrinsic parameters.')
                    # # obj_res = self.render_single_obj(z_obj_cur, z_extr_gt=gt_extr[:, s], scene_ids=scene_ids)
                    # obj_res = self.render_single_obj(z_obj_cur, z_extr_gt=gt_extr[:, s])

                    res_cur = {}
                    for k in self.output_types:
                        if k in obj_res:
                            res[k].append(obj_res[k])
                            res_cur[k] = tf.stack(res[k], axis=1)

                    # - combine all objects that were generated so far
                    combined_res = self.depth_ordering(res_cur, tf.zeros_like(res_cur['obj_rgb_pred'][:, 0]))
                    diffrgb_list.append(rgb_in - self.downsample_render(combined_res['rgb_pred'], antialiasing=True))
                    msk_list.append(self.downsample_render(combined_res['msk_pred'], antialiasing=True))

            # Final reconstruction
            for k, v in res.items():
                res[k] = K.stack(v, axis=1)

            # Latents off all objects
            z_obj_all = K.stack(z_obj_list, axis=1)  # (BS, N_slots, N_img, D)

            z_dict_tmp = {
                'z_shape': z_obj_all[..., :self.dim_shape],                                 # (BS, S, N_img, D_shape)
                'z_tex': z_obj_all[..., self.dim_shape:self.dim_shape + self.dim_app - 1],  # (BS, S, N_img, D_tex)
                'z_s': res['z_extr'][..., :1],                                              # (BS, S, N_img, D_s=1)
                'z_pos': res['z_extr'][..., 1:4],                                           # (BS, S, N_img, D_pos=3)
                'z_pos_x': res['z_extr'][..., 1:2],                                         # (BS, S, N_img, 1)
                'z_pos_y': res['z_extr'][..., 2:3],                                         # (BS, S, N_img, 1)
                'z_pos_z': res['z_extr'][..., 3:4],                                         # (BS, S, N_img, 1)
                'z_rot': res['z_extr'][..., 4:],                                            # (BS, S, N_img, D_rot=1)
                # 'z_rot_enc_sin': res['z_ext_enc'][:, :, :, 4:5],  # (BS, S, N_img, 1)
                # 'z_rot_enc_cos': res['z_ext_enc'][:, :, :, 5:]  # (BS, S, N_img, 1)
            }
            for k, v in z_dict_tmp.items():
                res[k + '_mean'] = v
                # res[k + '_var'] = tf.zeros_like(v)
            # res['z_obj_all'] = z_obj_all  # (BS, S, N_img, D)

        return res

    def predict_bg(self, input_imgs):
        """

        :param input_imgs:  (BS, N_imgs, H, W, 3)
        :return:            (BS, N_imgs, H, W, 3)
        """
        with tf.name_scope('render_bg'):
            with tf.name_scope('encoding'):
                assert (input_imgs.shape[-1] == self.dim_enc_bg_in)
                # input_imgs_reshaped = tf.reshape(input_imgs, (-1, self.img_size, self.img_size, self.dim_enc_bg_in))
                z_bg = self.encoder_bg([input_imgs])['z_mean']

            with tf.name_scope('latent'):
                # average of all bg encodings
                # z_bg = K.stack(z_bg_out, axis=1)
                # z_bg = K.mean(tf.reshape(z_bg, (-1, self.n_imgs, 2, self.dim_latent_bg)), axis=1)
                z_bg = K.mean(z_bg, axis=1)
                z_bg = activations.sigmoid(z_bg)

            with tf.name_scope('decoding'):
                bs = input_imgs.shape[0]
                bg_img = self.render_bg(z_bg, [bs, self.n_imgs, self.img_size_render, self.img_size_render, self.dim_feat])

        return bg_img

    # # ----------------------------------------------------------------------
    # # --- Scene Rendering

    @staticmethod
    def get_rot(z_extr_rot):
        """
        Compute object rotation based on inferred latent parameters.
        Options: (1) straight forward, (2) atan2 -> based on latent dimension
        :param z_extr_rot:  (BS, (S,) N_img, D_ext),  D_ext in [1, 2]
        :return:
        """

        if z_extr_rot.shape[-1] == 1:
            return z_extr_rot

        elif z_extr_rot.shape[-1] == 2:
            z_extr_rot = tf.math.atan2(z_extr_rot[..., 1:], z_extr_rot[..., 0:1]) / pi  # TODO: alternative: remove in renderer
            return z_extr_rot

        # elif z_extr_rot.shape[2] == 5:
        #
        #     z_rot_coars = z_extr_rot[..., :4]
        #     z_rot_fine = z_extr_rot[..., 4:5]
        #
        #     # softmax for probs
        #     z_rot_coars_prob = layers.Softmax(axis=-1)(z_rot_coars)
        #
        #     # restrict fine rotation
        #     z_rot_fine = 0.5 * activations.sigmoid(z_rot_fine) - 0.25
        #
        #     return z_rot_coars_prob, z_rot_fine

    def decompose_latent(self, z, z_extr_gt=None, scene_ids=None):

        # - Latent Decomposition
        z_shape = z[..., :self.dim_shape]
        # # -- TMP -- gt for testing, TODO: shape latent as input
        # if scene_ids is not None:
        #     z_shape = tf.gather(params=self._deepsdf.latents, indices=scene_ids)  # TODO: gt shape latent as input
        #     z_shape = tf.reshape(z_shape, (-1, self.n_imgs, self.dim_shape))
        z_shape = tf.reduce_max(z_shape, axis=1, keepdims=True)
        z_shape = tf.tile(z_shape, (1, self.n_imgs, 1))

        z_app = z[:, :, self.dim_shape:self.dim_shape + self.dim_app]
        z_app = tf.reduce_max(z_app, axis=1, keepdims=True)
        z_app = tf.tile(z_app, (1, self.n_imgs, 1))
        z_tex = z_app[..., :self.dim_app - 1]
        z_tex = tf.tile(z_tex, (1, self.n_imgs, 1))

        z_extr = z[..., self.dim_shape + self.dim_app:]
        z_extr_pos = z_extr[..., 0:3]
        z_extr_rot = self.get_rot(z_extr[..., 3:])
        z_extr_size = z_app[..., -1:]
        z_extr_size = (self.size_range[1] - self.size_range[0])*activations.sigmoid(z_extr_size) + self.size_range[0]
        z_extr = tf.concat([z_extr_size, z_extr_pos, z_extr_rot], axis=-1)
        # z_extr = tf.reshape(z_extr, (-1, EXTR_DIM))  # (BS*N, D_ext)
        # # -- TMP -- gt for testing, TODO: extr as input
        # if z_extr_gt is not None:
        #     # z_extr_gt = tf.reshape(z_extr_gt, (-1, EXTR_DIM))
        #     # z_extr = z_extr_gt  # all
        #     z_extr_gt = tf.reshape(z_extr_gt, (-1, EXTR_DIM))
        #     z_extr = tf.concat([z_extr_gt[:, :EXTR_DIM-1], z_extr[:, EXTR_DIM-1:]], axis=-1)  # only position
        # z_extr = tf.reshape(z_extr, (-1, self.n_imgs, EXTR_DIM))

        return z_shape, z_tex, z_extr

    def render_objs(self, z_obj_all, z_extr_gt=None):
        """
        Generate renderings for all objects individually.
        :param z_obj_all:   (BS, S, N_img, D)
        :param z_extr_gt:   (BS, S, N_img, D_extr)  - TODO: only for debugging, remove
        :return:    {'z_<>_mean': (BS, S, N_img, D_<>),    for <> in [shape, app, s, pos, rot]
                    'z_<>_stdvar': (BS, S, N_img, D_<>),
                    'imgs_rec': (BS, N_img, H, W, 3),
                    'occlusion': (BS, N_img, H, W, 1)}
        """

        res = {k: [] for k in self.output_types}

        for s in range(self.n_slots):
            z_obj = z_obj_all[:, s]

            if z_extr_gt is None:
                obj_res = self.render_single_obj(z_obj)
            else:
                obj_res = self.render_single_obj(z_obj, z_extr_gt[:, s])

            # append object results to lists
            for k in self.output_types:
                if k in obj_res:
                    res[k].append(obj_res[k])

        for k, v in res.items():
            res[k] = tf.stack(v, axis=1)

        # z_dict_tmp = {
        #     'z_shape': z_obj_all[..., :self.dim_shape],  # (BS, S, N_img, D_shape)
        #     'z_app': z_obj_all[..., self.dim_shape:self.dim_shape + self.dim_app - 1],  # (BS, S, N_img, D_app)
        #     'z_s': z_obj_all[..., self.dim_shape + self.dim_app - 1:self.dim_shape + self.dim_app],  # (BS, S, N_img, D_s)
        #     'z_pos': z_obj_all[..., self.dim_shape + self.dim_app:self.dim_shape + self.dim_app + 3],  # (BS, S, N_img, D_pos)
        #     'z_rot': z_obj_all[..., self.dim_shape + self.dim_app + 3:self.dim_shape + self.dim_app + 5]  # (BS, S, N_img, D_rot)
        # }
        # for k, v in z_dict_tmp.items():
        #     res[k + '_mean'] = v
        #     res[k + '_stdvar'] = tf.zeros_like(v)
        # res['z_obj_all'] = z_obj_all  # (BS, S, N_img, D)

        return res

    def downsample_render(self, img_in, antialiasing=False):
        if self.img_size_render == self.img_size:
            return img_in

        img_in_shape = tf.shape(img_in)
        num_channels = img_in_shape[-1]
        img_in = tf.reshape(img_in, [-1, self.img_size_render, self.img_size_render, num_channels])

        #from utils import viz
        #viz.show_image(img_in[0, ...].numpy(), 'img_in.png')

        if antialiasing:
            img_in = gaussian_blur(img_in, kernel_size=3, sigma=0.2)
            img_out = tf.image.resize(img_in, [self.img_size, self.img_size], antialias=False)
        else:
            img_out = tf.image.resize(img_in, [self.img_size, self.img_size], antialias=False, method='nearest')

        #viz.show_image(img_out[0, ...].numpy(), 'img_out.png')

        img_out_shape = img_in_shape.numpy()
        img_out_shape[-3] = self.img_size
        img_out_shape[-2] = self.img_size
        return tf.reshape(img_out, img_out_shape)

    def render_single_obj(self, z_obj, z_extr_gt=None, scene_ids=None):
        """
        Generate rendering (i.e. rgb, depth, occlusion) for a single object.
        :param z_obj:       (BS, N_img, D)
        :param z_extr_gt:   (BS, N_img, D_extr)  - TODO: only for debugging, remove
        :param scene_ids:   (BS, )              - TODO: only for debugging, remove
        :return: dictionary with object reconstruction results, eg. rgb image, depth map, occlusion, coordinates
        """

        z_shape, z_tex, z_extr = self.decompose_latent(z_obj, z_extr_gt, scene_ids)

        # - Rendering, shape
        z_shape_tmp = tf.reshape(z_shape, (-1, self.dim_shape))     # (BS*N, D_shape)
        z_extr_tmp = tf.reshape(z_extr, (-1, EXTR_DIM))             # (BS*N, D_extr=5)
        z_tex_tmp = tf.reshape(z_tex, (-1, self.dim_app-1))         # (BS*N_img, D_app)

        rendering_output = self.renderer(z_shape_tmp, z_extr_tmp)
        occlusion = tf.reshape(rendering_output['occlusion'], (-1, self.n_imgs, self.img_size_render, self.img_size_render, 1))
        occlusion_soft = tf.reshape(rendering_output['occlusion_soft'], (-1, self.n_imgs, self.img_size_render, self.img_size_render, 1))
        depth = tf.reshape(rendering_output['depth'], (-1, self.n_imgs, self.img_size_render, self.img_size_render, 1))
        cntr_coord = tf.reshape(rendering_output['cntr_coord'], (-1, self.n_imgs, 2))

        # -- Rendering, color
        pnts_silhouette = rendering_output['points_3d_silhouette']
        # pnts_silhouette_out = tf.reshape(pnts_silhouette, (-1, self.n_imgs, self.img_size_render*self.img_size_render, 3))
        pnts_silhouette = tf.reshape(pnts_silhouette, (-1, 1, 3))

        z_tex_tmp = tf.reshape(tf.tile(z_tex_tmp, (1, self.img_size_render * self.img_size_render)), (-1, 1, self.dim_app - 1))  # (BS*N_img*H*W, D_app)
        obj_color = self.decoder_rgb([pnts_silhouette], latent=z_tex_tmp)['sdf']
        obj_color = activations.sigmoid(obj_color)
        obj_color = tf.reshape(obj_color, (-1, self.n_imgs, self.img_size_render, self.img_size_render, self.dim_feat))  # (BS, N_img, H, W, 3)

        obj_img = obj_color * occlusion + (tf.ones_like(occlusion) - occlusion)

        return {
            'obj_rgb_pred': obj_img,             # (BS, N_img, H, W, 3)
            'obj_depth_pred': depth,             # (BS, N_img, H, W, 1)
            'obj_msk_pred': occlusion,           # (BS, N_img, H, W, 1)
            'obj_msk_soft_pred': occlusion_soft, # (BS, N_img, H, W, 1)
            'obj_cntr': cntr_coord,         # (BS, N_img, 2)
            'z_extr': z_extr,               # (BS, N_img, 5)
        }

    def render_bg(self, z_bg, img_size_render):
        """
        Generate background by either using single color from encoder or gt background image
        :param z_bg:        (..., 3)
        :param img_size_render:    <shape>
        :return:
        """
        if 'img' in self._bg.keys():
            print('MOSNet [INFO]: gt bg image')
            return self._bg['img']
        else:
            bg_color = tf.reshape(z_bg, (-1, 1, 1, 1, self.dim_feat))
            rec_img = bg_color * tf.ones(img_size_render)
            return rec_img

    def depth_ordering(self, obj_res, bg_rgb):
        """
        Pixel-wise depth ordering of objects.
        :param obj_res: dictionary with reconstruction results of all objects in scene, a.o.
                        {'obj_rec': (BS, N_slots', H, W, 1),
                         'depth', 'occlusion': (BS, N_slots', H, W, 1)
                        }
        :param bg_img:  (BS, N, H, W, 3)
        :return:        combined image, depth map, occlusion
        """

        bg_depth = self._bg['depth']
        depth = obj_res['obj_depth_pred']
        occlusion = obj_res['obj_msk_pred']
        obj_rec = obj_res['obj_rgb_pred']
        cur_n_slots = obj_rec.shape[1]

        # Set pixels which are not occupied by some object to high depth value
        d_nonvalid = tf.cast(tf.math.equal(depth, 0.), tf.float32)  # (BS, S, N, H, W, 1)
        depth_tmp = depth + self.depth_max * d_nonvalid

        # Get pixels with closest depth value
        d_min_valsort = tf.sort(depth_tmp, axis=1)[:, 0:1]  # (BS, 1, N_img, H, W, 1)
        d_closest = tf.cast(tf.math.equal(depth, d_min_valsort), tf.float32)  # (BS, S, N, H, W, 1)

        # Occlusion masks after object-wise occlusion is taken into account
        occlusion_all = tf.reshape(d_closest * occlusion,
                                   (-1, cur_n_slots, self.n_imgs, self.img_size_render, self.img_size_render, 1))

        # RGB rendering of all (current) objects on background (w.r.t. updated occlusion masks)
        rec_img = bg_rgb
        for s in range(cur_n_slots):
            obj_color = obj_rec[:, s]
            obj_occ = occlusion_all[:, s]
            rec_img = obj_color * obj_occ + rec_img * (tf.ones_like(obj_occ) - obj_occ)

        # Depth rendering of all (current) objects
        depth_obj = d_min_valsort[:, 0]
        occ_comb = tf.reduce_max(occlusion_all, axis=1)
        depth_all = depth_obj * occ_comb + bg_depth * (tf.ones_like(occ_comb) - occ_comb)

        return {'rgb_pred': rec_img,            # (BS, N_img, H, W, 3)
                'depth_pred': depth_all,        # (BS, N_img, H, W, 1)
                'msk_pred': occlusion_all,      # (BS, S, N_img, H, W, 1)
                }

    # ----------------------------------------------------------------------
    # --- 3D Constraints & Manipulaton

    def eval_pnts(self, pnts, z_shape, z_extr):
        """
        Returns SDF values (based on shape and extrinsic) for a set of given points.
        :param pnts:        (BS*N_img, N_pnts, 3)
        :param z_shape:     (BS*N_img, D_shape)
        :param z_extr:      (BS*N_img, D_extr)
        :return:    (N_pnts, 1),    sdf values at defined points
        """
        # Transfer points into object coordinate system
        s_obj, t_obj, R_obj = self.renderer.get_obj_trans(z_extr)
        pnts = tf.expand_dims(pnts, -1)
        pnts = s_obj * tf.einsum('bij,bajk->baik', R_obj, (pnts - t_obj))
        pnts = tf.squeeze(pnts, axis=-1)  # (BS*N_imgs, N_steps, 3)

        return self.renderer.get_sdf_values(pnts, z_shape)

    def crit_obj_intersection(self, rendering_res):
        """
        Returns measure whether two objects intersect in 3D.
        Between each pair of objects o1, o2, points along a line between their centers are sampled.
        Later, for any of these points p, it should hold that SDF_o1(p)+SDF_o2(p) >= 0.
        :param rendering_res:   dictionary with reconstruction results of all objects in scene, a.o.
                                { 'z_ext', 'z_shape_mean': (BS, S,  N_img, D_<>)}
        :return: updated rendering_res dictionary with
                    'obj_intersect_smpls':  (BS*N_img, S*(S-1)/2, N_steps, 1)
        """

        smpl_list = []

        for i in range(self.n_slots):
            obj1_extr = rendering_res['z_extr'][:, i]
            obj1_extr = tf.reshape(obj1_extr, (-1, EXTR_DIM))
            obj1_shape = rendering_res['z_shape_mean'][:, i]
            obj1_shape = tf.reshape(obj1_shape, (-1, self.dim_shape))

            for j in range(i + 1, self.n_slots):
                obj2_extr = rendering_res['z_extr'][:, j]
                obj2_extr = tf.reshape(obj2_extr, (-1, EXTR_DIM))
                obj2_shape = rendering_res['z_shape_mean'][:, i]
                obj2_shape = tf.reshape(obj2_shape, (-1, self.dim_shape))

                n_steps = 10
                x = tf.reshape(tf.linspace(0., 1., n_steps), (1, n_steps, 1))

                obj1_extr_pos = tf.expand_dims(obj1_extr[:, 1:4], axis=1)
                obj2_extr_pos = tf.expand_dims(obj2_extr[:, 1:4], axis=1)
                connecting_line = obj1_extr_pos + x * (obj2_extr_pos - obj1_extr_pos)

                sdf1 = self.eval_pnts(connecting_line, obj1_shape, obj1_extr)  # (BS*N_img, N_steps, 1)
                sdf2 = self.eval_pnts(connecting_line, obj2_shape, obj2_extr)

                smpl_list.append(sdf1 + sdf2)

        if len(smpl_list) > 0:
            obj_intersect_smpls = tf.stack(smpl_list, axis=1)  # (BS*N_img, S*(S-1)/2, N_steps, 1)
        else:
            obj_intersect_smpls = tf.zeros(())

        rendering_res['obj_intersect_smpls'] = obj_intersect_smpls

        return rendering_res

    def crit_ground(self, rendering_res):
        """
        Returns simplified measure whether an objects intersects with the ground.
        For each object o, the projection of the center on the ground is considered.
        Later, for any of these points p, it should hold that SDF_o(p) >= 0
        :param rendering_res:   dictionary with reconstruction results of all objects in scene, a.o.
                                { 'z_ext', 'z_shape_mean': (BS, S,  N_img, D_<>)}
        :return: updated rendering_res dictionary with
                    'obj_ground_smpls':  (...)
        """

        smpl_list = []

        for i in range(self.n_slots):
            obj_extr = rendering_res['z_extr'][:, i]
            obj_extr = tf.reshape(obj_extr, (-1, EXTR_DIM))
            obj_shape = rendering_res['z_shape_mean'][:, i]
            obj_shape = tf.reshape(obj_shape, (-1, self.dim_shape))

            proj_cntr = tf.concat([obj_extr[:, 1:3], tf.zeros_like(obj_extr[:, 3:4])], axis=-1)  # (BS*N_img, 3)
            proj_cntr = tf.reshape(proj_cntr, (-1, 1, 3))

            sdf_cntr = self.eval_pnts(proj_cntr, obj_shape, obj_extr)  # (BS*N_img, 1, 1)
            smpl_list.append(sdf_cntr)

        obj_ground_smpls = tf.concat(smpl_list, axis=1)  # (BS*N_img, S, 1)

        rendering_res['obj_ground_smpls'] = obj_ground_smpls

        return rendering_res

    # def manipulate_obj(self, z_obj_all, mode):
    #     """
    #     Method to manipulate the inferred objects' latents (best used during qualitative evaluation).
    #     :param z_obj_all:   (BS, S, N_img, D)
    #     :param mode:        type of manipulation from ['switch', 'pos', 'sample', 'rotate', 'remove', 'shape', 'texture']
    #     :return:
    #     """
    #
    #     z_shape = z_obj_all[..., :self.dim_shape]
    #     z_app = z_obj_all[..., self.dim_shape:self.dim_shape + self.dim_app]
    #     z_extr = z_obj_all[..., -self.dim_extr:]
    #     xy = z_extr[..., 0:2]
    #     z = z_extr[..., 2:3]
    #     r = z_extr[..., 3:]
    #
    #     delta = 1.0
    #
    #     if mode == 'switch':
    #         print('[INFO] MOSNet.manipulate_obj: switch object')
    #         xy = tf.concat([xy[:, 1:2], xy[:, 0:1], xy[:, 2:]], axis=1)
    #         z_obj_all_new = tf.concat([z_shape, z_app, xy, z, r], axis=-1)
    #     if mode == 'pos':
    #         print('[INFO] MOSNet.manipulate_obj: move object')
    #         x = xy[:, 0:1, ..., 0:1]
    #         y = xy[:, 0:1, ..., 1:2]
    #         # x = x + delta
    #         y = y + delta
    #         xy = tf.concat([tf.concat([x, y], axis=-1), xy[:, 1:2], xy[:, 2:3]], axis=1)
    #         z_obj_all_new = tf.concat([z_shape, z_app, xy, z, r], axis=-1)
    #     elif mode == 'sample':
    #         print('[INFO] MOSNet.manipulate_obj: sample object position')
    #         xy = tf.random.uniform(tf.shape(z_extr[..., 0:2]), minval=-1.5, maxval=1.5)
    #         # r = tf.random.uniform(tf.shape(r), minval=-1., maxval=1.)
    #         z_obj_all_new = tf.concat([z_shape, z_app, xy, z, r], axis=-1)
    #     elif mode == 'rotate':
    #         print('[INFO] MOSNet.manipulate_obj: rotate objects randomly')
    #         r = tf.random.uniform(tf.shape(r), minval=-1., maxval=1.)
    #         z_obj_all_new = tf.concat([z_shape, z_app, xy, z, r], axis=-1)
    #     elif mode == 'remove':
    #         print('[INFO] MOSNet.manipulate_obj: remove object')
    #         z_obj_all_new = tf.concat([z_obj_all[:, 1:2], z_obj_all[:, 1:2], z_obj_all[:, 2:]], axis=1)
    #     elif mode == 'shape':
    #         print('[INFO] MOSNet.manipulate_obj: interpolate between objects\' shapes')
    #         z_shape_a = z_shape[:, 0:1]
    #         z_shape_b = z_shape[:, 1:2]
    #         z_shape_c = z_shape[:, 2:3]
    #         z_shape_a = (1. - delta) * z_shape_a + delta * z_shape_b
    #         # z_shape_a = (1. - delta) * z_shape_a + delta * z_shape_c
    #         z_shape = tf.concat([z_shape_a, z_shape_b, z_shape_c], axis=1)
    #         z_obj_all_new = tf.concat([z_shape, z_app, xy, z, r], axis=-1)
    #     elif mode == 'texture':
    #         print('[INFO] MOSNet.manipulate_obj: interpolate between objects\' textures')
    #         # # alternative: switch textures
    #         # z_app = tf.concat([z_app[:, 1:2], z_app[:, 0:1], z_app[:, 2:]], axis=1)
    #         # # alternative: replace textures
    #         # z_app = tf.concat([z_app[:, 0:1], z_app[:, 0:1], z_app[:, 2:]], axis=1)
    #         z_app_a = z_app[:, 0:1]
    #         z_app_b = z_app[:, 1:2]
    #         z_app_c = z_app[:, 2:3]
    #         z_app_a = (1. - delta) * z_app_a + delta * z_app_b
    #         # z_app_a = (1. - delta) * z_app_a + delta * z_app_c
    #         z_app = tf.concat([z_app_a, z_app_b, z_app_c], axis=1)
    #         z_obj_all_new = tf.concat([z_shape, z_app, xy, z, r], axis=-1)
    #     elif mode == 'camera':
    #         z_obj_all_new = tf.concat([z_shape, z_app, xy, z, r], axis=-1)
    #         # org: [3.74057, -3.25382, 2.67183], [1.0765, 0., 0.8549]
    #
    #         # right:
    #         # self._renderer.set_new_camera_pose(t=[3.2538, 3.7406, 2.67183], rot=[1.0765, 0., 2.4257])     # +90°
    #         # self._renderer.set_new_camera_pose(t=[4.4376, 2.2106, 2.67183], rot=[1.0765, 0., 2.033])      # +66.5°
    #         # self._renderer.set_new_camera_pose(t=[4.9458, 0.3442, 2.67183], rot=[1.0765, 0., 1.6403])     # +45°
    #         # self._renderer.set_new_camera_pose(t=[4.7010, -1.5747, 2.67183], rot=[1.0765, 0., 1.2476])    # +22.5°
    #         # left
    #         # self._renderer.set_new_camera_pose(t=[2.2106, -4.4376, 2.67183], rot=[1.0765, 0., 0.4622])    # -22.5°
    #         # self._renderer.set_new_camera_pose(t=[0.3442, -4.9458, 2.67183], rot=[1.0765, 0., 0.0695])    # -45°
    #         # self._renderer.set_new_camera_pose(t=[-1.5747, -4.7010, 2.67183], rot=[1.0765, 0., -0.3232])   # -66.5°
    #         # self._renderer.set_new_camera_pose(t=[-3.2538, -3.7406, 2.67183], rot=[1.0765, 0., -0.7159])   # -90°
    #         # behind
    #         # self._renderer.set_new_camera_pose(t=[-3.7406, 3.2538, 2.67183], rot=[1.0765, 0., 3.9968])    # 180°
    #
    #     elif mode != "":
    #         print('ERROR: unknown mode', mode)
    #         exit()
    #
    #     z_obj_all_new = tf.Print(z_obj_all_new, [tf.shape(z_obj_all_new), tf.shape(z_obj_all)], 'Manipulation')
    #
    #     return z_obj_all_new

    # ----------------------------------------------------------------------
    # --- Objectives & Evaluation

    @staticmethod
    def wl(loss_list, loss_dict, params):
        loss_total = 0.
        for (name, loss) in loss_list:
            loss_dict['l-nw_' + name] = loss
            name = 'w-' + name
            if name in params and not (isinstance(params[name], float) and params[name] == 0):
                loss_dict[name.replace('w-', 'l_')] = params[name] * loss
                loss_total += params[name] * loss
        return loss_total

    def get_loss(self, outputs, inputs, params):
        """
        Define training objectives.
        :param outputs: s. forward() function
        :param inputs:  s. get_placeholder() function
        :param params:  dict with weights for loss function
        :return:    losses, {'loss_total' (), ...}
        """

        use_gt_based_losses = True
        if params['w-depth'] == 0. and params['w-rgb_sil'] == 0. and params['w-depth_sil'] == 0. and params['w-extr'] == 0. and params['w-normal'] == 0. and params['w-normal_sil'] == 0.:
            use_gt_based_losses = False # For render-and-compare optimization no GT data can be used in the loss
                                        # computation, only rgb reconstruction loss and losses based solely on the
                                        # predicted latents can be taken into account. If this variable is False, only
                                        # the losses necessary for render-and-compare optimization will be computed,
                                        # saving compute time.


        rgb_pred = outputs['rgb_pred']
        depth_pred = outputs['depth_pred']

        rgb_gt = inputs['rgb_gt']       #[batch_size, 1, h, w, 3]
        if use_gt_based_losses:
            depth_gt = inputs['depth_gt']   #[batch_size, 1, h, w]

        # from utils import viz
        # viz.show_image(rgb_pred[0, 0, ...].numpy(), 'render_rgb_pred.png')
        # viz.show_image(outputs['rgb_pred_full_size'][0, 0, ...].numpy(), 'render_rgb_pred_full_size.png')
        # viz.show_image(rgb_gt[0, 0, ...].numpy(), 'render_rgb_gt.png')
        # render_depth_gt = np.transpose(depth_gt, (0, 1, 3, 2))
        # viz.show_depth_list(tf.expand_dims(render_depth_gt[0, ...], axis=-1), 'render_depth_gt.png')
        # render_depth_pred = np.transpose(depth_pred, (0, 1, 3, 2, 4))
        # viz.show_depth_list(render_depth_pred[0, ...], 'render_depth_pred.png')


        batch_size = rgb_gt.shape[0]

        # Adjusting contrast with tf.image.adjust_contrast leads to LookupError: gradient registry has no entry for: AdjustContrastv2
        # The code below does the same:
        mean = tf.reduce_mean(rgb_gt, axis=(2, 3)).numpy()
        mean_tensor = tf.ones_like(rgb_gt).numpy()
        for i in range(batch_size):
            for channel in range(3):
                mean_tensor[i, 0, :, :, channel] *= mean[i, 0, channel]
        rgb_gt = (rgb_gt - mean_tensor) * self.contrast_factor + mean_tensor

        mean = tf.reduce_mean(rgb_pred, axis=(2, 3)).numpy()
        mean_tensor = tf.ones_like(rgb_pred).numpy()
        for i in range(batch_size):
            for channel in range(3):
                mean_tensor[i, 0, :, :, channel] *= mean[i, 0, channel]
        rgb_pred = (rgb_pred - mean_tensor) * self.contrast_factor + mean_tensor

        # from utils import viz
        # print(tf.reduce_max(rgb_gt))
        # print(tf.reduce_min(rgb_gt))
        # rgb_gt = tf.clip_by_value(rgb_gt, clip_value_min=0, clip_value_max=1) # just for visualization
        # viz.show_image(rgb_gt[0, 0, ...].numpy(), './rgb_gt_adj_contr.png')
        # rgb_pred = tf.clip_by_value(rgb_pred, clip_value_min=0, clip_value_max=1) # just for visualization
        # viz.show_image(rgb_pred[0, 0, ...].numpy(), './rgb_pred_adj_contr.png')


        h = w = rgb_gt.shape[2]

        if use_gt_based_losses:
            #from utils import viz
            normal_gt_valid, _, _ = create_normal_img(depth_gt) # The normal images are 2 pixels smaller in both height and width (the valid area of the convolution)
            #viz.show_image(normal_gt_valid.numpy()[0, 0, ...], './normal_gt_valid.png')
            # Add padding to the normal image to make it the same size as the input:
            normal_gt = np.zeros([batch_size, h, w, 3])
            normal_gt[:, 1:-1, 1:-1, :] = tf.reshape(normal_gt_valid, [batch_size, h-2, w-2, 3])
            normal_gt[:, 0, :, :] = normal_gt[:, 1, :, :]
            normal_gt[:, -1, :, :] = normal_gt[:, -2, :, :]
            normal_gt[:, :, 0, :] = normal_gt[:, :, 1, :]
            normal_gt[:, :, -1, :] = normal_gt[:, :, -2, :]
            #viz.show_image(normal_gt[0, ...], './normal_gt.png')
            # Convolution result with horizontal and vertical Sobel edge detection kernels:
            normal_edges = tf.image.sobel_edges(tf.Variable(normal_gt))
            normal_edges = tf.reduce_sum(tf.abs(normal_edges), axis=(-2, -1))
            # Select only the pixel above the threshold:
            normal_edges = tf.cast(normal_edges>self.silhouette_mask_normal_thresh, tf.float32)
            #viz.show_depth_list((normal_edges*255)[0,...].numpy().reshape([1, 1, h, w, 1]), './normal_edges.png')


            depth_edges = tf.image.sobel_edges(tf.reshape(depth_gt, [batch_size, h, w, 1]))
            # Convolution result with horizontal and vertical Sobel edge detection kernels:
            depth_edges = tf.abs(depth_edges[:, :, :, :, 0])+tf.abs(depth_edges[:, :, :, :, 1])
            # Select only the pixel above the threshold:
            depth_edges = tf.cast(depth_edges>=self.silhouette_mask_depth_thresh, tf.float32)

            silhouettes_list = []
            for i in range(batch_size):
                # Combine the edge images from depth and normal data:
                edge_img = depth_edges[i:i+1, ...]
                edge_img += tf.reshape(normal_edges[i:i+1, ...], [1, h, w, 1])
                edge_img = tf.cast(edge_img>0., tf.float32)

                # In the normal map there is a discontinuity where the depth is clipped to depth_max, here we romove the affected pixels
                edge_img -= tf.expand_dims(tf.cast(depth_gt[i, ...]>self.depth_max-0.5, tf.float32), axis=-1)
                edge_img = tf.clip_by_value(edge_img, clip_value_min=0., clip_value_max=1.)

                #from utils import viz
                #viz.show_depth_list((edge_img*255)[0,...].numpy().reshape([1, 1, h, w, 1]), './all_edges.png')

                # The dilated version of depth_edge_img with kernel size 3x3 (to get more context around the edges):
                edge_img_dil = tf.nn.max_pool2d(edge_img, ksize=(3, 3), strides=1, padding="SAME")
                # These are the silhouette pixels around the edges of the objects in the scene;
                # they are particularly important for the precise reconstruction of the position and shape of an object:
                edge_img_dil = K.stack([edge_img_dil, edge_img_dil, edge_img_dil], axis=3)[0, :, :, :, 0]

                #viz.show_image((rgb_gt[i, 0, ...]).numpy(), './test_img.png')
                #viz.show_image((rgb_gt[i, 0, ...] * edge_img_dil).numpy(), './test_silhouette_img.png')

                # The number of silhouette pixels in the image (used to normalize):
                num_silhouette_pixels = tf.reduce_sum(edge_img_dil)/3  # divided by the number of channels
                # Normalize w.r.t. the number of silhouette pixels: If there are few silhouette pixels in an image (i.e., if
                # there are only small objects) those silhouette pixels are weighted more strongly
                silhouettes_list.append(edge_img_dil/num_silhouette_pixels)
            silhouette_pixels_batch = K.stack(silhouettes_list, axis=0)


        # RGB image reconstruction loss
        with tf.name_scope('loss_rgb'):
            rgb_pred = tf.reshape(rgb_pred, (-1, self.img_size, self.img_size, 3))
            rgb_gt = tf.reshape(rgb_gt, (-1, self.img_size, self.img_size, 3))

            # SSIM
            max_val = 1.0
            l_ssim = K.mean(tf.image.ssim(rgb_pred, rgb_gt, max_val))
            l_ms_ssim = K.mean(tf.image.ssim_multiscale(img1=rgb_pred, img2=rgb_gt, max_val=max_val,
                                                        power_factors=(0.0448, 0.2856, 0.3001)))
            # L2, wo Gauss
            l_rgb_l2_org = K.mean(losses.mean_squared_error(rgb_pred, rgb_gt))

            # Gaussian Kernel
            if self.gauss_kernel > 0:
                sigma = params['gauss_sigma']
                kernelsize = self.gauss_kernel * 2 + 1

                rgb_pred = gaussian_blur(rgb_pred, kernel_size=kernelsize, sigma=sigma)
                rgb_gt = gaussian_blur(rgb_gt, kernel_size=kernelsize, sigma=sigma)

            # L2
            l_rgb_l2 = K.mean(losses.mean_squared_error(rgb_pred, rgb_gt))

            # Final
            l_rgb = l_rgb_l2

        # RGB silhouette image reconstruction loss
        if use_gt_based_losses:
            with tf.name_scope('loss_rgb_sil'):
                # L2
                # rgb_pred and rgb_gt are already smoothed with the appropriate kernel and contrast-adjusted by self.contrast_factor at this point
                l_rgb_sil_l2 = K.mean(losses.mean_squared_error(rgb_pred * silhouette_pixels_batch, rgb_gt * silhouette_pixels_batch))

                # Final
                l_rgb_sil = l_rgb_sil_l2

        # Depth reconstruction loss
        if use_gt_based_losses:
            with tf.name_scope('depth_loss'):

                # Gaussian Kernel
                depth_pred = tf.reshape(depth_pred, (-1, self.img_size, self.img_size, 1))
                depth_gt = tf.reshape(depth_gt, (-1, self.img_size, self.img_size, 1))

                # L1, wo Gauss
                l_depth_org = tf.reduce_mean(tf.abs(depth_pred - depth_gt))

                if self.gauss_kernel > 0:
                    depth_pred = gaussian_blur(depth_pred, kernel_size=kernelsize, sigma=sigma)
                    depth_gt = gaussian_blur(depth_gt, kernel_size=kernelsize, sigma=sigma)

                # Depth difference map
                depth_diff = depth_pred - depth_gt

                # L1, final
                l_depth = tf.reduce_mean(tf.abs(depth_diff))

        # Depth silhouette reconstruction loss
        if use_gt_based_losses:
            with tf.name_scope('depth_loss_sil'):
                # Depth difference map
                # depth_pred and depth_gt are already smoothed with the appropriate kernel at this point
                depth_diff_sil = depth_pred * silhouette_pixels_batch[:, :, :, 0:1] - depth_gt * silhouette_pixels_batch[:, :, :, 0:1]

                # L1, final
                l_depth_sil = tf.reduce_mean(tf.abs(depth_diff_sil))

        # Normal reconstruction loss
        if use_gt_based_losses:
            with tf.name_scope('normal_loss'):
                depth_pred = outputs['depth_pred']      # usage of depth wo gauss smoothing and anti-aliasing
                depth_gt = inputs['depth_gt']


                # Object mask:
                bg_batch = tf.expand_dims(self._bg['depth'][:, :, 0], axis=0)   # [1, img_size, img_size]
                if self.img_size_render != self.img_size:
                    bg_batch = tf.expand_dims(tf.image.resize(self._bg['depth'], [self.img_size, self.img_size])[:, :, 0], axis=0)
                bg_batch = tf.expand_dims(bg_batch, axis=0)                     # [1, 1, img_size, img_size]
                bg_batch = tf.repeat(bg_batch, batch_size, axis=0)              # [BS, 1, img_size, img_size]
                obj_mask_batch = tf.cast((bg_batch - depth_gt) > 0. , tf.float32)
                obj_mask_batch = tf.nn.max_pool2d(obj_mask_batch, ksize=(3, 3), strides=1, padding="SAME") # Dilation
                obj_mask_batch = tf.stack((obj_mask_batch, obj_mask_batch), axis=-1)
                # print(tf.reduce_max(obj_mask_batch))
                # print(tf.reduce_min(obj_mask_batch))

                # from utils import  viz
                # viz.show_depth_list((obj_mask_batch*255)[:, :, :, :, 0], './obj_mask.png')
                # viz.show_image(inputs['rgb_gt'][2, 0, ...].numpy(), './obj_rgb.png')

                # For each image normalize by the number of pixels belonging to objects:
                obj_mask_batch = obj_mask_batch.numpy()
                for i in range(batch_size):
                    num_mask_pixels = tf.reduce_sum(obj_mask_batch[i, :, :, :, 0])
                    obj_mask_batch[i, :, :, :, :] /= num_mask_pixels
                    # print(num_mask_pixels)
                    # print(tf.reduce_max(obj_mask_batch[i, :, :, :, :]))
                    # print(tf.reduce_min(obj_mask_batch[i, :, :, :, :]))
                    # print('\n')
                obj_mask_batch = tf.Variable(obj_mask_batch)

                normal_pred, dzdx_pred, dzdy_pred = create_normal_img(depth_pred)
                normal_gt, dzdx_gt, dzdy_gt = create_normal_img(depth_gt)

                dzdxy_pred = tf.concat([dzdx_pred, dzdy_pred], axis=-1)
                dzdxy_gt = tf.concat([dzdx_gt, dzdy_gt], axis=-1)

                use_object_mask = False

                if use_object_mask:
                    l_normal = K.mean(losses.mean_squared_error(dzdxy_pred*obj_mask_batch[:, :, 1:-1, 1:-1, :], dzdxy_gt*obj_mask_batch[:, :, 1:-1, 1:-1, :]))
                else:
                    l_normal = K.mean(losses.mean_squared_error(dzdxy_pred, dzdxy_gt))

        # Silhouette normal reconstruction loss
        if use_gt_based_losses:
            with tf.name_scope('normal_loss_sil'):
                silhouette_normal_mask_batch = tf.expand_dims(silhouette_pixels_batch[:, 1:-1, 1:-1, 0:2], axis=1)
                l_normal_sil = K.mean(losses.mean_squared_error(dzdxy_pred*silhouette_normal_mask_batch, dzdxy_gt*silhouette_normal_mask_batch))

                l_ctrl_normal_l2 = K.mean(losses.mean_squared_error(dzdxy_pred, dzdxy_gt))
                l_ctrl_normal_l1 = tf.reduce_mean(tf.abs(dzdxy_pred - dzdxy_gt))

                # Normal maps for visualization
                normal_pred = 0.5 * tf.ones_like(normal_pred) + 0.5 * normal_pred
                normal_gt = 0.5 * tf.ones_like(normal_gt) + 0.5 * normal_gt

        # Shape regularization loss
        with tf.name_scope('z-reg_loss'):
            z_shape = outputs['z_shape_mean']
            z_tex = outputs['z_tex_mean']

            l_shape_reg = tf.reduce_mean(tf.reduce_sum(K.square(z_shape), axis=-1))
            l_tex_reg = tf.reduce_mean(tf.reduce_sum(K.square(z_tex), axis=-1))

            l_z_reg = l_shape_reg + l_tex_reg

        # 3D pose estimation helpers -> only for testing/ tensorboard, TODO: object matching would be required..
        if use_gt_based_losses:
            with tf.name_scope('extr_3D_loss'):

                extr_loss_list = []

                for perm in list(permutations(range(self.n_slots))):
                    z_extr_pred = outputs['z_extr']
                    z_extr_gt = inputs['z_extr_gt']

                    z_extr_gt = tf.gather(z_extr_gt, perm, axis=1)

                    l_extr_3d_s = K.mean(tf.abs(z_extr_pred[..., 0] - z_extr_gt[..., 0]))
                    l_extr_3d_s2 = K.mean(tf.square(z_extr_pred[..., 0] - z_extr_gt[..., 0]))

                    l_extr_3d_x = K.mean(tf.abs(z_extr_pred[..., 1] - z_extr_gt[..., 1]))
                    l_extr_3d_y = K.mean(tf.abs(z_extr_pred[..., 2] - z_extr_gt[..., 2]))
                    l_extr_3d_z = K.mean(tf.abs(z_extr_pred[..., 3] - z_extr_gt[..., 3]))
                    l_extr_3d_pos = K.mean(tf.math.sqrt(tf.square(z_extr_pred[..., 1] - z_extr_gt[..., 1]) +
                                                        tf.square(z_extr_pred[..., 2] - z_extr_gt[..., 2]) +
                                                        tf.square(z_extr_pred[..., 3] - z_extr_gt[..., 3])))

                    # l_extr_3d_r_simple_2 = K.mean(tf.math.sqrt(2. * (1. - tf.math.cos(pi * (z_extr_pred[..., 4] - z_extr_gt[..., 4]))) + eps))
                    l_extr_3d_r = (pi / 2.) * K.mean(1. - tf.math.cos(pi * (z_extr_pred[..., 4] - z_extr_gt[..., 4])))

                    l_extr_3d_r_sym180 = (pi / 2.) * K.mean(
                        1. - tf.math.cos(pi * (z_extr_pred[..., 4] - z_extr_gt[..., 4]) + pi))
                    l_extr_3d_r_sym90 = (pi / 2.) * K.mean(
                        1. - tf.math.cos(pi * (z_extr_pred[..., 4] - z_extr_gt[..., 4]) + pi / 2.))
                    l_extr_3d_r_sym270 = (pi / 2.) * K.mean(
                        1. - tf.math.cos(pi * (z_extr_pred[..., 4] - z_extr_gt[..., 4]) - pi / 2.))
                    l_extr_3d_r_sym = tf.math.minimum(l_extr_3d_r_sym180, l_extr_3d_r)
                    l_extr_3d_r_sym_b = tf.math.minimum(l_extr_3d_r_sym90, l_extr_3d_r_sym270)
                    l_extr_3d_r_sym = tf.math.minimum(l_extr_3d_r_sym, l_extr_3d_r_sym_b)
                    l_extr_3d_r_sym = K.mean(l_extr_3d_r_sym)

                    l_extr = l_extr_3d_s2 + l_extr_3d_pos + l_extr_3d_r

                    extr_loss_list.append([l_extr, l_extr_3d_s, l_extr_3d_pos, l_extr_3d_x, l_extr_3d_y, l_extr_3d_z, l_extr_3d_r])

                extr_loss_list = tf.convert_to_tensor(extr_loss_list)
                l_extr_arg = tf.math.argmin(extr_loss_list[:, 2])
                l_extr_min = extr_loss_list[l_extr_arg]
                l_extr_min_list = [l_extr_min[i] for i in range(len(l_extr_min))]
                l_extr, l_extr_3d_s, l_extr_3d_pos, l_extr_3d_x, l_extr_3d_y, l_extr_3d_z, l_extr_3d_r = l_extr_min_list

        # Extrinsic regularization loss -> object center should be projected into the image plane
        with tf.name_scope('extr_reg_loss'):
            cntr_coords_pred = outputs['obj_cntr']

            l_cntr_1 = tf.clip_by_value(-1. * cntr_coords_pred, 0., 100.)
            l_cntr_2 = tf.clip_by_value(cntr_coords_pred - self.img_size, 0., 100.)

            l_cntr = K.mean(l_cntr_1 + l_cntr_2)

        # Ground loss -> object should not intersect with the ground (measured at projection of object center)
        with tf.name_scope('ground_loss'):
            z_extr_pred_z = outputs['z_extr'][..., 3:4]
            l_ground_obj_cntr_above = K.mean(tf.clip_by_value(-1. * z_extr_pred_z, 0., 100.))

            obj_ground_smpls = outputs['obj_ground_smpls']  # (BS*n_imgs, n_slots, 1)
            l_ground_obj_proj_above = K.mean(tf.clip_by_value(-1. * obj_ground_smpls, 0., 100.))
            l_ground_intersect = K.mean(tf.abs(obj_ground_smpls))

            l_ground = l_ground_obj_cntr_above + l_ground_obj_proj_above
            # l_ground = l_ground_obj_cntr_above + l_ground_intersect

        # Intersection loss -> objects should not intersect with each other
        with tf.name_scope('obj_intersect_loss'):
            obj_intersect_smpls = outputs['obj_intersect_smpls']  # (BS*N_img, n_obj_combis, n_steps, 1)
            l_intersect = K.mean(tf.clip_by_value(-1. * obj_intersect_smpls, 0., 100.))


        if not use_gt_based_losses:
            l_depth = 0.
            l_rgb_sil = 0.
            l_depth_sil = 0.
            l_normal = 0.
            l_normal_sil = 0.
            l_extr = 0.

        loss_dict = {}

        #loss_list = [('rgb', l_rgb), ('depth', l_depth), ('rgb_sil', l_rgb_sil), ('depth_sil', l_depth_sil), ('normal', l_normal), ('z-reg', l_z_reg), ('ground', l_ground)]
        loss_list = [('rgb', l_rgb), ('depth', l_depth), ('rgb_sil', l_rgb_sil), ('depth_sil', l_depth_sil),
                     ('z-reg', l_z_reg), ('ground', l_ground), ('normal', l_normal), ('normal_sil', l_normal_sil)]
        loss_list = loss_list + [('intersect', l_intersect), ('extr', l_extr)]
        loss = self.wl(loss_list, loss_dict, params)

        loss_dict['loss_total'] = loss

        loss_dict['l-ctrl_cntr'] = l_cntr
        if use_gt_based_losses:
            loss_dict['l-ctrl_3d_x'] = l_extr_3d_x
            loss_dict['l-ctrl_3d_y'] = l_extr_3d_y
            loss_dict['l-ctrl_3d_z'] = l_extr_3d_z
            loss_dict['l-ctrl_3d_s'] = l_extr_3d_s
            loss_dict['l-ctrl_3d_r'] = l_extr_3d_r
            loss_dict['l-ctrl_3d_r_sym'] = l_extr_3d_r_sym
        loss_dict['l-ctrl_ground_obj_cntr_above'] = l_ground_obj_cntr_above
        loss_dict['l-ctrl_ground_obj_proj_above'] = l_ground_obj_proj_above
        loss_dict['l-ctrl_ground_intersect'] = l_ground_intersect
        if use_gt_based_losses:
            loss_dict['l-ctrl_normal'] = l_ctrl_normal_l2
            loss_dict['l-ctrl_normal_l1'] = l_ctrl_normal_l1
        # loss_dict['l-ctrl_ssim'] = l_ssim
        # loss_dict['l-ctrl_ms-ssim'] = l_ms_ssim
        loss_dict['l-ctrl_rgb_l2'] = l_rgb_l2_org
        if use_gt_based_losses:
            loss_dict['l-ctrl_depth_l1'] = l_depth_org
        loss_dict['l-ctrl_intersect'] = l_intersect
        # visualizations
        #loss_dict['tmp_depth_diff'] = tf.reshape(depth_diff, (-1, self.n_imgs, self.img_size, self.img_size, 1))
        #loss_dict['normal_pred'] = normal_pred
        if self.gauss_kernel > 0:
            loss_dict['rgb_pred_gauss'] = tf.reshape(rgb_pred, (-1, self.n_imgs, self.img_size, self.img_size, 3))
            loss_dict['depth_pred_gauss'] = tf.reshape(depth_pred, (-1, self.n_imgs, self.img_size, self.img_size, 1))

        return loss_dict


class MultiObj3DNet_w_refiner(MultiObj3DNet):

    def __init__(self, cnfg):

        super(MultiObj3DNet_w_refiner, self).__init__(cnfg)

        #Setting all parts of the model other than the refiner untrainable, requires those parts to be pre-trained
        self.encoder_bg.trainable = False
        self.encoder_obj.trainable = False
        self.decoder_sdf.trainable = False
        self.decoder_rgb.trainable = True
        self.renderer.trainable = True

        with tf.name_scope("refine_objects"):
            self.ref_convs = cnfg['model']['mosnet']['enc_convs']
            self.ref_fc = cnfg['model']['mosnet']['enc_fc']
            self.refiner_obj = RefinerNet(dim_in=self.dim_enc_obj_in+4, dim_out=self.dim_latent_obj,
                                          conv_filters=self.ref_convs, mlp_units=self.ref_fc)
            self.refiner_obj.trainable = True

        self.silhouette_mask_normal_thresh = 3. #set to 5. for the silhouette only, set to 3. to also get the edges of cubes
        self.silhouette_mask_depth_thresh = 5.

        # This function is needed in the render-and-compare optimization to provide a common interface for MultiObj3DNet and MultiObj3DNet_w_refiner
    def gen_latents_single_slot(self, input, rgb_in, res, msk_list, diffrgb_list, training=False):

        # Get Object Encoding
        z_obj_cur = self.encoder_obj([input], training=training)['z_mean']

        # Get Object Rendering
        obj_res = self.render_single_obj(z_obj_cur)

        res_cur_enc = {}
        res_enc = {}
        for k, v in res.items():
            res_enc[k] = copy.deepcopy(v)
        for k in self.output_types:
            if k in obj_res:
                res_enc[k].append(obj_res[k])
                res_cur_enc[k] = tf.stack(res_enc[k], axis=1)

        # - combine all objects that were generated so far
        combined_res_enc = self.depth_ordering(res_cur_enc, tf.zeros_like(res_cur_enc['obj_rgb_pred'][:, 0]))


        # Get Object Refinement

        # # - refiner input
        msk_prev = tf.concat(msk_list, axis=1)
        msk_prev = tf.reduce_max(msk_prev, axis=1)

        msk_list_ = msk_list
        msk_list_.append(combined_res_enc['msk_pred'])
        msk_curr = tf.concat(msk_list_, axis=1)
        msk_curr = tf.reduce_max(msk_curr, axis=1)

        diffrgb_prev = diffrgb_list[-1]

        diffrgb_curr = rgb_in - combined_res_enc['rgb_pred']

        input = tf.concat([rgb_in, diffrgb_prev, diffrgb_curr, msk_prev, msk_curr], axis=-1)

        return self.refiner_obj([input], latents_in=z_obj_cur, training=training)

    def predict_objs(self, rgb_in, training, gt_extr=None, scene_ids=None):
        """

        :param rgb_in:      (BS, N_imgs, H, W, 3)
        :return:            {'img_rec': (BS, N_img, H, W, 3),
                             'depth': (BS, N_img, H, W, 1),
                             'obj_rec': (BS, N_slots, N_img, H, W, 3),
                             'occlusion', 'occlusion_soft': (BS, N_slots, N_img, H, W, 1),
                             'cntr_coords': (BS, N_obj, N_img, 2),
                             'z_ext': (BS, N_obj, N_img, D_ext),
                             'z_mean', 'z_stdvar': (BS, N_obj, N_img, D_shape)
                             }
        """
        test_visualizations = False

        with tf.name_scope('render_obj'):
            BS = tf.shape(rgb_in)[0]

            res = {k: [] for k in self.output_types}
            diffrgb_list = [rgb_in]
            msk_list = [tf.expand_dims(tf.zeros_like(rgb_in), axis=1)[..., :1]]
            # att_list = []
            z_obj_list = []

            for s in range(self.n_slots):
                with tf.name_scope('obj_' + str(s)):

                    # Get Object Encoding

                    # # - encoder input
                    msk_all = tf.concat(msk_list, axis=1)
                    msk_all = tf.reduce_max(msk_all, axis=1)
                    # att_list.append(msk_rendered)

                    input = tf.concat([rgb_in, diffrgb_list[-1], msk_all], axis=-1)
                    # input = tf.reshape(input, (BS * self.n_imgs, self.img_size, self.img_size, self.dim_enc_obj_in))

                    if test_visualizations:
                        from utils import viz
                        rgb = (np.transpose(rgb_in[0, ...], (1, 0, 2, 3))).reshape((64, 64, 3))
                        viz.show_image(rgb, 'test_enc_rgb_in.png')
                        diffrgb = (np.transpose(diffrgb_list[-1][0, ...], (1, 0, 2, 3))).reshape((64, 64, 3))
                        viz.show_image(diffrgb, 'test_enc_diffrgb.png')
                        mask = tf.stack([msk_all[0, ...], msk_all[0, ...], msk_all[0, ...]], axis=-1)
                        viz.show_image(mask[0, :, :, 0, :].numpy(), 'test_enc_mask.png')

                    # - deterministic latent
                    z_obj_cur_enc = self.encoder_obj([input], training=training)['z_mean']

                    # Get Object Rendering
                    obj_res_enc = self.render_single_obj(z_obj_cur_enc)

                    res_cur_enc = {}
                    res_enc = {}
                    for k, v in res.items():
                        res_enc[k] = copy.deepcopy(v)
                    for k in self.output_types:
                        if k in obj_res_enc:
                            res_enc[k].append(obj_res_enc[k])
                            res_cur_enc[k] = tf.stack(res_enc[k], axis=1)

                    # - combine all objects that were generated so far
                    combined_res_enc = self.depth_ordering(res_cur_enc, tf.zeros_like(res_cur_enc['obj_rgb_pred'][:, 0]))


                    # Get Object Refinement

                    # # - refiner input
                    msk_prev = tf.concat(msk_list, axis=1)
                    msk_prev = tf.reduce_max(msk_prev, axis=1)

                    msk_list_ = msk_list
                    msk_list_.append(combined_res_enc['msk_pred'])
                    msk_curr = tf.concat(msk_list_, axis=1)
                    msk_curr = tf.reduce_max(msk_curr, axis=1)

                    diffrgb_prev = diffrgb_list[-1]

                    diffrgb_curr = rgb_in - combined_res_enc['rgb_pred']

                    input = tf.concat([rgb_in, diffrgb_prev, diffrgb_curr, msk_prev, msk_curr], axis=-1)

                    if test_visualizations:
                        rgb = (np.transpose(rgb_in[0, ...], (1, 0, 2, 3))).reshape((64, 64, 3))
                        viz.show_image(rgb, 'test_ref_rgb_in.png')
                        diffrgb_prev_img = (np.transpose(diffrgb_prev[0, ...], (1, 0, 2, 3))).reshape((64, 64, 3))
                        viz.show_image(diffrgb_prev_img, 'test_ref_diffrgb_prev.png')
                        diffrgb_curr_img = (np.transpose(diffrgb_curr[0, ...], (1, 0, 2, 3))).reshape((64, 64, 3))
                        viz.show_image(diffrgb_curr_img, 'test_ref_diffrgb_curr.png')
                        mask_prev_img = tf.stack([msk_prev[0, ...], msk_prev[0, ...], msk_prev[0, ...]], axis=-1)
                        viz.show_image(mask_prev_img[0, :, :, 0, :].numpy(), 'test_ref_mask_prev.png')
                        mask_curr_img = tf.stack([msk_curr[0, ...], msk_curr[0, ...], msk_curr[0, ...]], axis=-1)
                        viz.show_image(mask_curr_img[0, :, :, 0, :].numpy(), 'test_ref_mask_curr.png')

                    # - deterministic latent
                    z_obj_cur = self.refiner_obj([input], latents_in=z_obj_cur_enc, training=training)['z_mean']

                    # Get Refined Object Rendering
                    obj_res = self.render_single_obj(z_obj_cur)
                    # # TODO: extr / scene_ids as input
                    # print('[WARNING] MOSNet: Usage of GT extrinsic parameters.')
                    # # obj_res = self.render_single_obj(z_obj_cur, z_extr_gt=gt_extr[:, s], scene_ids=scene_ids)
                    # obj_res = self.render_single_obj(z_obj_cur, z_extr_gt=gt_extr[:, s])

                    res_cur = {}
                    for k in self.output_types:
                        if k in obj_res:
                            res[k].append(obj_res[k])
                            res_cur[k] = tf.stack(res[k], axis=1)

                    # - combine all objects that were generated so far
                    combined_res = self.depth_ordering(res_cur, tf.zeros_like(res_cur['obj_rgb_pred'][:, 0]))



                    z_obj_list.append(z_obj_cur)
                    diffrgb_list.append(rgb_in - combined_res['rgb_pred'])
                    msk_list.append(combined_res['msk_pred'])

                    if test_visualizations:
                        diffrgb_list_new = (np.transpose(diffrgb_list[-1][0, ...], (1, 0, 2, 3))).reshape((64, 64, 3))
                        viz.show_image(diffrgb_list_new, 'test_diffrgb_list_new.png')
                        msk_all_new = tf.concat(msk_list, axis=1)
                        msk_all_new = tf.reduce_max(msk_all_new, axis=1)
                        msk_all_new = tf.stack([msk_all_new[0,  ...], msk_all_new[0,  ...], msk_all_new[0,  ...]], axis=-1)
                        viz.show_image(msk_all_new[0, :, :, 0, :].numpy(), 'test_mask_list_new.png')

            # Final reconstruction
            for k, v in res.items():
                res[k] = K.stack(v, axis=1)

            # Latents off all objects
            z_obj_all = K.stack(z_obj_list, axis=1)  # (BS, N_slots, N_img, D)

            z_dict_tmp = {
                'z_shape': z_obj_all[..., :self.dim_shape],                                 # (BS, S, N_img, D_shape)
                'z_tex': z_obj_all[..., self.dim_shape:self.dim_shape + self.dim_app - 1],  # (BS, S, N_img, D_tex)
                'z_s': res['z_extr'][..., :1],                                              # (BS, S, N_img, D_s=1)
                'z_pos': res['z_extr'][..., 1:4],                                           # (BS, S, N_img, D_pos=3)
                'z_pos_x': res['z_extr'][..., 1:2],                                         # (BS, S, N_img, 1)
                'z_pos_y': res['z_extr'][..., 2:3],                                         # (BS, S, N_img, 1)
                'z_pos_z': res['z_extr'][..., 3:4],                                         # (BS, S, N_img, 1)
                'z_rot': res['z_extr'][..., 4:],                                            # (BS, S, N_img, D_rot=1)
                # 'z_rot_enc_sin': res['z_ext_enc'][:, :, :, 4:5],  # (BS, S, N_img, 1)
                # 'z_rot_enc_cos': res['z_ext_enc'][:, :, :, 5:]  # (BS, S, N_img, 1)
            }
            for k, v in z_dict_tmp.items():
                res[k + '_mean'] = v
                # res[k + '_var'] = tf.zeros_like(v)
            # res['z_obj_all'] = z_obj_all  # (BS, S, N_img, D)

        return res


# ---------------------------------------------------------------------------------------------------------------------

class MultiObj3DNetRefineConv(MultiObj3DNet):

    def __init__(self, cnfg):
        super(MultiObj3DNetRefineConv, self).__init__(cnfg)

        with tf.name_scope("conv_refine"):
            # self.dim_refine_in = 3 + 1 + 2 + 1    # (rgb+depth+normal+mask)
            self.dim_refine_in = 3 + 2             # (rgb+normal)
            self.conv_refine_filters = [self.dim_refine_in]
            self.conv_refine = ConvNet(self.dim_refine_in, dim_out=3, conv_filters=self.conv_refine_filters)

    def call(self, inputs, training=False, manipulation=None):

        res = super(MultiObj3DNetRefineConv, self).call(inputs, training, manipulation)

        # combined_res_new = self.depth_ordering(res, res['bg_img'], final=True)
        # for k, v in combined_res_new.items():
        #     res[k] = v
        #
        # return res

        rgb_pred = res['rgb_pred']
        msk_pred = tf.reduce_sum(res['msk_pred'], axis=1)
        depth_pred = res['depth_pred']
        paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
        depth_pred_pad = tf.pad(depth_pred, paddings, "SYMMETRIC")
        _, dzdx_pred, dzdy_pred = create_normal_img(depth_pred_pad)

        refine_input = tf.concat([rgb_pred, dzdx_pred, dzdy_pred], axis=-1)
        # refine_input = tf.concat([rgb_pred, depth_pred, dzdx_pred, dzdy_pred, msk_pred], axis=-1)
        rgb_new = self.conv_refine(refine_input)['output']
        rgb_new = activations.sigmoid(rgb_new)
        res['rgb_pred_refine'] = rgb_new

        return res

    # def depth_ordering(self, obj_res, bg_rgb, final=False):
    #     """
    #     Apply refine-convnet here..
    #     """
    #
    #     res = super(MultiObj3DNetRefineConv, self).depth_ordering(obj_res, bg_rgb)
    #
    #     rgb_res = self.conv_refine(res['rgb_pred'])['output']
    #     if not final:
    #         rgb_res = tf.stop_gradient(rgb_res)
    #     rgb_res = activations.sigmoid(rgb_res)
    #     res['rgb_pred'] = rgb_res
    #
    #     return res

    def get_loss(self, outputs, inputs, params):
        loss_dict = super(MultiObj3DNetRefineConv, self).get_loss(outputs, inputs, params)

        rgb_pred = outputs['rgb_pred_refine']
        rgb_gt = inputs['rgb_gt']

        # RGB image reconstruction loss
        with tf.name_scope('loss_rgb_refine'):
            rgb_pred = tf.reshape(rgb_pred, (-1, self.img_size, self.img_size, 3))
            rgb_gt = tf.reshape(rgb_gt, (-1, self.img_size, self.img_size, 3))

            # Gaussian Kernel
            if self.gauss_kernel > 0:
                sigma = params['gauss_sigma']
                kernelsize = self.gauss_kernel * 2 + 1

                rgb_pred = gaussian_blur(rgb_pred, kernel_size=kernelsize, sigma=sigma)
                rgb_gt = gaussian_blur(rgb_gt, kernel_size=kernelsize, sigma=sigma)

            # L2
            l_rgb_refine = K.mean(losses.mean_squared_error(rgb_pred, rgb_gt))

        add_loss_list = [('rgb-refine', l_rgb_refine)]
        l_refine = self.wl(add_loss_list, loss_dict, params)
        loss = loss_dict['loss_total'] + l_refine

        loss_dict['loss_total'] = loss

        return loss_dict


# ---------------------------------------------------------------------------------------------------------------------


def get_model(cnfg_model, ext):
    print('Load Model: MOSNet')
    if ext == 'org':
        model = MultiObj3DNet(cnfg_model)
    elif ext == 'refine':
        model = MultiObj3DNet_w_refiner(cnfg_model)
    elif ext == 'refine_conv':
        model = MultiObj3DNetRefineConv(cnfg_model)
    else:
        print('MOSNet, Model Variation {} not defined.'.format(ext))
        exit(1)

    return model
