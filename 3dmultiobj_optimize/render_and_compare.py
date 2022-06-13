import gc
import math
import os
import tensorflow as tf
import argparse
import copy
import tensorflow.keras.backend as K
import itertools

from datetime import datetime, timedelta
from utils import data_provider
from utils import viz
from utils.shared_funcs import *
from utils.tf_funcs import *
from preprocess.scale_off import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clevr', help='Data Set')
    parser.add_argument('--data_dir', help='Data dir')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--eval_dir', default='eval', help='Log dir [default: eval]')
    parser.add_argument('--model', default='mosnet-org', help='Model name')
    parser.add_argument('--no_gpu', default='0', help='Do not use GPU')
    parser.add_argument('--config', default='', help='Configuration file [cnfg_<model>_<dataset>]')
    parser.add_argument('--message', default='', help='Message that specifies settings etc. (for log dir name)')
    parser.add_argument('--split', default='val', help='[train, val]')
    parser.add_argument('--results_dir', default='./results', help='The directory to store the results in')

    #Allows to specify a range of scenes in the dataset to be optimized:
    parser.add_argument('--start_scene', default=None)
    parser.add_argument('--stop_scene', default=None)
    return parser.parse_args()

def main():
    # Create dataset iterator over the validation set
    dataset_val, iterator_val = get_data_iterator(DATA, CONFIG['r-and-c'])
    iterator = iterator_val

    print("+++++++++++++++++++++++++++++++")

    # Restore weights of the model
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=MODEL)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(LOG_DIR, 'ckpts'), max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        epoch = int(ckpt.step)
        print("Restored from {}, epoch {}".format(manager.latest_checkpoint, epoch))
    else:
        print('[ERROR] render_and_compare.py, no pre-trained model found at ' + LOG_DIR, 'red')
        exit(1)

    print("+++++++++++++++++++++++++++++++")

    # Set all weights untrainable
    for layer in MODEL.layers:
        layer.trainable = False
    print("Set all weights untrainable")

    # Define the training parameters and get the model input
    params = {}
    weights = CONFIG['model'][model_base_name]['l-weights']
    for k, w in weights.items():
        params['w-' + k] = w
    params['w-z-reg'] = tf.constant(0.025)

    loss_list = []

    if CONFIG['r-and-c']['loss_eval']:
        CONFIG['r-and-c']['num_examples'] = 1000

    start_time_main = datetime.now()
    time_now_str = '{:%d.%m_%H:%M}'.format(start_time_main)
    print("Starting optimization for {} images; Current time: {}".format(CONFIG['r-and-c']['num_examples'], time_now_str))

    if CONFIG['r-and-c']['produce_dataset']:
        data_train = data_provider.DataSetMultiObj(FLAGS.data_dir, 'train', CONFIG['data'])
        r_and_c_opt.create_optimized_dataset(data_train, params, './opt_dataset_{}_{}'.format(FLAGS.message, datetime.now().strftime("%Y_%m_%d-%I_%M_%S")))

    if FLAGS.start_scene is not None and FLAGS.stop_scene is not None:
        start = int(FLAGS.start_scene)
        stop = int(FLAGS.stop_scene)
    else:
        start = 0
        stop = CONFIG['r-and-c']['num_examples']

    print("Start: {}, Stop: {}".format(start, stop))

    for _ in range(start):
        input = next(iterator)

    for i in range(start, stop):
        gc.collect(generation=0)
        gc.collect(generation=1)
        gc.collect(generation=2)

        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++")
        time_now_str = '{:%d.%m_%H:%M}'.format(datetime.now())
        print("Starting optimization for image {}; Current time: {}".format(i, time_now_str))
        print("\nComputing example {}".format(i))

        input = next(iterator)
        input = MODEL.get_input(input)

        if CONFIG['r-and-c']['loss_eval'] and i==start:
            params['gauss_sigma'] = 0.
            res = MODEL([input['rgb_in']])
            losses = MODEL.get_loss(res, input, params)

            sum_losses_pred = {}
            sum_losses_pred_opt = {}

            for k, _ in losses.items():
                sum_losses_pred[k] = 0.
                sum_losses_pred_opt[k] = 0.

        results = r_and_c_opt.optimize_latents(input, params)
        #results = r_and_c_opt.optimize_latents_with_gt_extr(input, params)

        res = results["res"]
        loss_list.append(results["losses"])

        if not CONFIG['model']['mosnet']['anti_aliasing']:
            latents_file = open('./log/clevr_mosnet-org(' + FLAGS.message + '_eval_w_r_and_c)/latents_20.npy', 'ab')
            if CONFIG['r-and-c']['learning_rate_shape'] > 0:
                latents_file = open('./log/clevr_mosnet-org(' + FLAGS.message + '_eval_w_r_and_c_shape_updates_no_shape_regularizer)/latents_20.npy', 'ab')
        else:
            latents_file = open('./log/clevr_mosnet-org(' + FLAGS.message + '_eval_w_r_and_c_anti_aliasing_depth_w_aa)/latents_20.npy', 'ab')
        opt_latents = tf.concat([res['z_shape_mean'], res['z_tex_mean'], res['z_s_mean'], res['z_pos_x_mean'],  res['z_pos_y_mean'],  res['z_pos_z_mean'], res['z_rot_mean']], axis=-1).numpy()
        np.save(latents_file, opt_latents)
        print("saved latents for scene {}".format(i))
        print(opt_latents)
        if not CONFIG['model']['mosnet']['anti_aliasing']:
            latents_file = open('./log/clevr_mosnet-org(' + FLAGS.message + '_eval_w_r_and_c)/latents_21.npy', 'ab')
            if CONFIG['r-and-c']['learning_rate_shape'] > 0:
                latents_file = open('./log/clevr_mosnet-org(' + FLAGS.message + '_eval_w_r_and_c_shape_updates_no_shape_regularizer)/latents_21.npy', 'ab')
        else:
            latents_file = open('./log/clevr_mosnet-org(' + FLAGS.message + '_eval_w_r_and_c_anti_aliasing_depth_w_aa)/latents_21.npy', 'ab')
        opt_latents = res['z_all'].numpy()
        np.save(latents_file, opt_latents)
        print("saved latents for scene {}".format(i))
        print(opt_latents)

        # Display results:
        img_size = CONFIG['data']['img_size']
        n_img = CONFIG['model'][model_base_name]['n_imgs']

        # Display the input/ ground truth
        rgb_gt = input['rgb_gt'][0]
        rgb_gt = (np.transpose(rgb_gt, (1, 0, 2, 3))).reshape((img_size, n_img * img_size, 3))
        viz.show_image(rgb_gt, CONFIG['r-and-c']['results_dir']+'/ex_{}_rgb_gt.png'.format(i))

        msk_gt = input['msk_gt'][0]
        msk_gt = (np.transpose(msk_gt, (2, 1, 3, 0))).reshape((img_size, n_img * img_size, 3))
        viz.show_image(msk_gt, CONFIG['r-and-c']['results_dir']+'/ex_{}_msk_gt.png'.format(i))

        depth_gt = input['depth_gt'][0]
        depth_gt = (np.transpose(depth_gt, (1, 0, 2))).reshape((n_img, img_size, img_size, 1))
        viz.show_depth_list(np.transpose(depth_gt), CONFIG['r-and-c']['results_dir']+'/ex_{}_depth_gt.png'.format(i))
        normal_gt, _, _ = create_normal_img(depth_gt)
        normal_gt = (np.transpose(normal_gt, (1, 0, 2, 3))).reshape((img_size-2, n_img*img_size-2, 3))
        normal_gt = 0.5 * np.ones(normal_gt.shape) + 0.5 * normal_gt #For better visualization
        viz.show_image(normal_gt, CONFIG['r-and-c']['results_dir']+'/ex_{}_normal_gt.png'.format(i))

        # Display the output of the model
        model_res = MODEL([input['rgb_in']])
        if CONFIG['r-and-c']['loss_eval']:
            losses_pred = MODEL.get_loss(model_res, input, params)
            for k, v in losses_pred.items():
                sum_losses_pred[k] += v
            print("loss_pred = {}".format(losses_pred['loss_total']))
        rgb_pred = model_res['rgb_pred'][0]
        rgb_pred = (np.transpose(rgb_pred, (1, 0, 2, 3))).reshape((img_size, n_img*img_size, 3))
        viz.show_image(rgb_pred, CONFIG['r-and-c']['results_dir']+'/ex_{}_rgb_pred.png'.format(i))
        depth_pred = model_res['depth_pred'][0, :, :, :, 0]
        depth_pred = (np.transpose(depth_pred, (1, 0, 2))).reshape((n_img, img_size, img_size, 1))
        viz.show_depth_list(np.transpose(depth_pred), CONFIG['r-and-c']['results_dir']+'/ex_{}_depth_pred.png'.format(i))
        normal_pred, _, _ = create_normal_img(depth_pred)
        normal_pred = (np.transpose(normal_pred, (1, 0, 2, 3))).reshape((img_size-2, n_img*img_size-2, 3))
        normal_pred = 0.5 * np.ones(normal_gt.shape) + 0.5 * normal_pred
        viz.show_image(normal_pred, CONFIG['r-and-c']['results_dir']+'/ex_{}_normal_pred.png'.format(i))

        for obj in range (tf.shape(model_res['obj_rgb_pred'])[1]):
            obj_rgb = MODEL.downsample_render(model_res['obj_rgb_pred'])[0, obj, 0, ...].numpy().reshape((img_size, img_size, 3))
            viz.show_image(obj_rgb, CONFIG['r-and-c']['results_dir']+'/ex_{}_rgb_pred_obj_{}.png'.format(i, obj))

        # Display the output of the model with iterative refinement
        rgb_pred_opt = res['rgb_pred'][0]
        if CONFIG['r-and-c']['loss_eval']:
            losses_pred_opt = results['losses']
            for k, v in losses_pred_opt.items():
                sum_losses_pred_opt[k] += v
            print("loss_pred_opt = {}".format(losses_pred_opt['loss_total']))
        rgb_pred_opt = (np.transpose(rgb_pred_opt, (1, 0, 2, 3))).reshape((img_size, n_img*img_size, 3))
        viz.show_image(rgb_pred_opt, CONFIG['r-and-c']['results_dir']+'/ex_{}_rgb_pred_opt.png'.format(i))
        if CONFIG['model']['mosnet']['anti_aliasing']:
            rgb_pred_opt_full_size = res['rgb_pred_full_size'][0]
            rgb_pred_opt_full_size = (np.transpose(rgb_pred_opt_full_size, (1, 0, 2, 3))).reshape((img_size*2, n_img*img_size*2, 3))
            viz.show_image(rgb_pred_opt_full_size, CONFIG['r-and-c']['results_dir']+'/ex_{}_rgb_pred_opt_full_size.png'.format(i))
        depth_pred_opt = res['depth_pred'][0, :, :, :, 0]
        depth_pred_opt = (np.transpose(depth_pred_opt, (1, 0, 2))).reshape((n_img, img_size, img_size, 1))
        viz.show_depth_list(np.transpose(depth_pred_opt), CONFIG['r-and-c']['results_dir']+'/ex_{}_depth_pred_opt.png'.format(i))
        normal_pred_opt, _, _ = create_normal_img(depth_pred_opt)
        normal_pred_opt = (np.transpose(normal_pred_opt, (1, 0, 2, 3))).reshape((img_size-2, n_img*img_size-2, 3))
        normal_pred_opt = 0.5 * np.ones(normal_gt.shape) + 0.5 * normal_pred_opt
        viz.show_image(normal_pred_opt, CONFIG['r-and-c']['results_dir']+'/ex_{}_normal_pred_opt.png'.format(i))

        for obj in range (tf.shape(res['obj_rgb_pred'])[1]):
            obj_rgb = res['obj_rgb_pred'][0, obj, 0, ...].numpy().reshape((img_size, img_size, 3))
            viz.show_image(obj_rgb, CONFIG['r-and-c']['results_dir']+'/ex_{}_rgb_pred_opt_obj_{}.png'.format(i, obj))

        normal_pred, dzdx_pred, dzdy_pred = create_normal_img(depth_pred)
        normal_gt, dzdx_gt, dzdy_gt = create_normal_img(depth_gt)

        if CONFIG['r-and-c']['loss_eval']:
            print("\nAvg losses w/o  optimization: ")
            for k, v in sum_losses_pred.items():
                print("{}: {}".format(k, v/(i+1)))
            print("\nAvg losses with optimization: ")
            for k, v in sum_losses_pred_opt.items():
                print("{}: {}".format(k, v/(i+1)))

    averages = {}
    for k, v in loss_list[0].items():
        averages[k] = 0.
    for i in range(CONFIG['r-and-c']['num_examples']):
        for k, v in loss_list[i].items():
            averages[k] += v/CONFIG['r-and-c']['num_examples']
    for k, v in averages.items():
        print("Average {}: {}".format(k, v))

    time_now = datetime.now()
    time_now_str = '{:%d.%m_%H:%M}'.format(time_now)
    print("\nDone optimtizing {} images; Current time: {}".format(CONFIG['r-and-c']['num_examples'], time_now_str))
    print("Optimization took {}".format(time_now - start_time_main))

    return

class Render_and_compare_optimization():
    def __init__(self, model, config):
        self.model = model
        self.config = config

        if self.config['r-and-c']['print_timing_info']:
            self.average_runtimes = {}
            self.average_runtimes['num_iterations'] = 0
            self.average_runtimes['gradient_descent_step'] = timedelta(0)
            self.average_runtimes['render_single_obj'] = timedelta(0)
            self.average_runtimes['predict_remaining_objs'] = timedelta(0)
            self.average_runtimes['gen_output_from_objs'] = timedelta(0)
            self.average_runtimes['get_loss'] = timedelta(0)

    def create_optimized_dataset(self, data, params, dataset_dir):
        num_scenes = data.get_size()
        _, iterator = get_data_iterator(data, CONFIG['r-and-c'])

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        latents_file = open(dataset_dir + '/latents.npy', 'ab')
        inputs_file = open(dataset_dir + '/inputs.npy', 'ab')

        #Write file with loss weights to the dataset directory:
        with open(dataset_dir + '/loss_weights.txt', 'a') as weights_file:
            for k, v in CONFIG['model']['mosnet']['l-weights'].items():
                weights_file.write("{}: {}\n".format(k, v))

        for i in range(num_scenes):
            print("++++++++++++++++++++++++++++")
            print("Scene #{}/{}".format(i, num_scenes))

            input_scene = next(iterator)
            input_scene = self.model.get_input(input_scene)

            results = self.optimize_latents(input_scene, params, latents_file=latents_file, inputs_file=inputs_file)



    def optimize_latents(self, input, params, latents_file=None, inputs_file=None):

        # for all slots:
        #   Combine the (now final) masks and diff_imgs from the previous slots to get the input for the encoder
        #   Run the encoder with the current mask and diff_img to get initial latent for the slot
        #   Render the current slot and perform inference for the rest of the model
        #   Get initial loss value
        #   while loss not converged:
        #       Gradient descent step on the latent vector of the current slot
        #       Render the current slot from the updated latent and perform inference for the rest of the model
        #       Compute loss
        #   Save the (now final) latent, mask and diff_img

        rgb_in = input['rgb_in']

        res = {k: [] for k in self.model.output_types}
        diffrgb_list = [rgb_in]
        msk_list = [tf.expand_dims(tf.zeros_like(rgb_in), axis=1)[..., :1]]
        z_obj_list = []

        for s in range(self.model.n_slots):
            if self.config['r-and-c']['print_loss_vals']: print("Optimizing slot ", s)
            start_time_slot = datetime.now()

            # Combine the (now final) masks and diff_imgs from the previous slots to get the input for the encoder:
            msk_all = tf.concat(msk_list, axis=1)
            msk_all = tf.reduce_max(msk_all, axis=1)
            slot_input = tf.concat([rgb_in, diffrgb_list[-1], msk_all], axis=-1)


            if not inputs_file is None:
                np.save(inputs_file, slot_input)
                print("Saved slot inputs to {}".format(inputs_file.name))

            # Run the encoder with the current mask and diff_img to get initial latent for the slot:
            #z_obj_cur = self.model.encoder_obj([slot_input], training=True)['z_mean']
            z_obj_cur = self.model.gen_latents_single_slot(slot_input, rgb_in, res, msk_list, diffrgb_list, training=True)['z_mean']

            if self.config['r-and-c']['print_loss_vals']: print("Optimizing latent for slot {} from encoder output:".format(s))
            opt_enc_obj = self.optimize_single_latent(z_obj_cur, input, params, self.config['r-and-c']["smoothing_kernels_enc"], res, diffrgb_list, msk_list, z_obj_list, s, rgb_in)
            if self.config['r-and-c']['print_loss_vals']: print("Done optimizing latent for slot {} from encoder output".format(s))

            z_obj_cur_shape_var = opt_enc_obj["z_obj_cur"]["z_obj_cur_shape_var"]
            z_obj_cur_tex_var   = opt_enc_obj["z_obj_cur"]["z_obj_cur_tex_var"]
            z_obj_cur_ext_var   = opt_enc_obj["z_obj_cur"]["z_obj_cur_ext_var"]
            losses = opt_enc_obj["losses"]
            loss_total = opt_enc_obj["losses"]["loss_total"]

            if self.config['r-and-c']['optimize_rand_obj']:
                if loss_total > self.config['r-and-c']['loss_thresh']:
                    z_len = self.model.dim_shape + self.model.dim_app + self.model.dim_extr
                    z_obj_cur = tf.random.uniform(shape=(1, 1, z_len), minval=-1e-10, maxval=1e-10, dtype=tf.dtypes.float32, seed=self.config['r-and-c']['rand_seed'])

                    if self.config['r-and-c']['print_loss_vals']: print("Loss with optimized encoder output too high, Optimizing latent for slot {} from random:".format(s))
                    opt_rand_obj = self.optimize_single_latent(z_obj_cur, input, params, self.config['r-and-c']["smoothing_kernels_rand"], res, diffrgb_list, msk_list, z_obj_list, s, rgb_in)
                    if self.config['r-and-c']['print_loss_vals']: print("Done optimizing latent for slot {} from random".format(s))
                    if (opt_rand_obj["losses"]["loss_total"] < loss_total):
                        if self.config['r-and-c']['print_loss_vals']: print("The optimization from a random vector resulted in a lower loss")
                        z_obj_cur_shape_var = opt_rand_obj["z_obj_cur"]["z_obj_cur_shape_var"]
                        z_obj_cur_tex_var   = opt_rand_obj["z_obj_cur"]["z_obj_cur_tex_var"]
                        z_obj_cur_ext_var   = opt_rand_obj["z_obj_cur"]["z_obj_cur_ext_var"]
                        losses = opt_rand_obj["losses"]
                        loss_total = opt_rand_obj["losses"]["loss_total"]

            obj_res = self.render_single_obj(z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var)

            #Save the (now final) latent, mask and diff_img for the slot
            z_obj_cur = tf.concat([z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var], axis=2)
            self.add_new_slot_results(res, z_obj_list, diffrgb_list, msk_list, z_obj_cur, obj_res, rgb_in)

            if not latents_file is None:
                np.save(latents_file, z_obj_cur)
                print("Saved latents to {}".format(latents_file.name))

            if self.config['r-and-c']['print_timing_info']: print("Total time spent on the slot: {}".format(datetime.now()-start_time_slot))
            if self.config['r-and-c']['print_loss_vals']: print("Slot {} optimized\n\n".format(s))

        # Manipulation of optimized objects:
        # if True:
        #     z_obj = z_obj_list[0].numpy()
        #     #z_obj[0, 0, -2:] = np.array([math.sin(math.asin(z_obj[0, 0, -2])+math.pi/4), math.cos(math.acos(z_obj[0, 0, -1])+math.pi/4)]) #Orientation
        #     #z_obj[0, 0, -5:-2] = z_obj[0, 0, -5:-2]+0.2*np.array([0, 0, 1])#Position
        #     z_obj[0, 0, -6] = z_obj[0, 0, -6]*2. #Scale
        #     z_obj_list[0] = tf.Variable(z_obj)
        #
        #
        #     z_obj_shape_var = tf.Variable(z_obj[:, :, :8])
        #     z_obj_tex_var = tf.Variable(z_obj[:, :, 8:16])
        #     z_obj_ext_var = tf.Variable(z_obj[:, :, 16:])
        #     obj_res = self.render_single_obj(z_obj_shape_var, z_obj_tex_var, z_obj_ext_var)
        #     for k, v in obj_res.items():
        #         if not k == "obj_msk_soft_pred":
        #             res[k][0] = copy.deepcopy(v)

        for k, v in res.items():
            res[k] = K.stack(v, axis=1)
        z_obj_all = K.stack(z_obj_list, axis=1)  # (BS, N_slots, N_img, D)
        z_dict_tmp = {
            'z_shape': z_obj_all[..., :self.model.dim_shape],                                    # (BS, S, N_img, D_shape)
            'z_tex': z_obj_all[..., self.model.dim_shape:self.model.dim_shape + self.model.dim_app - 1],   # (BS, S, N_img, D_tex)
            'z_s': res['z_extr'][..., :1],                                                  # (BS, S, N_img, D_s=1)
            'z_pos': res['z_extr'][..., 1:4],                                               # (BS, S, N_img, D_pos=3)
            'z_pos_x': res['z_extr'][..., 1:2],                                             # (BS, S, N_img, 1)
            'z_pos_y': res['z_extr'][..., 2:3],                                             # (BS, S, N_img, 1)
            'z_pos_z': res['z_extr'][..., 3:4],                                             # (BS, S, N_img, 1)
            'z_rot': res['z_extr'][..., 4:],                                                # (BS, S, N_img, D_rot=1)
        }
        for k, v in z_dict_tmp.items():
            res[k + '_mean'] = v

        res['z_all'] = z_obj_all

        if self.config['r-and-c']['post-processing']:
            for slot in range(self.model.n_slots-1, -1, -1):
                avg_color  = tf.reduce_sum(res['obj_rgb_pred'][0, slot,  0, ...]*res['obj_msk_pred'][0, slot,  0, ...], axis=[0, 1]) / tf.reduce_sum(res['obj_msk_pred'][0, slot,  0, ...])
                for slot_ in range(slot):
                    l2_distance = tf.nn.l2_loss(res['z_pos_mean'][0, slot, 0, :] - res['z_pos_mean'][0, slot_, 0, :])

                    avg_color_ = tf.reduce_sum(res['obj_rgb_pred'][0, slot_, 0, ...]*res['obj_msk_pred'][0, slot_, 0, ...], axis=[0, 1]) / tf.reduce_sum(res['obj_msk_pred'][0, slot_, 0, ...])
                    color_l2_distancce = tf.nn.l2_loss(avg_color, avg_color_)

                    num_obj_pixels = tf.reduce_sum(res['obj_msk_pred'][0, slot,  0, ...])
                    num_visible_pixels = num_obj_pixels - tf.reduce_sum(res['obj_msk_pred'][0, slot,  0, ...] * res['obj_msk_pred'][0, slot_,  0, ...])

                    # If the object in slot_ is 1) too close, 2) too similar in avg color, and 3) covers too much of the object in slot, then don't render slot
                    if      (l2_distance < 2. and color_l2_distancce < 0.3 and num_visible_pixels/num_obj_pixels < 0.2) \
                        or  num_visible_pixels == 0:

                        res['obj_msk_pred'] = res['obj_msk_pred'].numpy()
                        res['obj_msk_pred'][0, slot,  0, ...] = tf.zeros_like(res['obj_msk_pred'][0, slot_, 0, ...])
                        res['obj_msk_pred'] = tf.Variable(res['obj_msk_pred'])

                        res['obj_rgb_pred'] = res['obj_rgb_pred'].numpy()
                        res['obj_rgb_pred'][0, slot,  0, ...] = tf.zeros_like(res['obj_rgb_pred'][0, slot_, 0, ...])
                        res['obj_rgb_pred'] = tf.Variable(res['obj_rgb_pred'])

                        res['obj_depth_pred'] = res['obj_depth_pred'].numpy()
                        res['obj_depth_pred'][0, slot,  0, ...] = self.model._bg['depth']
                        res['obj_depth_pred'] = tf.Variable(res['obj_depth_pred'])

        bg_img = self.model.predict_bg(input['rgb_in'])
        depth_ordering = self.model.depth_ordering(res, bg_img)
        for k, v in depth_ordering.items():
            res[k] = v

        if self.model.img_size_render != self.model.img_size:
            res['rgb_pred_full_size'] = copy.deepcopy(res['rgb_pred'])
            res['depth_pred_full_size'] = copy.deepcopy(res['depth_pred'])
            res['msk_pred_full_size'] = copy.deepcopy(res['msk_pred'])
            res['obj_rgb_pred_full_size'] = copy.deepcopy(res['obj_rgb_pred'])
            res['obj_depth_pred_full_size'] = copy.deepcopy(res['obj_depth_pred'])
            res['obj_msk_pred_full_size'] = copy.deepcopy(res['obj_msk_pred'])
            rgb_pred = self.model.downsample_render(res['rgb_pred'], antialiasing=True)
            depth_pred = self.model.downsample_render(res['depth_pred'], antialiasing=True)
            msk_pred = self.model.downsample_render(res['msk_pred'], antialiasing=True)
            obj_rgb_pred = self.model.downsample_render(res['obj_rgb_pred'], antialiasing=True)
            obj_depth_pred = self.model.downsample_render(res['obj_depth_pred'], antialiasing=True)
            obj_msk_pred = self.model.downsample_render(res['obj_msk_pred'], antialiasing=True)
            res['rgb_pred'] = rgb_pred
            res['depth_pred'] = depth_pred
            res['msk_pred'] = msk_pred
            res['obj_rgb_pred'] = obj_rgb_pred
            res['obj_depth_pred'] = obj_depth_pred
            res['obj_msk_pred'] = obj_msk_pred

        results = {}
        results["res"] = res
        results["losses"] = losses

        return results


    @staticmethod
    def find_closest_target_latent(z_obj_cur, z_extr_gt, permutation_list):
        smallest_l2_distance = tf.Variable(np.inf, dtype=tf.float32)
        best_matching_slot = -1
        for i in range(tf.shape(z_extr_gt)[1]):
            if i not in permutation_list:
                l2_distance = tf.nn.l2_loss(z_obj_cur[0, 0, -5:-2] - z_extr_gt[0, i, 0, -4:-1])
                if l2_distance < smallest_l2_distance:
                    smallest_l2_distance = l2_distance
                    best_matching_slot = i
        permutation_list.append(best_matching_slot)
        return z_extr_gt[0, best_matching_slot, 0, :]


    #When using this, the learning rate for the extrinsics should be zero
    def optimize_latents_with_gt_extr(self, input, params):

        # for all slots:
        #   Combine the (now final) masks and diff_imgs from the previous slots to get the input for the encoder
        #   Run the encoder with the current mask and diff_img to get initial latent for the slot
        #   Render the current slot and perform inference for the rest of the model
        #   Get initial loss value
        #   while loss not converged:
        #       Gradient descent step on the latent vector of the current slot
        #       Render the current slot from the updated latent and perform inference for the rest of the model
        #       Compute loss
        #   Save the (now final) latent, mask and diff_img

        rgb_in = input['rgb_in']

        res = {k: [] for k in self.model.output_types}
        diffrgb_list = [rgb_in]
        msk_list = [tf.expand_dims(tf.zeros_like(rgb_in), axis=1)[..., :1]]
        z_obj_list = []

        permutation_list = []

        for s in range(self.model.n_slots):
            if self.config['r-and-c']['print_loss_vals']: print("Optimizing slot ", s)
            start_time_slot = datetime.now()

            # Combine the (now final) masks and diff_imgs from the previous slots to get the input for the encoder:
            msk_all = tf.concat(msk_list, axis=1)
            msk_all = tf.reduce_max(msk_all, axis=1)
            slot_input = tf.concat([rgb_in, diffrgb_list[-1], msk_all], axis=-1)

            # Run the encoder with the current mask and diff_img to get initial latent for the slot:
            #z_obj_cur = self.model.encoder_obj([slot_input], training=True)['z_mean']
            z_obj_cur = self.model.gen_latents_single_slot(slot_input, rgb_in, res, msk_list, diffrgb_list, training=True)['z_mean'].numpy()
            target_latent = self.find_closest_target_latent(z_obj_cur, input['z_extr_gt'].numpy(), permutation_list)
            #print(z_obj_cur)
            z_obj_cur[0, 0, -5:-2] = target_latent[-4:-1]
            z_obj_cur[0, 0, -2:] = np.array([math.sin(target_latent[-1]), math.cos(target_latent[-1])])
            z_obj_cur = tf.Variable(z_obj_cur)
            #print(z_obj_cur.numpy())



            if self.config['r-and-c']['print_loss_vals']: print("Optimizing latent for slot {} from encoder output:".format(s))
            opt_enc_obj = self.optimize_single_latent(z_obj_cur, input, params, self.config['r-and-c']["smoothing_kernels_enc"], res, diffrgb_list, msk_list, z_obj_list, s, rgb_in)
            if self.config['r-and-c']['print_loss_vals']: print("Done optimizing latent for slot {} from encoder output".format(s))

            z_obj_cur_shape_var = opt_enc_obj["z_obj_cur"]["z_obj_cur_shape_var"]
            z_obj_cur_tex_var   = opt_enc_obj["z_obj_cur"]["z_obj_cur_tex_var"]
            z_obj_cur_ext_var   = opt_enc_obj["z_obj_cur"]["z_obj_cur_ext_var"]
            losses = opt_enc_obj["losses"]
            loss_total = opt_enc_obj["losses"]["loss_total"]

            if self.config['r-and-c']['optimize_rand_obj']:
                if loss_total > self.config['r-and-c']['loss_thresh']:
                    z_len = self.model.dim_shape + self.model.dim_app + self.model.dim_extr
                    z_obj_cur = tf.random.uniform(shape=(1, 1, z_len), minval=-1e-10, maxval=1e-10, dtype=tf.dtypes.float32, seed=self.config['r-and-c']['rand_seed'])

                    if self.config['r-and-c']['print_loss_vals']: print("Loss with optimized encoder output too high, Optimizing latent for slot {} from random:".format(s))
                    opt_rand_obj = self.optimize_single_latent(z_obj_cur, input, params, self.config['r-and-c']["smoothing_kernels_rand"], res, diffrgb_list, msk_list, z_obj_list, s, rgb_in)
                    if self.config['r-and-c']['print_loss_vals']: print("Done optimizing latent for slot {} from random".format(s))
                    if (opt_rand_obj["losses"]["loss_total"] < loss_total):
                        if self.config['r-and-c']['print_loss_vals']: print("The optimization from a random vector resulted in a lower loss")
                        z_obj_cur_shape_var = opt_rand_obj["z_obj_cur"]["z_obj_cur_shape_var"]
                        z_obj_cur_tex_var   = opt_rand_obj["z_obj_cur"]["z_obj_cur_tex_var"]
                        z_obj_cur_ext_var   = opt_rand_obj["z_obj_cur"]["z_obj_cur_ext_var"]
                        losses = opt_rand_obj["losses"]
                        loss_total = opt_rand_obj["losses"]["loss_total"]


            z_obj_cur_shape_var = tf.Variable(z_obj_cur.numpy()[:, :, :8])
            z_obj_cur_tex_var   = tf.Variable(z_obj_cur.numpy()[:, :, 8:16])
            z_obj_cur_ext_var   = tf.Variable(z_obj_cur.numpy()[:, :, 16:])
            obj_res = self.render_single_obj(z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var)

            #Save the (now final) latent, mask and diff_img for the slot
            z_obj_cur = tf.concat([z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var], axis=2)
            self.add_new_slot_results(res, z_obj_list, diffrgb_list, msk_list, z_obj_cur, obj_res, rgb_in)

            if self.config['r-and-c']['print_timing_info']: print("Total time spent on the slot: {}".format(datetime.now()-start_time_slot))
            if self.config['r-and-c']['print_loss_vals']: print("Slot {} optimized\n\n".format(s))

        for k, v in res.items():
            res[k] = K.stack(v, axis=1)
        z_obj_all = K.stack(z_obj_list, axis=1)  # (BS, N_slots, N_img, D)
        z_dict_tmp = {
            'z_shape': z_obj_all[..., :self.model.dim_shape],                                    # (BS, S, N_img, D_shape)
            'z_tex': z_obj_all[..., self.model.dim_shape:self.model.dim_shape + self.model.dim_app - 1],   # (BS, S, N_img, D_tex)
            'z_s': res['z_extr'][..., :1],                                                  # (BS, S, N_img, D_s=1)
            'z_pos': res['z_extr'][..., 1:4],                                               # (BS, S, N_img, D_pos=3)
            'z_pos_x': res['z_extr'][..., 1:2],                                             # (BS, S, N_img, 1)
            'z_pos_y': res['z_extr'][..., 2:3],                                             # (BS, S, N_img, 1)
            'z_pos_z': res['z_extr'][..., 3:4],                                             # (BS, S, N_img, 1)
            'z_rot': res['z_extr'][..., 4:],                                                # (BS, S, N_img, D_rot=1)
        }
        for k, v in z_dict_tmp.items():
            res[k + '_mean'] = v

        res['z_all'] = z_obj_all
        bg_img = self.model.predict_bg(input['rgb_in'])
        depth_ordering = self.model.depth_ordering(res, bg_img)
        for k, v in depth_ordering.items():
            res[k] = v

        results = {}
        results["res"] = res
        results["losses"] = losses

        return results

    # def optimize_latents_with_gt_extr(self, input, params):
    #
    #     # for all slots:
    #     #   Combine the (now final) masks and diff_imgs from the previous slots to get the input for the encoder
    #     #   Run the encoder with the current mask and diff_img to get initial latent for the slot
    #     #   Render the current slot and perform inference for the rest of the model
    #     #   Get initial loss value
    #     #   while loss not converged:
    #     #       Gradient descent step on the latent vector of the current slot
    #     #       Render the current slot from the updated latent and perform inference for the rest of the model
    #     #       Compute loss
    #     #   Save the (now final) latent, mask and diff_img
    #
    #     rgb_in = input['rgb_in']
    #
    #     res = {k: [] for k in self.model.output_types}
    #     diffrgb_list = [rgb_in]
    #     msk_list = [tf.expand_dims(tf.zeros_like(rgb_in), axis=1)[..., :1]]
    #     z_obj_list = []
    #
    #     permutations = list(itertools.permutations([0, 1, 2]))
    #     lowest_loss = tf.Variable(np.inf, dtype=tf.float32)
    #     for permutation in permutations:
    #
    #         res_ = {k: [] for k in self.model.output_types}
    #         diffrgb_list_ = [rgb_in]
    #         msk_list_ = [tf.expand_dims(tf.zeros_like(rgb_in), axis=1)[..., :1]]
    #         z_obj_list_ = []
    #
    #         for s in range(self.model.n_slots):
    #             if self.config['r-and-c']['print_loss_vals']: print("Optimizing slot ", s)
    #             start_time_slot = datetime.now()
    #
    #             # Combine the (now final) masks and diff_imgs from the previous slots to get the input for the encoder:
    #             msk_all = tf.concat(msk_list_, axis=1)
    #             msk_all = tf.reduce_max(msk_all, axis=1)
    #             slot_input = tf.concat([rgb_in, diffrgb_list_[-1], msk_all], axis=-1)
    #
    #             # Run the encoder with the current mask and diff_img to get initial latent for the slot:
    #             #z_obj_cur = self.model.encoder_obj([slot_input], training=True)['z_mean']
    #             z_obj_cur = self.model.gen_latents_single_slot(slot_input, rgb_in, res_, msk_list_, diffrgb_list_, training=True)['z_mean'].numpy()
    #             #z_obj_cur[:, :, -2:] = input['z_extr_gt'][0, list(permutation)[s], 0, -2:]
    #             z_obj_cur[0, 0, -5:-2] = input['z_extr_gt'][0, list(permutation)[s], 0, -4:-1]
    #             z_obj_cur = tf.Variable(z_obj_cur)
    #
    #             render = self.model.render_single_obj(z_obj_cur)
    #             obj_rgb_pred = (np.transpose(render['obj_rgb_pred'][0, ...], (1, 0, 2, 3))).reshape((64, 64, 3))
    #             viz.show_image(obj_rgb_pred, './obj_rgb_pred.png')
    #
    #
    #             if self.config['r-and-c']['print_loss_vals']: print("Optimizing latent for slot {} from encoder output:".format(s))
    #             opt_enc_obj = self.optimize_single_latent(z_obj_cur, input, params, self.config['r-and-c']["smoothing_kernels_enc"], res_, diffrgb_list_, msk_list_, z_obj_list_, s, rgb_in)
    #             if self.config['r-and-c']['print_loss_vals']: print("Done optimizing latent for slot {} from encoder output".format(s))
    #
    #             z_obj_cur_shape_var = opt_enc_obj["z_obj_cur"]["z_obj_cur_shape_var"]
    #             z_obj_cur_tex_var   = opt_enc_obj["z_obj_cur"]["z_obj_cur_tex_var"]
    #             z_obj_cur_ext_var   = opt_enc_obj["z_obj_cur"]["z_obj_cur_ext_var"]
    #             losses = opt_enc_obj["losses"]
    #             loss_total = opt_enc_obj["losses"]["loss_total"]
    #
    #             if self.config['r-and-c']['optimize_rand_obj']:
    #                 if loss_total > self.config['r-and-c']['loss_thresh']:
    #                     z_len = self.model.dim_shape + self.model.dim_app + self.model.dim_extr
    #                     z_obj_cur = tf.random.uniform(shape=(1, 1, z_len), minval=-1e-10, maxval=1e-10, dtype=tf.dtypes.float32, seed=self.config['r-and-c']['rand_seed'])
    #
    #                     if self.config['r-and-c']['print_loss_vals']: print("Loss with optimized encoder output too high, Optimizing latent for slot {} from random:".format(s))
    #                     opt_rand_obj = self.optimize_single_latent(z_obj_cur, input, params, self.config['r-and-c']["smoothing_kernels_rand"], res_, diffrgb_list_, msk_list_, z_obj_list_, s, rgb_in)
    #                     if self.config['r-and-c']['print_loss_vals']: print("Done optimizing latent for slot {} from random".format(s))
    #                     if (opt_rand_obj["losses"]["loss_total"] < loss_total):
    #                         if self.config['r-and-c']['print_loss_vals']: print("The optimization from a random vector resulted in a lower loss")
    #                         z_obj_cur_shape_var = opt_rand_obj["z_obj_cur"]["z_obj_cur_shape_var"]
    #                         z_obj_cur_tex_var   = opt_rand_obj["z_obj_cur"]["z_obj_cur_tex_var"]
    #                         z_obj_cur_ext_var   = opt_rand_obj["z_obj_cur"]["z_obj_cur_ext_var"]
    #                         losses = opt_rand_obj["losses"]
    #                         loss_total = opt_rand_obj["losses"]["loss_total"]
    #
    #             obj_res = self.render_single_obj(z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var)
    #
    #             # if self.config["r-and-c"]["rand_positions"]:
    #             #     #Turn each object 180 degrees:
    #             #     z_obj_cur_ext_var = tf.constant(z_obj_cur_ext_var) #[pos_x, pos_y, pos_z, rot_cos, rot_sin]
    #             #     new_pos = tf.constant(z_obj_cur_ext_var[..., 0:3]) + tf.constant([0., 0., 0.], shape=(1, 1, 3))
    #             #
    #             #     angle = tf.math.atan2(z_obj_cur_ext_var[..., self.model.dim_extr-1], z_obj_cur_ext_var[..., self.model.dim_extr-2])
    #             #     new_orientation = tf.constant(angle, shape=(1, 1, 1))
    #             #     #new_orientation = tf.concat([z_obj_cur_ext_var[..., self.model.dim_extr-2:self.model.dim_extr-1], z_obj_cur_ext_var[..., self.model.dim_extr-1:]], axis=2)
    #             #
    #             #     #new_scale = tf.constant(z_obj_cur_ext_var[..., 0], shape=(1, 1, 1))
    #             #
    #             #     new_ext = tf.concat([new_pos, new_orientation], axis=2)
    #             #     obj_res = render_single_obj(z_obj_cur_shape_var, z_obj_cur_tex_var, new_ext)
    #
    #             #Save the (now final) latent, mask and diff_img for the slot
    #             z_obj_cur = tf.concat([z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var], axis=2)
    #             self.add_new_slot_results(res_, z_obj_list_, diffrgb_list_, msk_list_, z_obj_cur, obj_res, rgb_in)
    #
    #             if self.config['r-and-c']['print_timing_info']: print("Total time spent on the slot: {}".format(datetime.now()-start_time_slot))
    #             if self.config['r-and-c']['print_loss_vals']: print("Slot {} optimized\n\n".format(s))
    #
    #         if losses['loss_total'] < lowest_loss:
    #             res = res_
    #             diffrgb_list = diffrgb_list_
    #             msk_list = msk_list_
    #             z_obj_list = z_obj_list_
    #             best_permutation = permutation
    #
    #     print("Best permutation: {}".format(best_permutation))
    #
    #     for k, v in res.items():
    #         res[k] = K.stack(v, axis=1)
    #     z_obj_all = K.stack(z_obj_list, axis=1)  # (BS, N_slots, N_img, D)
    #     z_dict_tmp = {
    #         'z_shape': z_obj_all[..., :self.model.dim_shape],                                    # (BS, S, N_img, D_shape)
    #         'z_tex': z_obj_all[..., self.model.dim_shape:self.model.dim_shape + self.model.dim_app - 1],   # (BS, S, N_img, D_tex)
    #         'z_s': res['z_extr'][..., :1],                                                  # (BS, S, N_img, D_s=1)
    #         'z_pos': res['z_extr'][..., 1:4],                                               # (BS, S, N_img, D_pos=3)
    #         'z_pos_x': res['z_extr'][..., 1:2],                                             # (BS, S, N_img, 1)
    #         'z_pos_y': res['z_extr'][..., 2:3],                                             # (BS, S, N_img, 1)
    #         'z_pos_z': res['z_extr'][..., 3:4],                                             # (BS, S, N_img, 1)
    #         'z_rot': res['z_extr'][..., 4:],                                                # (BS, S, N_img, D_rot=1)
    #     }
    #     for k, v in z_dict_tmp.items():
    #         res[k + '_mean'] = v
    #
    #     res['z_all'] = z_obj_all
    #     bg_img = self.model.predict_bg(input['rgb_in'])
    #     depth_ordering = self.model.depth_ordering(res, bg_img)
    #     for k, v in depth_ordering.items():
    #         res[k] = v
    #
    #     results = {}
    #     results["res"] = res
    #     results["losses"] = losses
    #
    #     return results

    def render_single_obj(self, z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var):
        z_obj_cur = tf.concat([z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var], 2)
        return self.model.render_single_obj(z_obj_cur)

    def gen_output_from_objs(self, res_obj, rgb_in):
        bg_img = self.model.predict_bg(rgb_in)

        combined_res = self.model.depth_ordering(res_obj, bg_img)
        res = res_obj
        res['bg_img'] = bg_img
        for k, v in combined_res.items():
            res[k] = v

        res = self.model.crit_obj_intersection(res)
        res = self.model.crit_ground(res)

        return res

    def predict_remaining_objs(self, res, obj_res, diffrgb_list, msk_list, z_obj_list, z_obj_cur, s, rgb_in):
        # Since the results for the current and the subsequent slots are not final we dont add them to the final data
        # structures and instead add them to these ephemeral ones
        res_ = copy.deepcopy(res)
        diffrgb_list_ = copy.deepcopy(diffrgb_list)
        msk_list_ = copy.deepcopy(msk_list)
        z_obj_list_ = copy.deepcopy(z_obj_list)

        # Update the lists and res_ to include the current slot
        self.add_new_slot_results(res_, z_obj_list_, diffrgb_list_, msk_list_, z_obj_cur, obj_res, rgb_in)

        # Infer remaining latents and render all objects:
        for s_ in range(s+1, self.model.n_slots):
            msk_all = tf.concat(msk_list_, axis=1)
            msk_all = tf.reduce_max(msk_all, axis=1)
            input = tf.concat([rgb_in, diffrgb_list_[-1], msk_all], axis=-1)
            #z_obj_cur = self.model.encoder_obj([input], training=True)['z_mean']
            z_obj_cur = self.model.gen_latents_single_slot(input, rgb_in, res_, msk_list_, diffrgb_list_, training=True)['z_mean']
            obj_res = self.model.render_single_obj(z_obj_cur)
            self.add_new_slot_results(res_, z_obj_list_, diffrgb_list_, msk_list_, z_obj_cur, obj_res, rgb_in)

        # Final reconstruction
        for k, v in res_.items():
            res_[k] = K.stack(v, axis=1)

        # Latents of all objects
        z_obj_all = K.stack(z_obj_list_, axis=1)  # (BS, N_slots, N_img, D)

        z_dict_tmp = {
            'z_shape': z_obj_all[..., :self.model.dim_shape],                                    # (BS, S, N_img, D_shape)
            'z_tex': z_obj_all[..., self.model.dim_shape:self.model.dim_shape + self.model.dim_app - 1],   # (BS, S, N_img, D_tex)
            'z_s': res_['z_extr'][..., :1],                                                 # (BS, S, N_img, D_s=1)
            'z_pos': res_['z_extr'][..., 1:4],                                              # (BS, S, N_img, D_pos=3)
            'z_pos_x': res_['z_extr'][..., 1:2],                                            # (BS, S, N_img, 1)
            'z_pos_y': res_['z_extr'][..., 2:3],                                            # (BS, S, N_img, 1)
            'z_pos_z': res_['z_extr'][..., 3:4],                                            # (BS, S, N_img, 1)
            'z_rot': res_['z_extr'][..., 4:],                                               # (BS, S, N_img, D_rot=1)
        }
        for k, v in z_dict_tmp.items():
            res_[k + '_mean'] = v

        return res_

    def add_new_slot_results(self, res, z_obj_list, diffrgb_list, msk_list, z_obj_cur, obj_res, rgb_in):
        z_obj_list.append(z_obj_cur)

        res_cur = {}
        for k in self.model.output_types:
            if k in obj_res:
                res[k].append(obj_res[k])
                res_cur[k] = tf.stack(res[k], axis=1)

        combined_res = self.model.depth_ordering(res_cur, tf.zeros_like(res_cur['obj_rgb_pred'][:, 0]))
        diffrgb_list.append(rgb_in - self.model.downsample_render(combined_res['rgb_pred'], antialiasing=True))
        msk_list.append(self.model.downsample_render(combined_res['msk_pred'], antialiasing=True))

        return

    def optimize_single_latent(self, initial_latent, input, params, kernels, res, diffrgb_list, msk_list, z_obj_list, s, rgb_in):
        z_obj_cur = initial_latent

        optimizer_shape = tf.keras.optimizers.Adam(learning_rate=self.config['r-and-c']['learning_rate_shape'])
        optimizer_tex = tf.keras.optimizers.Adam(learning_rate=self.config['r-and-c']['learning_rate_tex'])
        optimizer_ext = tf.keras.optimizers.Adam(learning_rate=self.config['r-and-c']['learning_rate_ext'])

        if self.config['r-and-c']['optimization_order'][0] == 'all' and len(self.config['r-and-c']['optimization_order']) != 1:
            print("[ERROR]: The optimization order must be either a permutation of ['shape', 'tex', 'ext'] or ['all'] to optimize the entire latent at once.")
            exit(1)
        if (self.config['r-and-c']['optimization_order'][0] == 'shape' or self.config['r-and-c']['optimization_order'][0] == 'tex' or self.config['r-and-c']['optimization_order'][0] == 'ext') and len(self.config['r-and-c']['optimization_order']) != 3:
            print("[ERROR]: The optimization order must be either a permutation of ['shape', 'tex', 'ext'] or ['all'] to optimize the entire latent at once.")
            exit(1)

        for kernel in kernels:
            for sub_latent in self.config['r-and-c']['optimization_order']:
                if self.config['r-and-c']['print_loss_vals']: print("Optimizing with kernel size {} ...".format(kernel))
                self.model.gauss_kernel = kernel
                params['gauss_sigma'] = tf.constant(kernel/3.)

                z_obj_cur_shape = tf.constant(z_obj_cur[..., :self.model.dim_shape])
                z_obj_cur_shape_var = tf.Variable(initial_value=z_obj_cur_shape, trainable=True)
                z_obj_cur_tex = tf.constant(z_obj_cur[..., self.model.dim_shape:self.model.dim_shape+self.model.dim_app])
                z_obj_cur_tex_var = tf.Variable(initial_value=z_obj_cur_tex, trainable=True)
                z_obj_cur_ext = tf.constant(z_obj_cur[..., self.model.dim_shape+self.model.dim_app:])
                z_obj_cur_ext_var = tf.Variable(initial_value=z_obj_cur_ext, trainable=True)

                #The tape needs to be persistent as we want to compute 3 sets of gradients, not just one
                with tf.GradientTape(persistent=True) as tape:
                    # Render the current slot and perform inference for the rest of the model
                    obj_res = self.render_single_obj(z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var)

                    res_obj = self.predict_remaining_objs(res, obj_res , diffrgb_list, msk_list, z_obj_list, tf.concat([z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var], 2), s, rgb_in)
                    output = self.gen_output_from_objs(res_obj, rgb_in)

                    if self.model.img_size_render != self.model.img_size:
                        output['rgb_pred_full_size'] = copy.deepcopy(output['rgb_pred'])
                        output['depth_pred_full_size'] = copy.deepcopy(output['depth_pred'])
                        rgb_pred = self.model.downsample_render(output['rgb_pred'], antialiasing=True)
                        depth_pred = self.model.downsample_render(output['depth_pred'], antialiasing=True)
                        output['rgb_pred'] = rgb_pred
                        output['depth_pred'] = depth_pred

                    # Get initial loss value:
                    losses = self.model.get_loss(output, input, params)
                    loss_total = losses['loss_total']
                    if self.config['r-and-c']['print_loss_vals']: print("loss_total before slot optimization = {}".format(loss_total))

                prev_loss = tf.Variable(np.inf, dtype=tf.float32)
                num_iterations = 0
                if kernel == 0.:
                    eps = self.config['r-and-c']['eps_final']
                else:
                    eps = self.config['r-and-c']['eps']
                while (prev_loss - loss_total > eps):
                    num_iterations += 1
                    prev_loss = loss_total

                    start_time = datetime.now()

                    #Gradient descent step on the latent vector of the current slot
                    if sub_latent=='shape':
                        print("Applying shape update")
                        gradients_shape = tape.gradient(loss_total, [z_obj_cur_shape_var])
                        optimizer_shape.apply_gradients(zip(gradients_shape, [z_obj_cur_shape_var]))
                    elif sub_latent=='tex':
                        print("Applying tex update")
                        gradients_tex = tape.gradient(loss_total, [z_obj_cur_tex_var])
                        optimizer_tex.apply_gradients(zip(gradients_tex, [z_obj_cur_tex_var]))
                    elif sub_latent=='ext':
                        print("Applying ext update")
                        gradients_ext = tape.gradient(loss_total, [z_obj_cur_ext_var])
                        optimizer_ext.apply_gradients(zip(gradients_ext, [z_obj_cur_ext_var]))
                    elif sub_latent=='all':
                        gradients_shape = tape.gradient(loss_total, [z_obj_cur_shape_var])
                        optimizer_shape.apply_gradients(zip(gradients_shape, [z_obj_cur_shape_var]))
                        gradients_tex = tape.gradient(loss_total, [z_obj_cur_tex_var])
                        optimizer_tex.apply_gradients(zip(gradients_tex, [z_obj_cur_tex_var]))
                        gradients_ext = tape.gradient(loss_total, [z_obj_cur_ext_var])
                        optimizer_ext.apply_gradients(zip(gradients_ext, [z_obj_cur_ext_var]))
                    else:
                        print("[ERROR]: The optimization order must be either a permutation of ['shape', 'tex', 'ext'] or ['all'] to optimize the entire latent at once.")
                        exit(1)

                    if self.config['r-and-c']['print_timing_info']:
                        self.average_runtimes['num_iterations'] += 1
                        self.average_runtimes['gradient_descent_step'] += (datetime.now()-start_time)

                    with tf.GradientTape(persistent=True) as tape:
                        # Render the current slot from the updated latent and perform inference for the rest of the model
                        start_time = datetime.now()
                        obj_res = self.render_single_obj(z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var)
                        if self.config['r-and-c']['print_timing_info']:
                            self.average_runtimes['render_single_obj'] += (datetime.now()-start_time)

                        start_time = datetime.now()
                        res_obj = self.predict_remaining_objs(res, obj_res , diffrgb_list, msk_list, z_obj_list, tf.concat([z_obj_cur_shape_var, z_obj_cur_tex_var, z_obj_cur_ext_var], 2), s, rgb_in)
                        if self.config['r-and-c']['print_timing_info']:
                            self.average_runtimes['predict_remaining_objs'] += (datetime.now()-start_time)

                        start_time = datetime.now()
                        output = self.gen_output_from_objs(res_obj, rgb_in)
                        if self.config['r-and-c']['print_timing_info']:
                            self.average_runtimes['gen_output_from_objs'] += (datetime.now()-start_time)

                        # Compute loss
                        start_time = datetime.now()
                        if self.model.img_size_render != self.model.img_size:
                            output['rgb_pred_full_size'] = copy.deepcopy(output['rgb_pred'])
                            output['depth_pred_full_size'] = copy.deepcopy(output['depth_pred'])
                            rgb_pred = self.model.downsample_render(output['rgb_pred'], antialiasing=True)
                            depth_pred = self.model.downsample_render(output['depth_pred'], antialiasing=True)
                            output['rgb_pred'] = rgb_pred
                            output['depth_pred'] = depth_pred
                        losses = self.model.get_loss(output, input, params)
                        if self.config['r-and-c']['print_timing_info']:
                            self.average_runtimes['get_loss'] += (datetime.now()-start_time)

                        loss_total = losses['loss_total']

                    if self.config['r-and-c']['print_loss_vals']: print("loss_total = {}".format(loss_total))
                    if num_iterations > self.config['r-and-c']['max_num_iterations']:
                        if self.config['r-and-c']['print_loss_vals']:
                            print("Exceeded the maximum number of iterations, breaking from the optimization loop")
                        break

        res = {}
        res["z_obj_cur"] = {}
        res["z_obj_cur"]["z_obj_cur_shape_var"] = z_obj_cur_shape_var
        res["z_obj_cur"]["z_obj_cur_tex_var"]   = z_obj_cur_tex_var
        res["z_obj_cur"]["z_obj_cur_ext_var"]   = z_obj_cur_ext_var
        res["losses"] = losses

        if self.config['r-and-c']['print_loss_vals']: print("Finished optimizing slot {}".format(s))
        return res

if __name__ == "__main__":

    FLAGS = parse_arguments()

    if FLAGS.message == "":
        print('Need to specify message/ name for  model that should be evaluated[--message]')
        exit()
    model_base_name, model_ext_name = FLAGS.model.split('-')

    if FLAGS.config == "":
        FLAGS.config = 'cnfg_' + model_base_name + '_' + FLAGS.dataset

    if FLAGS.no_gpu == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    LOG_DIR = os.path.join(FLAGS.log_dir, FLAGS.dataset + '_' + FLAGS.model + '(' + FLAGS.message + ')')
    if not os.path.exists(os.path.join(LOG_DIR, 'ckpts', 'checkpoint')):
        print('No pre-trained model found at ', LOG_DIR)
        exit()

    # Config parameters:
    cnfg_file = os.path.join(LOG_DIR, FLAGS.config + '.pkl')
    CONFIG = load_config(cnfg_file)

    if 'anti_aliasing' not in CONFIG['model']['mosnet']:
        CONFIG['model']['mosnet']['anti_aliasing'] = False

    CONFIG['r-and-c'] = {}                              # Holds the config parameters specific to the render-and-compare optimization
    for k, v in CONFIG['training'].items():
        CONFIG['r-and-c'][k] = v

    CONFIG['r-and-c']['post-processing'] = False        # Whether to delete superfluous object slots in post-processing
    CONFIG['r-and-c']['produce_dataset'] = False        # Whether to produce an optimized pseudo-ground-truth dataset
                                                        # based on the training set to train a model in a supervised manner
    CONFIG['r-and-c']['loss_eval'] = True               # Loss Evaluation;  Only if produce_dataset = False

    CONFIG['r-and-c']['batch_size'] = 1                 # Optimize one image at a time; Since each object needs its
                                                        # individual number of GD steps this can not be more
    if CONFIG['r-and-c']['produce_dataset']:
        CONFIG['r-and-c']['learning_rate_shape'] = 0.01     # Learning rate for the shape, can only reasonably be
                                                            # optimized with GT depth and normal data, otherwise, the
                                                            # objects will be smoothed towards the mean shape
    else:
        CONFIG['r-and-c']['learning_rate_shape'] = 0.
    CONFIG['r-and-c']['learning_rate_tex'] = 0.001      # Learning rate for the texture
    CONFIG['r-and-c']['learning_rate_ext'] = 0.01       # Learning rate for the extrinsics
    CONFIG['r-and-c']['eps_final'] = 0.0000000001       # Convergence parameter for final output
    CONFIG['r-and-c']['eps'] = 0.0001                   # Convergence parameter for blurred images
    CONFIG['r-and-c']['num_examples'] = 200             # The number of examples in the validation set we want to render and compare for
    CONFIG['r-and-c']['optimize_rand_obj'] = False      # Whether to optimize a slot from random if the encoder output produces too large a loss
    CONFIG['r-and-c']['rand_seed'] = 42                 # Random seed
    CONFIG['r-and-c']['loss_thresh'] = 0.0001           # Loss threshold above which a random vector is optimized instead of the encoder output
    CONFIG['r-and-c']['print_timing_info'] = True       # Whether to print how long the optimization steps took
    CONFIG['r-and-c']['smoothing_kernels_rand'] = [16., 4., 0.]     # The sizes of the smoothing kernels when optimizing from random initialization, in order of usage
    CONFIG['r-and-c']['smoothing_kernels_enc'] = [0.]               # The sizes of the smoothing kernels when optimizing encoder output
    CONFIG['r-and-c']['results_dir'] = FLAGS.results_dir
    CONFIG['r-and-c']['print_loss_vals'] = True
    CONFIG['r-and-c']['max_num_iterations'] = 50        # The maximum number of iterations with the current smoothing kernel
    CONFIG['r-and-c']['optimization_order'] = ['all']   # The order in which the shape, texture and extrinsics parts of the latents are optimized;
                                                        # Can be any permutation of ['ext', 'tex', 'shape'] or ['all'] to optimize all at once


    if CONFIG['r-and-c']['produce_dataset']:
        # To produce an optimized dataset based on the training set, we should use all GT data available. The weights
        # are the same as during training.
        CONFIG['model']['mosnet']['l-weights']['rgb'] = 1.
        CONFIG['model']['mosnet']['l-weights']['depth'] = 0.1
        CONFIG['model']['mosnet']['l-weights']['rgb_sil'] = 4000000.
        CONFIG['model']['mosnet']['l-weights']['depth_sil'] = 50.
        CONFIG['model']['mosnet']['l-weights']['z-reg'] = 0.0025
        CONFIG['model']['mosnet']['l-weights']['extr'] = 0.
        CONFIG['model']['mosnet']['l-weights']['intersect'] = 0.001
        CONFIG['model']['mosnet']['l-weights']['ground'] = 0.01
        CONFIG['model']['mosnet']['l-weights']['normal'] = 5.
        CONFIG['model']['mosnet']['l-weights']['normal_sil'] = 10000000.
    else:
        # Since the encoder only receives the RGB data as input, we must only use that data in the render-and-compare
        # optimization on the validation set, i.e., all losses that require depth data need to have weight zero.
        CONFIG['model']['mosnet']['l-weights']['rgb'] = 1.
        CONFIG['model']['mosnet']['l-weights']['depth'] = 0.
        CONFIG['model']['mosnet']['l-weights']['rgb_sil'] = 0.
        CONFIG['model']['mosnet']['l-weights']['depth_sil'] = 0.
        CONFIG['model']['mosnet']['l-weights']['z-reg'] = 0.0025
        CONFIG['model']['mosnet']['l-weights']['extr'] = 0.
        CONFIG['model']['mosnet']['l-weights']['intersect'] = 0.001
        CONFIG['model']['mosnet']['l-weights']['ground'] = 0.01
        CONFIG['model']['mosnet']['l-weights']['normal'] = 0.
        CONFIG['model']['mosnet']['l-weights']['normal_sil'] = 0.

    CONFIG['model']['mosnet']['anti_aliasing'] = False

    model_file = os.path.join(LOG_DIR, model_base_name + '.py')
    model_module = load_module_from_log(model_base_name, model_file)
    MODEL = model_module.get_model(CONFIG, model_ext_name)

    # GT background
    bg_img, bg_depth = data_provider.load_bg_img(FLAGS.data_dir, CONFIG['data']['img_size'])

    if CONFIG['model']['mosnet']['anti_aliasing']:
        bg_img = tf.image.resize(bg_img, [128, 128])
        bg_depth = tf.image.resize(bg_depth, [128, 128])
        CONFIG['model']['mosnet']['gauss_sigma'] = 0.

    MODEL.set_gt_bg(depth=bg_depth)

    #Render-and-compare object:
    r_and_c_opt = Render_and_compare_optimization(MODEL, CONFIG)

    # Load data
    DATA = data_provider.DataSetMultiObj(FLAGS.data_dir, FLAGS.split, CONFIG['data'])

    if not os.path.exists(CONFIG['r-and-c']['results_dir']):
        os.makedirs(CONFIG['r-and-c']['results_dir'])


    main()

    if CONFIG['r-and-c']['print_timing_info']:
        num_iterations = r_and_c_opt.average_runtimes['num_iterations']
        for k, v in r_and_c_opt.average_runtimes.items():
            r_and_c_opt.average_runtimes[k] = v/num_iterations
        print("\nTiming info for optimize_single_latent: ")
        print("Avg time spent on gradient descent step: {}".format(r_and_c_opt.average_runtimes['gradient_descent_step']))
        print("Avg time spent on render_single_obj: {}".format(r_and_c_opt.average_runtimes['render_single_obj']))
        print("Avg time spent on predict_remaining_objs: {}".format(r_and_c_opt.average_runtimes['predict_remaining_objs']))
        print("Avg time spent on gen_output_from_objs: {}".format(r_and_c_opt.average_runtimes['gen_output_from_objs']))
        print("Avg time spent on get_loss: {}".format(r_and_c_opt.average_runtimes['get_loss']))

    exit(0)
