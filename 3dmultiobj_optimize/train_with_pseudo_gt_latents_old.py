import argparse
import os
import multiprocessing
import logging
import itertools
import copy
import numpy as np

from datetime import datetime, timedelta
from utils import data_provider
from utils import viz
from utils.shared_funcs import *
from utils.tf_funcs import *
from render_and_compare import *

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
    parser.add_argument('--latents_file', default='', help='CSV file that stores the pre-computed latent vectors')
    parser.add_argument('--compute_new_latents', default='0', help='Whether to compute new pseudo-ground-truth latents')
    return parser.parse_args()

class training_with_pseudo_gt_latents():
    def __init__(self, CONFIG, MODEL, params, render_and_compare_opt_model, params_r_and_c_opt_model, DATA_TRAIN):
        self.config = CONFIG
        self.model = MODEL
        self.params = params
        self.r_and_c_opt_model = render_and_compare_opt_model
        self.params_r_and_c_opt_model = params_r_and_c_opt_model
        self.data = DATA_TRAIN
        self.num_batches = (self.data.get_size() // self.config['train_with_pseudo_gt_latents']['batch_size']) + 1
        dataset_train, iterator_train = get_data_iterator(self.data, self.config['train_with_pseudo_gt_latents'])
        self.iterator = iterator_train

        tf.random.set_seed(42)

        # Restore weights of the render_and_compare_opt_model
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=self.r_and_c_opt_model)
        manager = tf.train.CheckpointManager(ckpt, os.path.join(LOG_DIR+'_r_and_c', 'ckpts'), max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            epoch = int(ckpt.step)
            print("Restored self.r_and_c_opt_model from {}, epoch {}".format(manager.latest_checkpoint, epoch))
            logging.info("Restored self.r_and_c_opt_model from {}, epoch {}".format(manager.latest_checkpoint, epoch))
        else:
            print('[ERROR] render_and_compare.py, no pre-trained model found at ' + LOG_DIR+'_r_and_c')
            logging.error('[ERROR] render_and_compare.py, no pre-trained model found at ' + LOG_DIR+'_r_and_c')
            exit(1)

        # Restore weights of the model
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(LOG_DIR, 'ckpts'), max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        if self.manager.latest_checkpoint:
            epoch = int(self.ckpt.step)
            print("Restored self.model from {}, epoch {}".format(self.manager.latest_checkpoint, epoch))
            logging.info("Restored self.model from {}, epoch {}".format(self.manager.latest_checkpoint, epoch))
        else:
            print('[ERROR] render_and_compare.py, no pre-trained model found at ' + LOG_DIR)
            logging.error('[ERROR] render_and_compare.py, no pre-trained model found at ' + LOG_DIR)
            exit(1)

        self.render_and_compare_opt = Render_and_compare_optimization(self.r_and_c_opt_model, self.config)

        # Restore the pseudo-gt latents from file:
        print("Restoring pseudo-gt latents from {}...".format(self.config['train_with_pseudo_gt_latents']['latents_file']))
        logging.info("Restoring pseudo-gt latents from {}...".format(self.config['train_with_pseudo_gt_latents']['latents_file']))
        try:
            latents_raw = np.loadtxt(self.config['train_with_pseudo_gt_latents']['latents_file'])
            batch_size = self.config['train_with_pseudo_gt_latents']['batch_size']
            n_slots = self.model.n_slots
            dim_latent = self.config['model']['mosnet']['dim_latent']
            if not latents_raw.shape[0] == self.num_batches*batch_size*n_slots*dim_latent:
                print("[ERROR]: The number of latents provided in the file does not match the data set")
                logging.error("[ERROR]: The number of latents provided in the file does not match the data set")
                exit(1)
            latents = latents_raw.reshape([self.num_batches, batch_size, n_slots, 1, dim_latent])
            self.latents = tf.convert_to_tensor(latents, dtype=tf.float32)
            print("Successfully restored latents from file")
            logging.info("Successfully restored latents from file")

            self.mean = tf.math.reduce_mean(self.latents, axis=[0, 1, 2, 3])
            self.stddev  = tf.math.reduce_std(self.latents, axis=[0, 1, 2, 3])
        except OSError:
            print("No latents file provided or it could not be found")
            print("New pseudo-gt latents can be computed with self.compute_new_latents()")

        if self.config['train_with_pseudo_gt_latents']['analyze_diffs']:
            self.diffs = []

    def train_with_pseudo_gt_latents(self):
        for epoch in range(self.config['train_with_pseudo_gt_latents']['num_epochs']):
            _, self.iterator = get_data_iterator(self.data, self.config['train_with_pseudo_gt_latents'])
            print("EPOCH = {}".format(epoch))
            logging.info("EPOCH = {}".format(epoch))
            latents_losses = []
            total_losses = []
            total_losses_running_avgs = []
            all_losses = []

            for batch in range(self.num_batches):
                self.run_update_step_for_batch(batch, latents_losses, total_losses, total_losses_running_avgs, all_losses)

            if self.config['train_with_pseudo_gt_latents']['analyze_diffs']:
                diffs_tensor = K.stack(self.diffs, axis=0)
                mean_diffs = tf.math.reduce_mean(diffs_tensor, axis=[0, 1, 2])
                var_diffs = tf.math.reduce_variance(diffs_tensor, axis=[0, 1, 2])
                mean_abs_diffs = tf.math.reduce_mean(tf.abs(diffs_tensor), axis=[0, 1, 2])
                var_abs_diffs = tf.math.reduce_variance(tf.abs(diffs_tensor), axis=[0, 1, 2])
                max_diffs = tf.math.reduce_max(tf.abs(diffs_tensor), axis=[0, 1, 2])
                min_diffs = tf.math.reduce_min(tf.abs(diffs_tensor), axis=[0, 1, 2])

                print("Mean differences after epoch {}:\n{}".format(epoch, mean_diffs))
                print("Variances of the differences after epoch {}:\n{}".format(epoch, var_diffs))
                print("Mean abs. differences after epoch {}:\n{}".format(epoch, mean_abs_diffs))
                print("Variances of the abs. differences after epoch {}:\n{}".format(epoch, var_abs_diffs))
                print("Maximum absolute differences after epoch {}:\n{}".format(epoch, max_diffs))
                print("Minimum absolute differences after epoch {}:\n{}".format(epoch, min_diffs))
                logging.info("Mean differences after epoch {}:\n{}".format(epoch, mean_diffs))
                logging.info("Variances of the differences after epoch {}:\n{}".format(epoch, var_diffs))
                logging.info("Mean abs. differences after epoch {}:\n{}".format(epoch, mean_abs_diffs))
                logging.info("Variances of the abs. differences after epoch {}:\n{}".format(epoch, var_abs_diffs))
                logging.info("Maximum absolute differences after epoch {}:\n{}".format(epoch, max_diffs))
                logging.info("Minimum absolute differences after epoch {}:\n{}".format(epoch, min_diffs))

                self.diffs = []

            avg_total_loss = tf.reduce_mean(total_losses)
            avg_latents_loss = tf.reduce_mean(latents_losses)
            all_losses_avg_string = ""
            for k, v in all_losses[0].items():
                if not (k=="normal_pred" or k=="tmp_depth_diff"):
                    all_losses_avg_string += "Avg " + k + ": "
                    avg = 0.
                    for loss_dict in all_losses:
                        avg += loss_dict[k]
                    avg /= len(all_losses)
                    all_losses_avg_string += "{}\n".format(avg)

            print("After epoch {}: Avg total loss = {}, Avg latents loss = {}\n{}".format(epoch, avg_total_loss, avg_latents_loss, all_losses_avg_string))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
            logging.info("After epoch {}: Avg total loss = {}, Avg latents loss = {}\n{}".format(epoch, avg_total_loss, avg_latents_loss, all_losses_avg_string))
            logging.info("+++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

            save_path = self.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            logging.info("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            self.ckpt.step.assign_add(1)

        time_now_str = '{:%d.%m_%H:%M}'.format(datetime.now())
        print("Finished training at {}".format(time_now_str))
        logging.info("Finished training at {}".format(time_now_str))

    def run_update_step_for_batch(self, batch_num, latents_losses, total_losses, total_losses_running_avgs, all_losses):
        start_time_batch = datetime.now()
        # Get the batch inputs:
        batch_inputs = next(self.iterator)
        batch_inputs = self.model.get_input(batch_inputs)

        # The optimized latents for the batch:
        z_pred_opt = self.latents[batch_num, ...]

        if self.config['train_with_pseudo_gt_latents']['render_results']:
            if batch_num==0:
                img_size = self.config['data']['img_size']
                batch_size = self.config['train_with_pseudo_gt_latents']['batch_size']
                for i in range(batch_size):
                    for j in range(self.model.n_slots):
                        single_obj_z = z_pred_opt[i, j:j+1, ...]
                        slot_rendering = self.r_and_c_opt_model.render_single_obj(single_obj_z)
                        obj_rgb = slot_rendering['obj_rgb_pred'][0, 0, ...].numpy().reshape((img_size, img_size, 3))
                        viz.show_image(obj_rgb, self.config['r-and-c']['results_dir']+'/ex_{}_rgb_pred_opt_obj_{}_from_file.png'.format(batch_num*batch_size+i, j))

        # Compute the squared difference loss between the encoder output of the model and the pseudo-ground-truth latents:
        rgb_in = batch_inputs['rgb_in']

        res = {k: [] for k in self.model.output_types}
        diffrgb_list = [rgb_in]
        msk_list = [tf.expand_dims(tf.zeros_like(rgb_in), axis=1)[..., :1]]
        z_obj_list = []

        with tf.GradientTape(persistent=True) as tape:
            for s in range(self.model.n_slots):
                msk_all = tf.concat(msk_list, axis=1)
                msk_all = tf.reduce_max(msk_all, axis=1)
                slot_input = tf.concat([rgb_in, diffrgb_list[-1], msk_all], axis=-1)

                z_obj_cur = self.model.encoder_obj([slot_input], training=True)['z_mean']
                z_obj_list.append(z_obj_cur)

                obj_res = self.model.render_single_obj(z_obj_cur)
                res_cur = {}
                for k in self.model.output_types:
                    if k in obj_res:
                        res[k].append(obj_res[k])
                        res_cur[k] = tf.stack(res[k], axis=1)
                combined_res = self.model.depth_ordering(res_cur, tf.zeros_like(res_cur['obj_rgb_pred'][:, 0]))
                diffrgb_list.append(rgb_in - combined_res['rgb_pred'])
                msk_list.append(combined_res['msk_pred'])

                if self.config['train_with_pseudo_gt_latents']['render_results']:
                    if batch_num==0:
                        img_size = self.config['data']['img_size']
                        batch_size = self.config['train_with_pseudo_gt_latents']['batch_size']
                        for i in range(self.config['train_with_pseudo_gt_latents']['batch_size']):
                            obj_rgb = obj_res['obj_rgb_pred'][i, 0, ...].numpy().reshape((img_size, img_size, 3))
                            viz.show_image(obj_rgb, self.config['r-and-c']['results_dir'] +'/ex_{}_rgb_pred_obj_{}.png'.format(batch_num * batch_size + i, s))

            z_pred = K.stack(z_obj_list, axis=1)

            output = self.model([rgb_in])
            losses = self.model.get_loss(output, batch_inputs, self.params)

            latents_loss = self.get_latent_loss(z_pred_opt, z_pred)

        print("BATCH {}: loss_total = {}, latents_loss = {}".format(batch_num, losses['loss_total'], latents_loss))
        logging.info("BATCH {}: loss_total = {}, latents_loss = {}".format(batch_num, losses['loss_total'], latents_loss))
        latents_losses.append(latents_loss)
        total_losses.append(losses['loss_total'])
        all_losses.append(losses)
        if batch_num >= 9:
            print("Running avgs: loss_total = {}, latents_loss = {}".format(tf.reduce_mean(total_losses[batch_num - 9:batch_num + 1]), tf.reduce_mean(latents_losses[batch_num - 9:batch_num + 1])))
            logging.info("Running avgs: loss_total = {}, latents_loss = {}".format(tf.reduce_mean(total_losses[batch_num - 9:batch_num + 1]), tf.reduce_mean(latents_losses[batch_num - 9:batch_num + 1])))
            total_losses_running_avgs.append(tf.reduce_mean(total_losses[batch_num - 9:batch_num + 1]))

        # Gradient descent updates on the trainable variables of the model:
        print("Updating trainable variables...")
        logging.info("Updating trainable variables...")

        optimizer_encoder = tf.keras.optimizers.Adam(learning_rate=self.config['train_with_pseudo_gt_latents']['learning_rate_latents'])
        variables_encoder = self.model.encoder_obj.trainable_variables
        gradients = tape.gradient(latents_loss, variables_encoder)
        optimizer_encoder.apply_gradients(zip(gradients, variables_encoder))

        #optimizer_model = tf.keras.optimizers.Adam(learning_rate=self.config['train_with_pseudo_gt_latents']['learning_rate'])
        #variables_model = self.model.trainable_variables
        #gradients = tape.gradient(losses['loss_total'], variables_model)
        #optimizer_model.apply_gradients(zip(gradients, variables_model))

        print("Done")
        logging.info("Done")
        print("Took {} to process batch {}.".format(datetime.now() - start_time_batch, batch_num))
        logging.info("Took {} to process batch {}.".format(datetime.now() - start_time_batch, batch_num))

        print("+++++++++++++++++++++++++++++++")
        logging.info("+++++++++++++++++++++++++++++++")


    def find_closest_slot_permutation(self, z_pred_single_img, z_pred_opt_single_img):
        unmatched_slots = []
        for s in range(self.model.n_slots):
            unmatched_slots.append(s)
        permutation_list = []

        for s in range(self.model.n_slots):
            smallest_l2_distance = tf.Variable(np.inf, dtype=tf.float32)
            best_matching_slot = -1
            for s_ in unmatched_slots:
                l2_distance = tf.nn.l2_loss(z_pred_single_img[s, 0, -5:-2] - z_pred_opt_single_img[s_, 0, -5:-2])
                if l2_distance < smallest_l2_distance:
                    smallest_l2_distance = l2_distance
                    best_matching_slot = s_
            permutation_list.append(best_matching_slot)
            unmatched_slots.remove(best_matching_slot)

        return permutation_list


    def get_latent_loss(self, z_pred_opt, z_pred):
        loss_batch = 0.

        for i in range(self.config['train_with_pseudo_gt_latents']['batch_size']):
            z_pred_single_img = z_pred[i, ...]
            z_pred_opt_single_img = z_pred_opt[i, ...]

            # For each object, find the closest optimized object:
            permutation = self.find_closest_slot_permutation(z_pred_single_img, z_pred_opt_single_img)

            # Permute the optimized latents:
            z_pred_opt_single_img_permuted_slots_list = []
            for p in permutation:
                z_pred_opt_single_img_permuted_slots_list.append(z_pred_opt_single_img[p, ...])
            z_pred_opt_single_img_permuted_slots = K.stack(z_pred_opt_single_img_permuted_slots_list, axis=0)

            # Normalize all entries of the latents s.t. mean=0, var=1:
            z_pred_single_img = (z_pred_single_img-self.mean)/self.stddev
            z_pred_opt_single_img_permuted_slots = (z_pred_opt_single_img_permuted_slots-self.mean)/self.stddev

            loss_single_img = tf.math.reduce_mean(tf.keras.losses.MSE(z_pred_single_img, z_pred_opt_single_img_permuted_slots))
            loss_batch += loss_single_img

            if self.config['train_with_pseudo_gt_latents']['analyze_diffs']:
                diff = z_pred_single_img - z_pred_opt_single_img_permuted_slots
                self.diffs.append(diff)

        return loss_batch

    # def get_latent_loss(self, z_pred_opt, z_pred):
    #     # Normalize all entries of the latents s.t. mean=0, var=1
    #     z_pred_opt = (z_pred_opt-self.mean)/self.stddev
    #     z_pred = (z_pred-self.mean)/self.stddev
    #
    #     # For each image compute the loss for all permutations of the slots, select the lowest
    #     loss_batch = 0.
    #     for i in range(self.config['train_with_pseudo_gt_latents']['batch_size']):
    #         z_pred_opt_i = z_pred_opt[i, ...]
    #         z_pred_i = z_pred[i, ...]
    #         permutations = list(itertools.permutations([0, 1, 2]))
    #         lowest_loss = tf.Variable(np.inf, dtype=tf.float32)
    #         for permutation in permutations:
    #             permutation = list(permutation)
    #             z_pred_opt_i_permuted_slots_list = []
    #             for p in permutation:
    #                 z_pred_opt_i_permuted_slots_list.append(z_pred_opt_i[p, ...])
    #             z_pred_opt_i_permuted_slots = K.stack(z_pred_opt_i_permuted_slots_list, axis=0)
    #
    #             #loss_mask = np.zeros(self.model.dim_latent_obj)
    #             #loss_mask[self.model.dim_shape + self.model.dim_app:self.model.dim_shape + self.model.dim_app+3] = np.ones(3)
    #             #loss = tf.math.reduce_mean(tf.keras.losses.MSE(z_pred_i*loss_mask, z_pred_opt_i_permuted_slots*loss_mask))
    #
    #             loss = tf.math.reduce_mean(tf.keras.losses.MSE(z_pred_i, z_pred_opt_i_permuted_slots))
    #
    #             #print("{}: {}".format(loss, permutation))
    #             if loss < lowest_loss:
    #                 lowest_loss = loss
    #                 if self.config['train_with_pseudo_gt_latents']['analyze_diffs']:
    #                     diff = z_pred_i - z_pred_opt_i_permuted_slots
    #         #print("lowest_loss_batch: {}".format(lowest_loss))
    #         loss_batch += lowest_loss
    #         if self.config['train_with_pseudo_gt_latents']['analyze_diffs']: self.diffs.append(diff)
    #     #print("loss_batch: {}".format(loss_batch))
    #     return loss_batch

    def compute_new_latents(self):
        csv_filename = 'z_pred_opt_{}.csv'.format(datetime.now().strftime("%Y_%m_%d-%I_%M_%S"))
        print("Computing new latents, Will store new latents in {}".format(csv_filename))
        logging.info("Computing new latents, Will store new latents in {}".format(csv_filename))

        for batch in range(self.num_batches):
            # Get the optimized latents using render-and-compare:
            print("BATCH {}: optimizing latents...".format(batch))
            logging.info("BATCH {}: optimizing latents...".format(batch))

            # Get the batch inputs:
            batch_inputs = next(self.iterator)
            batch_inputs = self.model.get_input(batch_inputs)

            z_img_list = []
            for i in range(self.config['train_with_pseudo_gt_latents']['batch_size']):
                single_img_input = {}
                for k, v in batch_inputs.items():
                    single_img_input[k] = v[i:i+1, ...]
                if self.config['train_with_pseudo_gt_latents']['render_results']:
                    img_size = self.config['data']['img_size']
                    batch_size = self.config['train_with_pseudo_gt_latents']['batch_size']
                    img_num = i
                    if batch==0:
                        rgb_gt = single_img_input['rgb_gt'][0]
                        rgb_gt = (np.transpose(rgb_gt, (1, 0, 2, 3))).reshape((img_size, img_size, 3))
                        viz.show_image(rgb_gt, self.config['r-and-c']['results_dir']+'/ex_{}_rgb_gt.png'.format(batch*batch_size+img_num))

                results = self.render_and_compare_opt.optimize_latents(single_img_input, self.params_r_and_c_opt_model)
                z_img = results['res']['z_all']  # The latents of the current image
                print("Image {} done".format(i))
                logging.info("Image {} done".format(i))

                if self.config['train_with_pseudo_gt_latents']['render_results']:
                    res = []
                    img_size = self.config['data']['img_size']
                    batch_size = self.config['train_with_pseudo_gt_latents']['batch_size']
                    if batch==0:
                        for j in range(self.model.n_slots):
                            single_obj_z = z_img[:, j, ...]
                            slot_rendering = self.render_and_compare_opt.model.render_single_obj(single_obj_z)
                            res.append(slot_rendering)
                            obj_rgb = slot_rendering['obj_rgb_pred'][0, 0, ...].numpy().reshape((img_size, img_size, 3))
                            viz.show_image(obj_rgb, self.config['r-and-c']['results_dir']+'/ex_{}_rgb_pred_opt_obj_{}.png'.format(batch*batch_size+i, j))

                z_img_list.append(z_img[0, ...])
            z_pred_opt = K.stack(z_img_list, axis=0)

            with open(csv_filename, "ab") as f:
                np.savetxt(f, z_pred_opt.numpy().reshape([-1]))
            print("Saved optimized latents to csv file")
            logging.info("Saved optimized latents to csv file")

            print("Done")
            logging.info("Done")

        print("Done optimizing latents for all batches, saved results to {}".format(csv_filename))
        logging.info("Done optimizing latents for all batches, saved results to {}".format(csv_filename))

        latents_raw = np.loadtxt(csv_filename)
        batch_size = self.config['train_with_pseudo_gt_latents']['batch_size']
        n_slots = self.model.n_slots
        dim_latent = self.config['model']['mosnet']['dim_latent']
        if not latents_raw.shape[0] == self.num_batches*batch_size*n_slots*dim_latent:
            print("[ERROR]: The number of latents provided in the file does not match the data set")
            logging.error("[ERROR]: The number of latents provided in the file does not match the data set")
            exit(1)
        latents = latents_raw.reshape([self.num_batches, batch_size, n_slots, 1, dim_latent])
        self.latents = tf.convert_to_tensor(latents, dtype=tf.float32)
        print("Successfully restored latents from file")
        logging.info("Successfully restored latents from file")

        self.mean = tf.math.reduce_mean(self.latents, axis=[0, 1, 2, 3])
        self.stddev  = tf.math.reduce_std(self.latents, axis=[0, 1, 2, 3])
        self.config['train_with_pseudo_gt_latents']['latents_file'] = csv_filename



#######################################################################################################################

# def main():
#     print("+++++++++++++++++++++++++++++++")
#     logging.info("+++++++++++++++++++++++++++++++")
#
#     if (not CONFIG['train_with_pseudo_gt_latents']['compute_new_pseudo_gt_latents']) and CONFIG['train_with_pseudo_gt_latents']['latents_file']=='':
#         print("[ERROR] No latents file specified, add one with '--latents_file <path_to_file>'")
#         logging.error("[ERROR] No latents file specified, add one with '--latents_file <path_to_file>'")
#         exit(1)
#
#     config_string = "CONFIG['r-and-c']:\n"
#     for k, v in CONFIG['r-and-c'].items():
#         config_string += "{}: {}\n".format(k, CONFIG['r-and-c'][k])
#     config_string += "CONFIG['train_with_pseudo_gt_latents']:\n"
#     for k, v in CONFIG['train_with_pseudo_gt_latents'].items():
#         config_string += "{}: {}\n".format(k, CONFIG['train_with_pseudo_gt_latents'][k])
#     print(config_string)
#     logging.info(config_string)
#
#     # Restore weights of the render_and_compare_opt_model
#     ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=render_and_compare_opt_model)
#     manager = tf.train.CheckpointManager(ckpt, os.path.join(LOG_DIR+'_r_and_c', 'ckpts'), max_to_keep=3)
#     ckpt.restore(manager.latest_checkpoint).expect_partial()
#     if manager.latest_checkpoint:
#         epoch = int(ckpt.step)
#         print("Restored render_and_compare_opt_model from {}, epoch {}".format(manager.latest_checkpoint, epoch))
#         logging.info("Restored render_and_compare_opt_model from {}, epoch {}".format(manager.latest_checkpoint, epoch))
#     else:
#         print('[ERROR] render_and_compare.py, no pre-trained model found at ' + LOG_DIR+'_r_and_c')
#         logging.error('[ERROR] render_and_compare.py, no pre-trained model found at ' + LOG_DIR+'_r_and_c')
#         exit(1)
#
#     # Restore weights of the model
#     ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=MODEL)
#     manager = tf.train.CheckpointManager(ckpt, os.path.join(LOG_DIR, 'ckpts'), max_to_keep=3)
#     ckpt.restore(manager.latest_checkpoint).expect_partial()
#     if manager.latest_checkpoint:
#         epoch = int(ckpt.step)
#         print("Restored MODEL from {}, epoch {}".format(manager.latest_checkpoint, epoch))
#         logging.info("Restored MODEL from {}, epoch {}".format(manager.latest_checkpoint, epoch))
#     else:
#         print('[ERROR] render_and_compare.py, no pre-trained model found at ' + LOG_DIR)
#         logging.error('[ERROR] render_and_compare.py, no pre-trained model found at ' + LOG_DIR)
#         exit(1)
#
#     render_and_compare_opt = Render_and_compare_optimization(render_and_compare_opt_model, CONFIG)
#
#     # The name of the csv file we will save the optimized latents to
#     csv_filename = 'z_pred_opt_{}.csv'.format(datetime.now().strftime("%Y_%m_%d-%I_%M_%S"))
#
#     start_time = datetime.now()
#     time_now_str = '{:%d.%m_%H:%M}'.format(start_time)
#     print("Started training at {}".format(time_now_str))
#     logging.info("Started training at {}".format(time_now_str))
#
#     print("+++++++++++++++++++++++++++++++")
#     logging.info("+++++++++++++++++++++++++++++++")
#
#     num_batches = (DATA_TRAIN.get_size() // CONFIG['train_with_pseudo_gt_latents']['batch_size']) + 1
#
#     if CONFIG['train_with_pseudo_gt_latents']['compute_new_pseudo_gt_latents']:
#         print("No latents file provided, will store new latents in {}".format(csv_filename))
#         logging.info("No latents file provided, will store new latents in {}".format(csv_filename))
#         # Create dataset iterator over the training set
#         dataset_train, iterator_train = get_data_iterator(DATA_TRAIN, CONFIG['train_with_pseudo_gt_latents'])
#         iterator = iterator_train
#
#         print("EPOCH = 0, will compute new pseudo-gt latents")
#         logging.info("EPOCH = 0, will compute new pseudo-gt latents")
#         latents_losses = []
#         total_losses = []
#         total_losses_running_avgs = []
#         all_losses = []
#
#         for batch in range(num_batches):
#             run_update_step_for_batch(batch, iterator, latents_losses, total_losses, total_losses_running_avgs, all_losses, manager, ckpt, csv_filename)
#
#         avg_total_loss = tf.reduce_mean(total_losses)
#         avg_latents_loss = tf.reduce_mean(latents_losses)
#         all_losses_avg_string = ""
#         for k, v in all_losses[0].items():
#             if not (k=="normal_pred" or k=="tmp_depth_diff"):
#                 all_losses_avg_string += "Avg " + k + ": "
#                 avg = 0.
#                 for loss_dict in all_losses:
#                     avg += loss_dict[k]
#                 avg /= len(all_losses)
#                 all_losses_avg_string += "{}\n".format(avg)
#
#         print("After epoch 0: Avg total loss = {}, Avg latents loss = {}\n{}".format(avg_total_loss, avg_latents_loss, all_losses_avg_string))
#         print("+++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
#         logging.info("After epoch 0: Avg total loss = {}, Avg latents loss = {}\n{}".format(avg_total_loss, avg_latents_loss, all_losses_avg_string))
#         logging.info("+++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
#
#
#
#         print("Restoring pseudo-gt latents from {}...".format(csv_filename))
#         logging.info("Restoring pseudo-gt latents from {}...".format(csv_filename))
#         latents_raw = np.loadtxt(csv_filename)
#     else:
#         print("Restoring pseudo-gt latents from {}...".format(CONFIG['train_with_pseudo_gt_latents']['latents_file']))
#         logging.info("Restoring pseudo-gt latents from {}...".format(CONFIG['train_with_pseudo_gt_latents']['latents_file']))
#         latents_raw = np.loadtxt(CONFIG['train_with_pseudo_gt_latents']['latents_file'])
#
#     batch_size = CONFIG['train_with_pseudo_gt_latents']['batch_size']
#     n_slots = MODEL.n_slots
#     dim_latent = CONFIG['model']['mosnet']['dim_latent']
#     if not latents_raw.shape[0] == num_batches*batch_size*n_slots*dim_latent:
#         print("[ERROR]: The number of latents provided in the file does not match the data set")
#         logging.error("[ERROR]: The number of latents provided in the file does not match the data set")
#         exit(1)
#     latents = latents_raw.reshape([num_batches, batch_size, n_slots, 1, dim_latent])
#     latents = tf.convert_to_tensor(latents, dtype=tf.float32)
#     print("Successfully restored latents from file")
#     logging.info("Successfully restored latents from file")
#
#     for epoch in range(1 * (CONFIG['train_with_pseudo_gt_latents']['compute_new_pseudo_gt_latents']), CONFIG['train_with_pseudo_gt_latents']['num_epochs']):
#         # Create dataset iterator over the training set
#         dataset_train, iterator_train = get_data_iterator(DATA_TRAIN, CONFIG['train_with_pseudo_gt_latents'])
#         iterator = iterator_train
#
#         print("EPOCH = {}".format(epoch))
#         logging.info("EPOCH = {}".format(epoch))
#         latents_losses = []
#         total_losses = []
#         total_losses_running_avgs = []
#         all_losses = []
#
#         for batch in range(num_batches):
#             run_update_step_for_batch(batch, iterator, latents_losses, total_losses, total_losses_running_avgs, all_losses, manager, ckpt, csv_filename, compute_latents=False, latents=latents)
#
#         avg_total_loss = tf.reduce_mean(total_losses)
#         avg_latents_loss = tf.reduce_mean(latents_losses)
#         all_losses_avg_string = ""
#         for k, v in all_losses[0].items():
#             if not (k=="normal_pred" or k=="tmp_depth_diff"):
#                 all_losses_avg_string += "Avg " + k + ": "
#                 avg = 0.
#                 for loss_dict in all_losses:
#                     avg += loss_dict[k]
#                 avg /= len(all_losses)
#                 all_losses_avg_string += "{}\n".format(avg)
#
#         print("After epoch {}: Avg total loss = {}, Avg latents loss = {}\n{}".format(epoch, avg_total_loss, avg_latents_loss, all_losses_avg_string))
#         print("+++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
#         logging.info("After epoch {}: Avg total loss = {}, Avg latents loss = {}\n{}".format(epoch, avg_total_loss, avg_latents_loss, all_losses_avg_string))
#         logging.info("+++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
#
#     time_now_str = '{:%d.%m_%H:%M}'.format(datetime.now())
#     print("Finished training at {}".format(time_now_str))
#     logging.info("Finished training at {}".format(time_now_str))
#
# @profile
# def run_update_step_for_batch(batch_num, iterator, latents_losses, total_losses, total_losses_running_avgs, all_losses, manager, ckpt, csv_filename, compute_latents=True, latents=None):
#     start_time_batch = datetime.now()
#     # Get the batch inputs:
#     batch_inputs = next(iterator)
#     batch_inputs = MODEL.get_input(batch_inputs)
#
#     if compute_latents:
#         # Get the optimized latents using render-and-compare:
#         print("BATCH {}: optimizing latents...".format(batch_num))
#         logging.info("BATCH {}: optimizing latents...".format(batch_num))
#
#         if not CONFIG['train_with_pseudo_gt_latents']['parallel_optimization']:
#             single_img_inputs = []
#             for i in range(CONFIG['train_with_pseudo_gt_latents']['batch_size']):
#                 single_img_input = {}
#                 for k, v in batch_inputs.items():
#                     single_img_input[k] = v[i:i+1, ...]
#                 single_img_input['img_num'] = i
#                 single_img_input['batch_num'] = batch_num
#                 #These two entries are very inefficient and not necessary, just to have a common interface with the parallel function:
#                 single_img_input['model'] = render_and_compare_opt_model
#                 single_img_input['config'] = CONFIG
#                 single_img_inputs.append(single_img_input)
#
#             z_img_list = []
#             for i in range(CONFIG['train_with_pseudo_gt_latents']['batch_size']):
#                 z_img_list.append(optimize_latents_single_img(single_img_inputs[i] ))
#             z_pred_opt = K.stack(z_img_list, axis=0)
#
#         else:
#             single_img_inputs = []
#             for i in range(CONFIG['train_with_pseudo_gt_latents']['batch_size']):
#                 single_img_input = {}
#                 for k, v in batch_inputs.items():
#                     single_img_input[k] = v[i:i+1, ...]
#                 single_img_input['img_num'] = i
#                 single_img_input['config'] = CONFIG
#                 single_img_input['flags'] = FLAGS
#                 single_img_input['params'] = params
#                 single_img_inputs.append(single_img_input)
#
#
#             z_img_list = []
#             pool = multiprocessing.Pool(processes=8)
#
#             z_img_list[:] = pool.map(par_optimize_latents_single_img, (single_img_inputs[i] for i in range(CONFIG['train_with_pseudo_gt_latents']['batch_size'])) )
#             z_pred_opt = K.stack(z_img_list, axis=0)
#
#         with open(csv_filename, "ab") as f:
#             np.savetxt(f, z_pred_opt.numpy().reshape([-1]))
#         print("Saved optimized latents to csv file")
#         logging.info("Saved optimized latents to csv file")
#
#         print("Done")
#         logging.info("Done")
#     else:
#         # Load the optimized latents from file:
#         z_pred_opt = latents[batch_num, ...]
#
#         if CONFIG['train_with_pseudo_gt_latents']['render_results']:
#             if batch_num==0:
#                 img_size = CONFIG['data']['img_size']
#                 batch_size = CONFIG['train_with_pseudo_gt_latents']['batch_size']
#                 for i in range(batch_size):
#                     for j in range(MODEL.n_slots):
#                         single_obj_z = z_pred_opt[i, j:j+1, ...]
#                         slot_rendering = render_and_compare_opt_model.render_single_obj(single_obj_z)
#                         obj_rgb = slot_rendering['obj_rgb_pred'][0, 0, ...].numpy().reshape((img_size, img_size, 3))
#                         viz.show_image(obj_rgb, CONFIG['r-and-c']['results_dir']+'/ex_{}_rgb_pred_opt_obj_{}_from_file.png'.format(batch_num*batch_size+i, j))
#
#     # Compute the squared difference loss between the encoder output of the model and the pseudo-ground-truth latents:
#     rgb_in = batch_inputs['rgb_in']
#
#     res = {k: [] for k in MODEL.output_types}
#     diffrgb_list = [rgb_in]
#     msk_list = [tf.expand_dims(tf.zeros_like(rgb_in), axis=1)[..., :1]]
#     z_obj_list = []
#
#     with tf.GradientTape(persistent=True) as tape:
#         for s in range(MODEL.n_slots):
#             msk_all = tf.concat(msk_list, axis=1)
#             msk_all = tf.reduce_max(msk_all, axis=1)
#             slot_input = tf.concat([rgb_in, diffrgb_list[-1], msk_all], axis=-1)
#
#             z_obj_cur = MODEL.encoder_obj([slot_input], training=True)['z_mean']
#             z_obj_list.append(z_obj_cur)
#
#             obj_res = MODEL.render_single_obj(z_obj_cur)
#             res_cur = {}
#             for k in MODEL.output_types:
#                 if k in obj_res:
#                     res[k].append(obj_res[k])
#                     res_cur[k] = tf.stack(res[k], axis=1)
#             combined_res = MODEL.depth_ordering(res_cur, tf.zeros_like(res_cur['obj_rgb_pred'][:, 0]))
#             diffrgb_list.append(rgb_in - combined_res['rgb_pred'])
#             msk_list.append(combined_res['msk_pred'])
#
#             if CONFIG['train_with_pseudo_gt_latents']['render_results']:
#                 if batch_num==0:
#                     img_size = CONFIG['data']['img_size']
#                     batch_size = CONFIG['train_with_pseudo_gt_latents']['batch_size']
#                     for i in range(CONFIG['train_with_pseudo_gt_latents']['batch_size']):
#                         obj_rgb = obj_res['obj_rgb_pred'][i, 0, ...].numpy().reshape((img_size, img_size, 3))
#                         viz.show_image(obj_rgb, CONFIG['r-and-c']['results_dir'] +'/ex_{}_rgb_pred_obj_{}.png'.format(batch_num * batch_size + i, s))
#
#         z_pred = K.stack(z_obj_list, axis=1)
#
#         output = MODEL([rgb_in])
#         losses = MODEL.get_loss(output, batch_inputs, params)
#
#         # TODO: This is just a hack, the script should only train the network if all latents for the trianing set are known and loaded from file
#         if isinstance(latents, tf.Tensor):
#             latents_loss = get_latent_loss(z_pred_opt, z_pred, latents)
#         else:
#             latents_loss = tf.math.reduce_mean(tf.keras.losses.MSE(z_pred_opt, z_pred))
#
#
#     print("BATCH {}: loss_total = {}, latents_loss = {}".format(batch_num, losses['loss_total'], latents_loss))
#     logging.info("BATCH {}: loss_total = {}, latents_loss = {}".format(batch_num, losses['loss_total'], latents_loss))
#     latents_losses.append(latents_loss)
#     total_losses.append(losses['loss_total'])
#     all_losses.append(losses)
#     if batch_num >= 9:
#         print("Running avgs: loss_total = {}, latents_loss = {}".format(tf.reduce_mean(total_losses[batch_num - 9:batch_num + 1]), tf.reduce_mean(latents_losses[batch_num - 9:batch_num + 1])))
#         logging.info("Running avgs: loss_total = {}, latents_loss = {}".format(tf.reduce_mean(total_losses[batch_num - 9:batch_num + 1]), tf.reduce_mean(latents_losses[batch_num - 9:batch_num + 1])))
#         total_losses_running_avgs.append(tf.reduce_mean(total_losses[batch_num - 9:batch_num + 1]))
#
#     # Gradient descent updates on the trainable variables of the model:
#     print("Updating trainable variables...")
#     logging.info("Updating trainable variables...")
#
#     optimizer_encoder = tf.keras.optimizers.Adam(learning_rate=CONFIG['train_with_pseudo_gt_latents']['learning_rate_latents'])
#     variables_encoder = MODEL.encoder_obj.trainable_variables
#     gradients = tape.gradient(latents_loss, variables_encoder)
#     optimizer_encoder.apply_gradients(zip(gradients, variables_encoder))
#
#     # TODO: remove
#     total_num_weights = 0
#     for gradient in gradients:
#         total_num_weights += tf.shape(tf.reshape(gradient, [-1]))
#     print(total_num_weights)
#
#     #optimizer_model = tf.keras.optimizers.Adam(learning_rate=CONFIG['train_with_pseudo_gt_latents']['learning_rate'])
#     #variables_model = MODEL.trainable_variables
#     #gradients = tape.gradient(losses['loss_total'], variables_model)
#     #optimizer_model.apply_gradients(zip(gradients, variables_model))
#
#     print("Done")
#     logging.info("Done")
#     print("Took {} to process batch {}.".format(datetime.now() - start_time_batch, batch_num))
#     logging.info("Took {} to process batch {}.".format(datetime.now() - start_time_batch, batch_num))
#
#     save_path = manager.save()
#     print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
#     logging.info("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
#     ckpt.step.assign_add(1)
#
#     print("+++++++++++++++++++++++++++++++")
#     logging.info("+++++++++++++++++++++++++++++++")
#
# def get_latent_loss(z_pred_opt, z_pred, latents):
#     # TODO: make the train_with_with_r_and_c script a class, such that the mean and variance of the latents can be class attributes
#
#     # Normalize all entries of the latents s.t. mean=0, var=1
#     mean = tf.math.reduce_mean(latents, axis=[0, 1, 2, 3])
#     stddev  = tf.math.reduce_std(latents, axis=[0, 1, 2, 3])
#     z_pred_opt = (z_pred_opt-mean)/stddev
#     z_pred = (z_pred-mean)/stddev
#
#     # For each image compute the loss for all permutations of the slots, select the lowest
#     loss_batch = 0.
#     for i in range(CONFIG['train_with_pseudo_gt_latents']['batch_size']):
#         z_pred_opt_i = z_pred_opt[i, ...]
#         z_pred_i = z_pred[i, ...]
#         permutations = list(itertools.permutations([0, 1, 2]))
#         lowest_loss = tf.Variable(np.inf, dtype=tf.float32)
#         for permutation in permutations:
#             permutation = list(permutation)
#             z_pred_opt_i_permuted_slots_list = []
#             for p in permutation:
#                 z_pred_opt_i_permuted_slots_list.append(z_pred_opt_i[p, ...])
#             z_pred_opt_i_permuted_slots = K.stack(z_pred_opt_i_permuted_slots_list, axis=0)
#             loss = tf.math.reduce_mean(tf.keras.losses.MSE(z_pred_i, z_pred_opt_i_permuted_slots))
#             #print("{}: {}".format(loss, permutation))
#             if loss < lowest_loss: lowest_loss = loss
#         #print("lowest_loss_batch: {}".format(lowest_loss))
#         loss_batch += lowest_loss
#     #print("loss_batch: {}".format(loss_batch))
#     return loss_batch
#
#
# def optimize_latents_single_img(single_img_input):
#     if CONFIG['train_with_pseudo_gt_latents']['render_results']:
#         img_size = CONFIG['data']['img_size']
#         batch_size = CONFIG['train_with_pseudo_gt_latents']['batch_size']
#         batch_num = single_img_input['batch_num']
#         img_num = single_img_input['img_num']
#         if batch_num==0:
#             rgb_gt = single_img_input['rgb_gt'][0]
#             rgb_gt = (np.transpose(rgb_gt, (1, 0, 2, 3))).reshape((img_size, img_size, 3))
#             viz.show_image(rgb_gt, CONFIG['r-and-c']['results_dir']+'/ex_{}_rgb_gt.png'.format(batch_num*batch_size+img_num))
#
#     render_and_compare_opt = Render_and_compare_optimization(single_img_input['model'], single_img_input['config'])
#     results = render_and_compare_opt.optimize_latents(single_img_input, params_r_and_c_opt_model)
#     z_img = results['res']['z_all']  # The latents of the current image
#     print("Image {} done".format(single_img_input['img_num']))
#     logging.info("Image {} done".format(single_img_input['img_num']))
#
#     if CONFIG['train_with_pseudo_gt_latents']['render_results']:
#         res = []
#         img_size = CONFIG['data']['img_size']
#         batch_size = CONFIG['train_with_pseudo_gt_latents']['batch_size']
#         batch_num = single_img_input['batch_num']
#         img_num = single_img_input['img_num']
#         if batch_num==0:
#             for j in range(MODEL.n_slots):
#                 single_obj_z = z_img[:, j, ...]
#                 slot_rendering = render_and_compare_opt.model.render_single_obj(single_obj_z)
#                 res.append(slot_rendering)
#                 obj_rgb = slot_rendering['obj_rgb_pred'][0, 0, ...].numpy().reshape((img_size, img_size, 3))
#                 viz.show_image(obj_rgb, CONFIG['r-and-c']['results_dir']+'/ex_{}_rgb_pred_opt_obj_{}.png'.format(batch_num*batch_size+img_num, j))
#
#     return z_img[0, ...]
#
# def par_optimize_latents_single_img(single_img_input):
#     # TODO: Still not quite correct
#
#     FLAGS = single_img_input['flags']
#     CONFIG = single_img_input['config']
#     params = single_img_input['params']
#
#     if FLAGS.message == "":
#         print('Need to specify message/ name for  model that should be evaluated[--message]')
#         exit()
#     model_base_name, model_ext_name = FLAGS.model.split('-')
#
#     LOG_DIR = os.path.join(FLAGS.log_dir, FLAGS.dataset + '_' + FLAGS.model + '(' + FLAGS.message + ')')
#     model_file = os.path.join(LOG_DIR, model_base_name + '.py')
#     model_module = load_module_from_log(model_base_name, model_file)
#     MODEL = model_module.get_model(CONFIG, model_ext_name)
#
#     bg_img, bg_depth = data_provider.load_bg_img(FLAGS.data_dir, CONFIG['data']['img_size'])
#     MODEL.set_gt_bg(depth=bg_depth)
#
#     render_and_compare_opt = Render_and_compare_optimization(MODEL, CONFIG)
#     results = render_and_compare_opt.optimize_latents(single_img_input, params)
#     z_img = results['res']['z_all']  # The latents of the current image
#     print("Image {} done".format(single_img_input['img_num']))
#     return z_img[0, ...]
#


if __name__ == "__main__":
    logging.basicConfig(filename='train_with_pseudo_gt_latents_{}.log'.format(datetime.now().strftime("%Y_%m_%d-%I_%M_%S")), level=logging.DEBUG)
    FLAGS = parse_arguments()

    if FLAGS.message == "":
        print('Need to specify message/ name for  model that should be evaluated[--message]')
        logging.error('Need to specify message/ name for  model that should be evaluated[--message]')
        exit()
    model_base_name, model_ext_name = FLAGS.model.split('-')

    if FLAGS.config == "":
        FLAGS.config = 'cnfg_' + model_base_name + '_' + FLAGS.dataset

    if FLAGS.no_gpu == '1':
        print("NOT USING GPU")
        logging.info("NOT USING GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Restore trained model
    LOG_DIR = os.path.join(FLAGS.log_dir, FLAGS.dataset + '_' + FLAGS.model + '(' + FLAGS.message + ')')
    if not os.path.exists(os.path.join(LOG_DIR, 'ckpts', 'checkpoint')):
        print('No pre-trained model found at ', LOG_DIR)
        logging.error('No pre-trained model found at ', LOG_DIR)
        exit()

    # Config parameters
    cnfg_file = os.path.join(LOG_DIR, FLAGS.config + '.pkl')
    CONFIG = load_config(cnfg_file)

    CONFIG['r-and-c'] = {}                      # Holds the config parameters specific to the render-and-compare optimization
    for k, v in CONFIG['training'].items():
        CONFIG['r-and-c'][k] = v

    CONFIG['r-and-c']['learning_rate_shape'] = 0.01     # Learning rate for the shape
    CONFIG['r-and-c']['learning_rate_tex'] = 0.001      # Learning rate for the texture
    CONFIG['r-and-c']['learning_rate_ext'] = 0.01       # Learning rate for the extrinsics
    CONFIG['r-and-c']['eps_final'] = 0.00000000000000001#0.00000000001      # Convergence parameter for final output
    CONFIG['r-and-c']['eps'] = 0.0001                   # Convergence parameter for blurred images
    CONFIG['r-and-c']['optimize_rand_obj'] = False      # Whether to optimize a slot from random if the encoder output produces too large a loss
    CONFIG['r-and-c']['rand_seed'] = 42                 # Random seed
    CONFIG['r-and-c']['loss_thresh'] = 0.01             # Loss threshold above which a random vector is optimized instead of the encoder output
    CONFIG['r-and-c']['print_timing_info'] = False      # Whether to print how long the optimization steps took
    CONFIG['r-and-c']['smoothing_kernels_rand'] = [16., 4., 0.]     # The sizes of the smoothing kernels when optimizing from random initialization, in order of usage
    CONFIG['r-and-c']['smoothing_kernels_enc'] = [0.]               # The sizes of the smoothing kernels when optimizing encoder output
    CONFIG['r-and-c']['results_dir'] = './results_retrained_no_rand_initialization'
    CONFIG['r-and-c']['print_loss_vals'] = False
    CONFIG['r-and-c']['max_num_iterations'] = 50        # The maximum number of iterations with the current smoothing kernel
    CONFIG['r-and-c']['optimization_order'] = ['all']

    CONFIG['train_with_pseudo_gt_latents'] = {}
    for k, v in CONFIG['training'].items():
        CONFIG['train_with_pseudo_gt_latents'][k] = v
    CONFIG['train_with_pseudo_gt_latents']['batch_size'] = 8
    CONFIG['train_with_pseudo_gt_latents']['learning_rate_latents'] = 0.000001
    CONFIG['train_with_pseudo_gt_latents']['parallel_optimization'] = False #TODO: Parallelization doesnt work correcly yet
    CONFIG['train_with_pseudo_gt_latents']['render_results'] = False
    CONFIG['train_with_pseudo_gt_latents']['num_epochs'] = 400
    CONFIG['train_with_pseudo_gt_latents']['compute_new_pseudo_gt_latents'] = FLAGS.compute_new_latents
    CONFIG['train_with_pseudo_gt_latents']['latents_file'] = FLAGS.latents_file
    CONFIG['train_with_pseudo_gt_latents']['analyze_diffs'] = True

    model_file = os.path.join(LOG_DIR, model_base_name + '.py')
    model_module = load_module_from_log(model_base_name, model_file)

    # The model to be trained:
    MODEL = model_module.get_model(CONFIG, model_ext_name)
    bg_img, bg_depth = data_provider.load_bg_img(FLAGS.data_dir, CONFIG['data']['img_size'])
    MODEL.set_gt_bg(depth=bg_depth)
    MODEL.gauss_kernel = 0 #This is necessary, as otherwise the kernel size would be 16

    # This is the model to perform the render-and-compare optimization with;
    # If we used MODEL here, there would be an uncontrolled feedback
    # loop between the r-and-c optimization and the gradient descent steps on MODEL
    render_and_compare_opt_model = model_module.get_model(CONFIG, model_ext_name)
    bg_img, bg_depth = data_provider.load_bg_img(FLAGS.data_dir, CONFIG['data']['img_size'])
    render_and_compare_opt_model.set_gt_bg(depth=bg_depth)

    # Load data
    DATA_TRAIN = data_provider.DataSetMultiObj(FLAGS.data_dir, 'train', CONFIG['data'])

    params = {}
    weights = CONFIG['model'][model_base_name]['l-weights']
    for k, w in weights.items():
        params['w-' + k] = w
    params['gauss_sigma'] = tf.constant(MODEL.gauss_kernel/3.)

    # Here we set these weights to the values they would have at the end of a training run, these can be set to different values as well, though
    params['w-z-reg'] = tf.constant(0.025)
    params['w-rgb_sil'] = tf.constant(0.)
    params['w-depth_sil'] = tf.constant(0.)

    params_r_and_c_opt_model = copy.deepcopy(params)

    if not os.path.exists(CONFIG['r-and-c']['results_dir']):
        os.makedirs(CONFIG['r-and-c']['results_dir'])

    config_string = "CONFIG['r-and-c']:\n"
    for k, v in CONFIG['r-and-c'].items():
        config_string += "{}: {}\n".format(k, CONFIG['r-and-c'][k])
    config_string += "CONFIG['train_with_pseudo_gt_latents']:\n"
    for k, v in CONFIG['train_with_pseudo_gt_latents'].items():
        config_string += "{}: {}\n".format(k, CONFIG['train_with_pseudo_gt_latents'][k])
    print(config_string)
    logging.info(config_string)

    training = training_with_pseudo_gt_latents(CONFIG, MODEL, params, render_and_compare_opt_model, params_r_and_c_opt_model, DATA_TRAIN)
    if CONFIG['train_with_pseudo_gt_latents']['compute_new_pseudo_gt_latents'] != '0':
        training.compute_new_latents()
    training.train_with_pseudo_gt_latents()

    #main()

    exit(0)
