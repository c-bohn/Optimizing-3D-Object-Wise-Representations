import numpy as np
import tensorflow as tf
import os
import copy
import argparse

from utils.shared_funcs import *
from utils import data_provider
from utils import viz



class training_with_pseudo_gt_latents():
    def __init__(self, CONFIG, MODEL, params, dataset_dir):
        self.config = CONFIG
        self.model = MODEL
        self.params = params
        self.dataset_dir = dataset_dir

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(LOG_DIR, 'ckpts'), max_to_keep=3)
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        if self.manager.latest_checkpoint:
            epoch = int(self.ckpt.step)
            print("Restored self.model from {}, epoch {}".format(self.manager.latest_checkpoint, epoch))
        else:
            print('No pre-trained model found at ' + LOG_DIR)

    def train_with_pseudo_gt_latents(self):
        # It is necessary to call the entire network once, in order to initialize the weights in the parts of the
        # network after the encoder. Otherwise random weights would be saved for those parts of the network.
        inputs_file = open('{}/inputs.npy'.format(self.dataset_dir), 'rb')
        input_batch_list = []
        for i in range(8):
            input_batch_list.append(np.load(inputs_file))
        input_batch = np.concatenate(input_batch_list, axis=0)
        input_batch = tf.Variable(input_batch[:, :, :, :, :3])
        output = self.model([input_batch])


        batch_size = self.config['train_with_pseudo_gt_latents']['batch_size']
        num_objects = 27000
        num_batches = num_objects//batch_size


        all_latents_list = []
        latents_file = open('{}/latents.npy'.format(self.dataset_dir), 'rb')
        for obj in range(num_objects):
            all_latents_list.append(np.load(latents_file))
        all_latents = np.stack(all_latents_list, axis=0)
        all_latents = tf.convert_to_tensor(all_latents, dtype=tf.float32)
        mean = tf.math.reduce_mean(all_latents, axis=0)
        stddev = tf.math.reduce_std(all_latents, axis=0)


        for epoch in range(self.config['train_with_pseudo_gt_latents']['num_epochs']):
            loss_sum = 0.

            #TODO: shuffle data

            learning_rate = self.config['train_with_pseudo_gt_latents']['learning_rate_latents'] / ((epoch+2)/2)

            latents_file = open('{}/latents.npy'.format(self.dataset_dir), 'rb')
            inputs_file = open('{}/inputs.npy'.format(self.dataset_dir), 'rb')

            print("Starting epoch {}, learning rate = {}".format(epoch, learning_rate))

            for batch in range(num_batches):
                input_batch_list = []
                latents_batch_list = []
                for i in range(batch_size):
                    input_batch_list.append(np.load(inputs_file))
                    latents_batch_list.append(np.load(latents_file))
                input_batch = tf.Variable(np.concatenate(input_batch_list, axis=0))
                latents_batch = tf.Variable(np.concatenate(latents_batch_list, axis=0))

                with tf.GradientTape(persistent=True) as tape:
                    pred_latents_batch = self.model.encoder_obj([input_batch], training=True)['z_mean']

                    #Normalize latents
                    #pred_latents_batch = (pred_latents_batch-mean)/stddev
                    #latents_batch = (latents_batch-mean)/stddev

                    latents_loss = tf.math.reduce_mean(tf.keras.losses.MSE(pred_latents_batch, latents_batch))
                    loss_sum += latents_loss

                if batch % 100 == 0:
                    print("EPOCH {}, BATCH {}/{}".format(epoch, batch, num_batches), end='\r')

                if CONFIG['train_with_pseudo_gt_latents']['render_results']:
                    for i in range(batch_size):
                        viz.show_image(input_batch.numpy()[i, 0, :, :, 3:6], './dataset/data_test_{}_diffrgb.png'.format(batch*batch_size+i))
                        single_obj_render = self.model.render_single_obj(latents_batch.numpy()[i:i+1, ...])['obj_rgb_pred']
                        viz.show_image(single_obj_render.numpy()[0, 0, ...], './dataset/data_test_{}_single_obj_render_opt.png'.format(batch*batch_size+i))
                        single_obj_render_pred = self.model.render_single_obj(pred_latents_batch.numpy()[i:i+1, ...])['obj_rgb_pred']
                        viz.show_image(single_obj_render_pred.numpy()[0, 0, ...], './dataset/data_test_{}_single_obj_render_pred.png'.format(batch*batch_size+i))

                optimizer_encoder = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                variables_encoder = self.model.encoder_obj.trainable_variables
                gradients = tape.gradient(latents_loss, variables_encoder)
                optimizer_encoder.apply_gradients(zip(gradients, variables_encoder))

            print("Avg loss during epoch {}: {}".format(epoch, loss_sum/num_batches))

            save_path = self.manager.save()
            print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
            self.ckpt.step.assign_add(1)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clevr', help='Data Set')
    parser.add_argument('--data_dir', help='Data dir, only needed to initialized the model correctly')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--eval_dir', default='eval', help='Log dir [default: eval]')
    parser.add_argument('--model', default='mosnet-org', help='Model name')
    parser.add_argument('--no_gpu', default='0', help='Do not use GPU')
    parser.add_argument('--config', default='', help='Configuration file [cnfg_<model>_<dataset>]')
    parser.add_argument('--message', default='', help='Message that specifies settings etc. (for log dir name)')
    parser.add_argument('--opt_data_dir', help='The directory holding the encoder input - latents pairs')
    return parser.parse_args()

if __name__ == "__main__":
    FLAGS = parse_arguments()

    if FLAGS.message == "":
        print('Need to specify message/ name for  model that should be evaluated[--message]')
        exit()
    model_base_name, model_ext_name = FLAGS.model.split('-')

    if FLAGS.config == "":
        FLAGS.config = 'cnfg_' + model_base_name + '_' + FLAGS.dataset

    if FLAGS.no_gpu == '1':
        print("NOT USING GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Restore trained model
    LOG_DIR = os.path.join(FLAGS.log_dir, FLAGS.dataset + '_' + FLAGS.model + '(' + FLAGS.message + ')')
    if not os.path.exists(os.path.join(LOG_DIR, 'ckpts', 'checkpoint')):
        print('No pre-trained model found at ', LOG_DIR)
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
    CONFIG['train_with_pseudo_gt_latents']['learning_rate_latents'] = 0.00001
    CONFIG['train_with_pseudo_gt_latents']['render_results'] = False
    CONFIG['train_with_pseudo_gt_latents']['num_epochs'] = 30

    if 'anti_aliasing' not in CONFIG['model']['mosnet']:
        CONFIG['model']['mosnet']['anti_aliasing'] = False

    model_file = os.path.join(LOG_DIR, model_base_name + '.py')
    model_module = load_module_from_log(model_base_name, model_file)

    # The model to be trained:
    MODEL = model_module.get_model(CONFIG, model_ext_name)
    bg_img, bg_depth = data_provider.load_bg_img(FLAGS.data_dir, CONFIG['data']['img_size'])
    MODEL.set_gt_bg(depth=bg_depth)
    MODEL.gauss_kernel = 0 #This is necessary, as otherwise the kernel size would be 16

    params = {}
    weights = CONFIG['model'][model_base_name]['l-weights']
    for k, w in weights.items():
        params['w-' + k] = w
    params['gauss_sigma'] = tf.constant(MODEL.gauss_kernel/3.)

    params['w-z-reg'] = tf.constant(0.025)
    params['w-rgb_sil'] = tf.constant(0.)
    params['w-depth_sil'] = tf.constant(0.)

    if not os.path.exists(CONFIG['r-and-c']['results_dir']):
        os.makedirs(CONFIG['r-and-c']['results_dir'])

    config_string = "CONFIG['r-and-c']:\n"
    for k, v in CONFIG['r-and-c'].items():
        config_string += "{}: {}\n".format(k, CONFIG['r-and-c'][k])
    config_string += "CONFIG['train_with_pseudo_gt_latents']:\n"
    for k, v in CONFIG['train_with_pseudo_gt_latents'].items():
        config_string += "{}: {}\n".format(k, CONFIG['train_with_pseudo_gt_latents'][k])
    print(config_string)

    dataset_dir = FLAGS.opt_data_dir

    training = training_with_pseudo_gt_latents(CONFIG, MODEL, params, dataset_dir)
    training.train_with_pseudo_gt_latents()

    exit(0)