import os
import argparse
from datetime import datetime
import numpy as np
import math

# Disable TF info messages (needs to be done before tf import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from utils import data_provider
from utils.shared_funcs import *
from utils.tf_funcs import *
import utils.viz as viz
import models.renderer as renderer

# GPU should not be allocated entirely at beginning
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clevr', help='Data Set')
    parser.add_argument('--data_dir', default='', help='Data dir')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--model', default='deepsdf-org', help='Model name [deepsdf, osnet]')
    parser.add_argument('--config', default='', help='Configuration file [cnfg_<model>_<dataset>]')
    parser.add_argument('--message', default='', help='Message that specifies settings etc. (for log dir name)')
    parser.add_argument('--split', default='train', help='[train, val]')
    parser.add_argument('--no_gpu', default='0', help='Do not use GPU')
    FLAGS = parser.parse_args()
    return FLAGS


# --------------------------------------------------
# --- Main Training

def main():

    # Create dataset iterator for training and validation
    dataset_val, iterator_val = get_data_iterator(DATA_VAL, CONFIG['training'], type='shapedec')
    iterators = {
        'val': iterator_val
    }

    print("+++++++++++++++++++++++++++++++")

    # Renderer
    diff_renderer = renderer.DiffRenderer(MODEL, CONFIG['renderer'])

    # Operator to restore all the variables.
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=MODEL)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(LOG_DIR, 'ckpts'), max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        epoch = int(ckpt.step)
        LOG_FILE.write("Restored from {}, epoch {}".format(manager.latest_checkpoint, epoch))
    else:
        print("[ERROR] No checkpoint found.")

    # Create ops dictionary
    ops = {
        'iterators': iterators
    }

    print("+++++++++++++++++++++++++++++++")

    # Qualitative evaluation
    eval_run(epoch, MODEL, diff_renderer, ops)
    # eval_run_traversal(epoch, diff_renderer, mode='shape', n_trav_steps=2, trav_fact=0.2)
    # eval_run_traversal(epoch, diff_renderer, mode='extr', n_trav_steps=4, trav_fact=1.)


def eval_run(epoch, net, renderer, ops):

    num_batches = (DATA_VAL.get_size() // CONFIG['training']['batch_size'])+1

    names = []
    img_list = []
    depth_list = []
    normal_list = []
    occ_list = []
    occ_soft_list = []
    gt_slices = []
    pred_slices = []

    for _ in range(num_batches):

        input_batch = next(ops['iterators']['val'])
        input_batch = net.get_input(input_batch)

        z_shape = net.get_latent(input_batch['scene_ids'])[:, 0]    # (BS, D)

        for i in range(CONFIG['training']['batch_size']):

            name = DATA_VAL.scene_names[input_batch['scene_ids'][i][0]]
            if name in names:
                continue
            names.append(name)

            # RGB image
            rgb_gt = input_batch['rgb_gt'][i]
            n_img, img_size, _, _ = rgb_gt.shape
            rgb_gt = (np.transpose(rgb_gt, (1, 0, 2, 3))).reshape((img_size, n_img*img_size, 3))
            img_list.append(rgb_gt)

            # sdf slices
            slice_list = []
            for a in ['x', 'y', 'z']:
                slice_coords = gen_slice(a, size=img_size)
                output_slices = net([slice_coords, input_batch['scene_ids'][i:i+1]])
                pred_slice = np.reshape(output_slices['sdf'], [-1, img_size, img_size, 1])

                pred_slice = viz.slice_coloring(pred_slice, col_scale=5.)
                pred_slice = np.flip(pred_slice, axis=1)

                slice_list.append(pred_slice)
            pred_slices.append(np.concatenate(slice_list, axis=0))  # (3, H, W, 1)
            gt_slices.append(input_batch['slice_gt'][i])

            # renderings
            z_extr = np.asarray([[2., 0., 0., 0., 1.]], dtype=np.float32)
            output = renderer(z_shape[i:i+1], z_extr)

            depth = output['depth']
            depth_list.append(depth)

            normal = viz.create_normal_img(depth)
            normal_list.append(normal[0])

            occlusion = output['occlusion']
            occlusion = np.expand_dims(occlusion, axis=-1)
            occ_list.append(occlusion[0])

    rgb_path = os.path.join(EVAL_DIR, 'ep{}_rgb_gt.png'.format(epoch))
    slice_path = os.path.join(EVAL_DIR, 'ep{}_slices.png'.format(epoch))
    depth_path = os.path.join(EVAL_DIR, 'ep{}_depth.png'.format(epoch))
    normal_path = os.path.join(EVAL_DIR, 'ep{}_normal.png'.format(epoch))
    occ_path = os.path.join(EVAL_DIR, 'ep{}_occ.png'.format(epoch))

    viz.show_img_list(img_list, rgb_path, axis=0)
    viz.show_slice_list(gt_slices, pred_slices, slice_path)
    viz.show_depth_list(depth_list, depth_path, use_color_map=False)
    viz.show_img_list(normal_list, normal_path, axis=0)
    viz.show_occ_list(occ_list, occ_path, axis=0)

    print('Images rendered for {} objects'.format(len(names)))


def eval_run_traversal(epoch, renderer, mode, n_trav_steps, trav_fact):

    z_shape = np.zeros((1, CONFIG['model'][model_base_name]['dim_latent']), dtype=np.float32)
    z_extr = np.asarray([[1., 0., 0., 0., 0.]], dtype=np.float32)

    if mode == 'shape':
        n_dims = z_shape.shape[1]
    elif mode == 'extr':
        n_dims = z_extr.shape[1]
    else:
        print('[ERROR] Latent traversal not possible for mode ', mode)

    depth_obj_trav_list = []
    normal_obj_trav_list = []
    for l in range(n_dims):
        depth_obj_trav_list.append([])
        normal_obj_trav_list.append([])

        for t in range(-n_trav_steps, n_trav_steps + 1):
            z_shape_update = np.copy(z_shape)
            z_extr_update = np.copy(z_extr)
            if mode == 'shape':
                z_shape_update[:, l] += trav_fact * t / n_trav_steps
            elif mode == 'extr':
                z_extr_update[:, l] += trav_fact * t / n_trav_steps

            output = renderer(z_shape_update, z_extr_update)

            depth = output['depth'][0]
            depth = np.nan_to_num(depth.numpy())
            depth_obj_trav_list[l].append(depth)

            normal = viz.create_normal_img(output['depth'])
            normal_obj_trav_list[l].append(normal[0])

    depth_path = os.path.join(EVAL_DIR, 'ep{}_traversal_{}_depth.png'.format(epoch, mode))
    depth_obj_trav_list = np.asarray(depth_obj_trav_list)
    viz.show_depth_list(depth_obj_trav_list, depth_path)

    normal_path = os.path.join(EVAL_DIR, 'ep{}_traversal_{}_normal.png'.format(epoch, mode))
    normal_obj_trav_list = np.asarray(normal_obj_trav_list)
    viz.show_img_list(normal_obj_trav_list, normal_path)

    print('Traversal images rendered for', mode)


# --------------------------------------------------
# ---


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

    # Load model and config file
    LOG_DIR = os.path.join(FLAGS.log_dir, FLAGS.dataset + '_' + FLAGS.model + '(' + FLAGS.message + ')')
    if not os.path.exists(os.path.join(LOG_DIR, 'ckpts', 'checkpoint')):
        print('No pre-trained model found at ', LOG_DIR)
        exit()

    # # TMP: -- new cnfg file
    # cnfg_file = os.path.join('config', FLAGS.config + '.py')
    # config_module = importlib.import_module('.' + FLAGS.config, 'config')
    # CONFIG = config_module.cnfg_dict
    # -- std
    cnfg_file = os.path.join(LOG_DIR, FLAGS.config + '.pkl')
    CONFIG = load_config(cnfg_file)

    model_file = os.path.join(LOG_DIR, model_base_name + '.py')
    model_module = load_module_from_log(model_base_name, model_file)
    MODEL = model_module.get_model(CONFIG['model']['deepsdf'], model_ext_name)

    # Create evaluation directory
    EVAL_DIR = os.path.join(LOG_DIR, 'eval')
    if not os.path.exists(EVAL_DIR):
        os.mkdir(EVAL_DIR)

    # Open Log-file
    LOG_FILE = LogFile(os.path.join(LOG_DIR, 'log_eval.txt'))
    LOG_FILE.write(str(FLAGS))
    LOG_FILE.write('{:%d.%m_%H:%M}'.format(datetime.now()))

    # Load data
    DATA_VAL = data_provider.DataSetShapeDec(FLAGS.data_dir, FLAGS.split, CONFIG['data'])

    main()
    LOG_FILE.write('--')

    exit(0)
