import os
import psutil
import gc
import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from math import pi
from termcolor import colored
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from render_and_compare import *

# Disable TF info messages (needs to be done before tf import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from utils import data_provider
import utils.viz as viz
from utils.shared_funcs import *
from utils.tf_funcs import *
from preprocess.scale_off import *
import models.renderer as renderer

# GPU should not be allocated entirely at beginning
# -- https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

MANIPULATION_MODE = ''  # ['', 'switch', 'sample', 'remove', 'shape', 'texture', 'mixed', 'pos']


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clevr', help='Data Set')
    parser.add_argument('--data_dir', help='Data dir')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--eval_dir', default='eval', help='Log dir [default: eval]')
    parser.add_argument('--model', default='mosnet-seq', help='Model name [mosnet-seq, mosnet-var]')
    parser.add_argument('--config', default='', help='Configuration file [cnfg_<model>_<dataset>]')
    parser.add_argument('--no_gpu', default='0', help='Do not use GPU')
    parser.add_argument('--split', default='val', help='[train, val]')
    parser.add_argument('--message', default='', help='Message that specifies settings etc. (for log dir name)')
    return parser.parse_args()


# --------------------------------------------------
# --- Main Training

def main():

    # Create dataset iterator for training and validation
    dataset_val, iterator_val = get_data_iterator(DATA_VAL, CONFIG['training'])
    iterators = {
        'val': iterator_val
    }

    print("+++++++++++++++++++++++++++++++")

    # Operator to restore all the variables.
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=MODEL)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(LOG_DIR, 'ckpts'), max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        epoch = int(ckpt.step)
        LOG_FILE.write("Restored from {}, epoch {}".format(manager.latest_checkpoint, epoch))
    else:
        print(colored('[ERROR] eval.py, no pre-trained model found at ' + LOG_DIR, 'red'))

    # Create ops dictionary
    ops = {
        'iterators': iterators
    }

    print("+++++++++++++++++++++++++++++++")

    if r_and_c: #Load the optimized latents from file
        opt_latents_file = open(os.path.join(LOG_DIR, 'eval', 'latents_21.npy'), 'rb')
        for i in range(1000):
            # [shape[8], tex[7], scale, posx, posy, posz, angle] for the 3 objects in the scene
            latent_from_file = np.load(opt_latents_file)
            optimized_latents_21[i, :, :] = latent_from_file[0, :, 0, :]

        opt_latents_file = open(os.path.join(LOG_DIR, 'eval', 'latents_20.npy'), 'rb')
        for i in range(1000):
            # [shape[8], tex[7], scale, posx, posy, posz, angle] for the 3 objects in the scene
            latent_from_file = np.load(opt_latents_file)
            optimized_latents_20[i, :, :] = latent_from_file[0, :, 0, :]

    # # Qualitative Evaluation
    max_n_imgs = 15
    # img_types = ['rgb', 'depth', 'depth_diff', 'normal', 'inst']  # , 'obj_msk', 'obj_rgb', 'obj_depth', 'obj_slice']
    img_types = ['rgb', 'depth', 'normal']
    #eval_run_single_imgs(epoch, MODEL, ops, img_types=img_types, max_n_imgs=max_n_imgs)
    _, iterators['val'] = get_data_iterator(DATA_VAL, CONFIG['training'])

    #eval_run_combi_imgs(epoch, max_n_imgs=max_n_imgs)
    _, iterators['val'] = get_data_iterator(DATA_VAL, CONFIG['training'])
    # # generate_obj_meshes(ops, start_epoch)

    # # Quantitative Evaluation
    eval_stats(epoch, MODEL, ops)
    _, iterators['val'] = get_data_iterator(DATA_VAL, CONFIG['training'])
    # eval_runtime(MODEL, ops)

    eval_3d_reconstruction(epoch, '../data_basic/clevr_christian/', ops, single_objs=False)
    _, iterators['val'] = get_data_iterator(DATA_VAL, CONFIG['training'])
    eval_3d_reconstruction(epoch, '../data_basic/clevr_christian/', ops, single_objs=True)


def get_scene_name(input_batch, i):
    """
    :param inputs_val:
    :param i:
    :return:
    """
    scene_id = input_batch['scene_ids'][i, 0].numpy()
    # if scene_id < len(DATA_VAL.scene_names):
    #     name = DATA_VAL.scene_names[scene_id]
    #     return name
    # else:
    #     return str(scene_id).zfill(6)
    return str(scene_id).zfill(6)


def eval_run_single_imgs(epoch, net, ops, img_types, max_n_imgs=100):
    """
    :param epoch:       training epoch at which model is evaluated
    :param net:         model that is evaluated
    :param ops:         dictionary of tensorflow operation
    :param img_types:   list of possible outputs from
                        ['rgb', 'depth', 'normal', 'inst_seg', 'att', 'obj_occ', 'obj_rec', 'obj_depth']
    :param max_n_imgs:  maximum number of scenes for which images are generated
    :return:
    """
    num_batches = (DATA_VAL.get_size() // CONFIG['training']['batch_size'])+1
    max_n_batches = (max_n_imgs // CONFIG['training']['batch_size'])+1

    img_size = CONFIG['data']['img_size']
    n_img = CONFIG['model'][model_base_name]['n_imgs']
    n_slots = CONFIG['model'][model_base_name]['n_slots']

    names = []
    imgs_dir = os.path.join(EVAL_DIR, 'imgs_single')
    if MANIPULATION_MODE != '':
        imgs_dir = imgs_dir + '_' + MANIPULATION_MODE
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    path_placeholder = os.path.join(imgs_dir, FLAGS.split + '_ep{}_{}_{}_{}.png')

    for _ in range(min(num_batches, max_n_batches)):

        input_batch = next(ops['iterators']['val'])
        input_batch = net.get_input(input_batch)
        output_batch = net([input_batch['rgb_in']])


        # msk_gt_plot = np.stack([input_batch['msk_gt'][1,0,0,:,:], input_batch['msk_gt'][1,1,0,:,:], input_batch['msk_gt'][1,2,0,:,:]], axis=-1)
        # plt.imshow(msk_gt_plot)
        # plt.show()
        # plt.imshow(input_batch['rgb_in'][1, 0, ...])
        # plt.show()
        # plt.imshow(input_batch['rgb_gt'][1, 0, ...])
        # plt.show()
        # plt.imshow(input_batch['depth_gt'][1,0,:,:])
        # plt.show()

        for i in range(CONFIG['training']['batch_size']):

            if len(names) >= max_n_imgs:
                break
            name = get_scene_name(input_batch, i)
            if name in names:
                continue
            names.append(name)

            # reconstructed image
            if 'rgb' in img_types:
                rgb_gt = input_batch['rgb_gt'][i]
                rgb_pred = output_batch['rgb_pred'][i]
                rgb_gt = (np.transpose(rgb_gt, (1, 0, 2, 3))).reshape((img_size, n_img * img_size, 3))
                rgb_pred = (np.transpose(rgb_pred, (1, 0, 2, 3))).reshape((img_size, n_img*img_size, 3))
                viz.show_image(rgb_gt, path_placeholder.format(epoch, name, 'rgb', 'gt'))
                viz.show_image(rgb_pred, path_placeholder.format(epoch, name, 'rgb', 'pred'))

            # depth
            if 'depth' in img_types:
                depth_in = input_batch['depth_gt'][i]
                depth_pred = output_batch['depth_pred'][i]  # [(N_depth, H, W, 1) for all objects]
                depth_in = (np.transpose(depth_in, (1, 0, 2))).reshape((n_img, img_size, img_size, 1))
                viz.show_depth_list([depth_in], path_placeholder.format(epoch, name, 'depth', 'gt'))
                viz.show_depth_list([depth_pred], path_placeholder.format(epoch, name, 'depth', 'pred'))
                if 'depth_diff' in img_types:
                    depth_diff = depth_pred - depth_in
                    depth_diff = viz.slice_coloring(depth_diff[0, :, :, 0])
                    viz.show_image(depth_diff, path_placeholder.format(epoch, name, 'depth', 'diff'))

            # normal
            if 'normal' in img_types:
                normal_gt = viz.create_normal_img(input_batch['depth_gt'][i])
                normal_pred = viz.create_normal_img(output_batch['depth_pred'][i])

                normal_gt = (np.transpose(normal_gt, (1, 0, 2, 3))).reshape((img_size, n_img*img_size, 3))
                normal_pred = (np.transpose(normal_pred, (1, 0, 2, 3))).reshape((img_size, n_img*img_size, 3))
                viz.show_image(normal_gt, path_placeholder.format(epoch, name, 'normal', 'gt'))
                viz.show_image(normal_pred, path_placeholder.format(epoch, name, 'normal', 'pred'))

            # instance segmentation, TODO: only for one image atm
            if 'inst' in img_types:
                msk_gt = np.expand_dims(input_batch['msk_gt'][i, :, 0], axis=-1)            # (S, H, W, 1)
                msk_pred = output_batch['msk_pred'][i, :, 0]                                # (S, H, W, 1)
                inst_gt = np.zeros((img_size, img_size, 3))
                inst_pred = np.zeros((img_size, img_size, 3))
                colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0.75, 0.75, 0.], [0., 0.75, 0.75]]

                for n in range(n_slots):
                    occ_gt = np.tile(msk_gt[n], (1, 1, 3))
                    occ_pred = np.tile(msk_pred[n], (1, 1, 3))
                    col = np.expand_dims(np.expand_dims(np.asarray(colors[n]), axis=0), axis=0)
                    inst_gt = inst_gt + col*occ_gt
                    inst_pred = inst_pred + col*occ_pred

                viz.show_image(inst_gt, path_placeholder.format(epoch, name, 'inst', 'gt'))
                viz.show_image(inst_pred, path_placeholder.format(epoch, name, 'inst', 'pred'))

            # object-wise results
            for n in range(n_slots):
                if 'obj_rgb' in img_types:
                    obj_rgb = output_batch['obj_rgb_pred'][i, n, 0]         # (-, S, N_img, H, W, 3) -> (H, W, 3)
                    viz.show_image(obj_rgb.numpy(), path_placeholder.format(epoch, name, 'obj_rgb_pred', n))

                if 'obj_msk' in img_types:
                    obj_msk = output_batch['obj_msk_pred'][i, n, 0]         # (-, S, N_img, H, W, 1) -> (H, W, 1)
                    viz.show_occ_list([obj_msk], path_placeholder.format(epoch, name, 'obj_msk_pred', n))

                if 'obj_depth' in img_types:
                    obj_depth = output_batch['obj_depth_pred'][i, n]        # (-, S, N_img, H, W, 3) -> (N_img, H, W, 1)
                    obj_normal = viz.create_normal_img(obj_depth)[0]
                    viz.show_image(obj_normal, path_placeholder.format(epoch, name, 'obj_normal_pred', n))

                if 'obj_slice' in img_types:
                    z_shape = output_batch['z_shape_mean']
                    z_shape = z_shape[i:i+1, n, 0, :CONFIG['model'][model_base_name]['dim_latent_split'][0]]
                    z_shape = tf.tile(tf.expand_dims(z_shape, axis=1), (1, img_size*img_size, 1))

                    slice_list = []
                    for a in ['x', 'y', 'z']:
                        slice_coords = gen_slice(a, size=img_size)
                        output_slices = net.decoder_sdf([slice_coords], latent=z_shape)
                        pred_slice = np.reshape(output_slices['sdf'], [-1, img_size, img_size, 1])

                        pred_slice = viz.slice_coloring(pred_slice, col_scale=5.)
                        pred_slice = np.flip(pred_slice, axis=1)

                        slice_list.append(pred_slice)
                    obj_slice_pred = np.concatenate(slice_list, axis=0)  # (3, H, W, 1)

                    slice_path = path_placeholder.format(epoch, name, 'slice_pred', n)
                    viz.show_slice_list(pred_slices=obj_slice_pred, path=slice_path)


def eval_run_combi_imgs(epoch, max_n_imgs=10):

    n_slots = CONFIG['model'][model_base_name]['n_slots']

    imgs_single_dir = os.path.join(EVAL_DIR, 'imgs_single')
    if MANIPULATION_MODE != '':
        imgs_single_dir = imgs_single_dir + '_' + MANIPULATION_MODE
    if not os.path.exists(imgs_single_dir):
        os.makedirs(imgs_single_dir)

    # get list of pre-rendered single images
    tmp_names = ['_'.join(p.split('_')[2:-2]) for p in os.listdir(imgs_single_dir) if 'rgb_pred' in p]
    tmp_names.sort()
    if max_n_imgs < len(tmp_names):
        tmp_names = tmp_names[:max_n_imgs]

    path_placeholder = os.path.join(imgs_single_dir, FLAGS.split + '_ep{}_{}_{}.png')
    obj_path_placeholder = os.path.join(imgs_single_dir, FLAGS.split + '_ep{}_{}_{}_{}.png')

    dict_pathes = {}
    scene_types = ['rgb', 'depth', 'normal', 'inst']
    object_types = ['objrgb', 'occ_pred']  # further options: attention, slices
    add_types = ['depth_diff']
    for t in scene_types + object_types + add_types:
        dict_pathes[t] = []

    # for n in range(max_n_imgs):
    for n in tmp_names:

        # Scene Reconstruction
        for t in scene_types:
            tmp_gt_path = path_placeholder.format(epoch, n, t+'_gt')
            tmp_pred_path = path_placeholder.format(epoch, n, t+'_pred')
            if os.path.exists(tmp_gt_path) and os.path.exists(tmp_pred_path):
                dict_pathes[t].append(tmp_gt_path)
                dict_pathes[t].append(tmp_pred_path)

        # Object Reconstructions
        for t in object_types:
            for m in range(n_slots):
                obj_path = obj_path_placeholder.format(epoch, n, t, m)
                if os.path.exists(obj_path):
                    dict_pathes[t].append(obj_path)

        # Additional Images
        for t in add_types:
            tmp_path = path_placeholder.format(epoch, n, t)
            if os.path.exists(tmp_path):
                dict_pathes[t].append(tmp_path)

    def combine_img_list(path_list, name, layout=None):
        if len(path_list) == 0:
            return
        img_path = os.path.join(EVAL_DIR, FLAGS.split + '_ep{}_{}.png'.format(epoch, name))
        if MANIPULATION_MODE != '':
            img_path = img_path.replace('.png', '_' + MANIPULATION_MODE + '.png')
        viz.combine_images(path_list, img_path, remove_imgs=0, layout=layout)
        print('Combined {} visualizations generated for {} images.'.format(name, len(path_list)))

    scene_layout = [int(len(dict_pathes['rgb']) / 2), 2]
    for t in scene_types:
        combine_img_list(dict_pathes[t], t, scene_layout)

    if len(dict_pathes['objrgb']) % n_slots != 0:
        print('[WARNING] eval.eval_run_combi_imgs(): Missing object images: {}'.format(len(dict_pathes['objrgb'])))
    obj_layout = [int(len(dict_pathes['objrgb']) / n_slots), n_slots]
    for t in object_types:
        combine_img_list(dict_pathes[t], t, obj_layout)

    for t in add_types:
        combine_img_list(dict_pathes[t], t)


def eval_stats(epoch, net, ops):

    num_batches = (DATA_VAL.get_size() // CONFIG['training']['batch_size'])

    n_slots = CONFIG['model'][model_base_name]['n_slots']

    min_inst_pixel = 25
    rot_symmetrie = [0, pi / 2., pi, 3 * pi / 2.]
    # rot_symmetrie = [0, pi]
    # rot_symmetrie = [0]
    b = 10 if (num_batches < 150) else 50

    names = []
    res = {}
    # reconstruction
    rms_error_list = []
    ssim_list = []
    psnr_list = []
    # depth
    depth_abs_rel_diff_list = []
    depth_squ_rel_diff_list = []
    depth_rmse_diff_list = []
    # instance segmentation
    inst_ap05_list = []
    inst_ar05_list = []
    inst_f1_05_list = []
    inst_ap_list = []
    n_found_obj_list = []
    # pose estimation
    match_id_dict = {}
    pos_3d_list = []
    extr_x_list = []
    extr_y_list = []
    extr_z_list = []
    extr_s_list = []
    extr_r_list = []
    extr_r_sym_list = []
    extr_r_gt_org_list = []
    extr_r_pred_org_list = []
    extr_r_pred_big_err_org_list = []
    extr_r_gt_big_err_org_list = []

    for batch_id in range(num_batches):

        if not r_and_c:
            if batch_id % b == 0:
                print('Current batch/total batch num: {}/{}'.format(batch_id, num_batches))
        else:
            print('Current batch/total batch num: {}/{}'.format(batch_id, num_batches))

        input_batch = next(ops['iterators']['val'])
        input_batch = net.get_input(input_batch)

        if r_and_c:
            latents = optimized_latents_21[batch_id, :, :] #The optimized latents need to have already been computed before
            latents = np.expand_dims(latents, axis=0)
            latents = np.expand_dims(latents, axis=2)
            latents_var = tf.Variable(latents.astype('float32'))

            output_batch = net.render_objs(latents_var)

            z_obj_all = latents #K.stack(z_obj_list, axis=1)  # (BS, N_slots, N_img, D)
            z_dict_tmp = {
                'z_shape': z_obj_all[..., :net.dim_shape],                                    # (BS, S, N_img, D_shape)
                'z_tex': z_obj_all[..., net.dim_shape:net.dim_shape + net.dim_app - 1],   # (BS, S, N_img, D_tex)
                'z_s': output_batch['z_extr'][..., :1],                                                  # (BS, S, N_img, D_s=1)
                'z_pos': output_batch['z_extr'][..., 1:4],                                               # (BS, S, N_img, D_pos=3)
                'z_pos_x': output_batch['z_extr'][..., 1:2],                                             # (BS, S, N_img, 1)
                'z_pos_y': output_batch['z_extr'][..., 2:3],                                             # (BS, S, N_img, 1)
                'z_pos_z': output_batch['z_extr'][..., 3:4],                                             # (BS, S, N_img, 1)
                'z_rot': output_batch['z_extr'][..., 4:],                                                # (BS, S, N_img, D_rot=1)
            }
            for k, v in z_dict_tmp.items():
                output_batch[k + '_mean'] = v

            output_batch['z_all'] = z_obj_all

            bg_img = net.predict_bg(input_batch['rgb_in'])
            depth_ordering = net.depth_ordering(output_batch, bg_img)

            for k, v in depth_ordering.items():
                output_batch[k] = v

            output_batch = net.crit_obj_intersection(output_batch)
            output_batch = net.crit_ground(output_batch)

            if CONFIG['model']['mosnet']['anti_aliasing']:
                rgb_pred = net.downsample_render(output_batch['rgb_pred'], antialiasing=True)
                depth_pred = net.downsample_render(output_batch['depth_pred'], antialiasing=True)
                msk_pred = net.downsample_render(output_batch['msk_pred'], antialiasing=True)
                obj_rgb_pred = net.downsample_render(output_batch['obj_rgb_pred'], antialiasing=True)
                obj_depth_pred = net.downsample_render(output_batch['obj_depth_pred'], antialiasing=True)
                obj_msk_pred = net.downsample_render(output_batch['obj_msk_pred'], antialiasing=True)
                output_batch['rgb_pred'] = rgb_pred
                output_batch['depth_pred'] = depth_pred
                output_batch['msk_pred'] = msk_pred
                output_batch['obj_rgb_pred'] = obj_rgb_pred
                output_batch['obj_depth_pred'] = obj_depth_pred
                output_batch['obj_msk_pred'] = obj_msk_pred

                # plt.imshow(rgb_pred[0, 0, ...])
                # plt.show()
        else:

            #print(datetime.now())
            output_batch = net([input_batch['rgb_in']])
            #print(datetime.now())
            #print("\n")

        for i in range(CONFIG['training']['batch_size']):

            name = get_scene_name(input_batch, i)
            if name in names:
                continue
            names.append(name)

            # --- RGB Image
            rgb_gt = input_batch['rgb_gt'][i].numpy()      # (N_img, H, W, 3)
            rgb_pred = output_batch['rgb_pred'][i].numpy()

            rms_error = np.sqrt(np.mean(np.square(rgb_gt-rgb_pred)))
            rms_error_list.append(rms_error)

            ssim_val = structural_similarity(rgb_gt[0], rgb_pred[0], multichannel=True)     # TODO only one view atm
            ssim_list.append(ssim_val)

            psnr_val = peak_signal_noise_ratio(rgb_gt[0], rgb_pred[0])                      # TODO only one view atm
            psnr_list.append(psnr_val)

            # --- Depth
            depth_gt = np.expand_dims(input_batch['depth_gt'][i].numpy(), axis=-1)  # (N_img, H, W, 1)
            depth_pred = output_batch['depth_pred'][i].numpy()

            abs_rel_diff = np.mean(np.abs(depth_pred-depth_gt)/depth_gt)
            depth_abs_rel_diff_list.append(abs_rel_diff)

            squ_rel_diff = np.mean(np.square(depth_pred-depth_gt)/depth_gt)
            depth_squ_rel_diff_list.append(squ_rel_diff)

            rmse_depth = np.sqrt(np.mean(np.square(depth_pred-depth_gt)))
            depth_rmse_diff_list.append(rmse_depth)

            # -- Instance, TODO: only for one image atm
            tmp_ap_ist = []
            msk_gt = input_batch['msk_gt'][i, :, 0].numpy()
            msk_pred = output_batch['msk_pred'][i, :, 0].numpy()
            for th in np.arange(0.5, 1., 0.05):
                inst_res = inst_ap(msk_gt, msk_pred, th, min_n_pxls=min_inst_pixel)
                ap_val = inst_res[0]
                tmp_ap_ist.append(ap_val)
            inst_ap_list.append(np.mean(tmp_ap_ist))

            ap_05, ar_05, f1_05, match_id, n_found_obj = inst_ap(msk_gt, msk_pred, 0.5, min_n_pxls=min_inst_pixel)
            inst_ap05_list.append(ap_05)
            inst_ar05_list.append(ar_05)
            inst_f1_05_list.append(f1_05)
            n_found_obj_list.append(n_found_obj)

            new_match_id = []
            used = []
            for m in match_id:
                pred_extr = output_batch['z_extr'][i, m[0], 0].numpy()
                pred_extr = pred_extr[1:4]
                d_list = []
                for j in range(n_slots):
                    if j in used:  # greedy version...
                        continue
                    gt_extr = input_batch['z_extr_gt'][i, j, 0].numpy()
                    d_list.append([j, np.sqrt(np.sum(np.square(pred_extr - gt_extr[1:4])))])
                d_list = np.asarray(d_list)
                argmin_d = np.argmin(d_list[:, 1])
                min_d = d_list[argmin_d, 0]
                new_match_id.append([m[0], int(min_d)])
                if min_d in used:
                    print('ERROR: pose error, gt object was assigned multiple times..', name)
                used.append(min_d)

            for m in new_match_id:
                pred_extr = output_batch['z_extr'][i, m[0], 0].numpy()
                gt_extr = input_batch['z_extr_gt'][i, m[1], 0].numpy()

                pos_3d_list.append(np.sqrt(np.sum(np.square(pred_extr[1:4] - gt_extr[1:4]))))
                extr_x_list.append((pred_extr[1] - gt_extr[1]))
                extr_y_list.append((pred_extr[2] - gt_extr[2]))
                extr_z_list.append((pred_extr[3] - gt_extr[3]))
                extr_s_list.append(pred_extr[0] / gt_extr[0])

                pred_r = (pred_extr[4] + 1) * pi
                gt_r = (gt_extr[4] + 1) * pi
                err_rot = min(2*pi-np.abs(pred_r - gt_r), np.abs(pred_r - gt_r))
                extr_r_list.append(err_rot)

                tmp = []
                for rs in rot_symmetrie:
                    pred_r_new = (pred_r + rs) % (2 * pi)
                    tmp.append(min(2*pi-np.abs(pred_r_new - gt_r), np.abs(pred_r_new - gt_r)))
                err_rot_sym = min(tmp)
                extr_r_sym_list.append(err_rot_sym)

                # TODO: tmp
                extr_r_gt_org_list.append(gt_extr[4])
                extr_r_pred_org_list.append(pred_extr[4])
                if err_rot > pi/4.:
                    extr_r_pred_big_err_org_list.append(pred_extr[4])
                    extr_r_gt_big_err_org_list.append(gt_extr[4])

                assert(err_rot_sym <= err_rot)

            match_id_dict[name] = match_id

    print('summarize evaluation...')

    n_found_obj_list = np.asarray(n_found_obj_list)
    un, un_counts = np.unique(n_found_obj_list, return_counts=True)

    res['epoch'] = epoch
    res['data_n_imgs'] = epoch
    res['min_inst_pixel'] = min_inst_pixel
    res['split'] = FLAGS.split
    res['data_dir'] = FLAGS.data_dir
    res['data_n_imgs'] = len(rms_error_list)

    res['n_found_obj'] = str([un, un_counts])
    res['inst_ap'] = str(np.mean(inst_ap_list))
    res['inst_ap05'] = str(np.mean(inst_ap05_list))
    res['inst_ar05'] = str(np.mean(inst_ar05_list))
    res['inst_f1-05'] = str(np.mean(inst_f1_05_list))

    res['rgb_rms'] = str(np.mean(rms_error_list))
    res['rgb_psnr'] = str(np.mean(psnr_list))
    res['rgb_ssim'] = str(np.mean(ssim_list))

    res['depth_rmse'] = str(np.mean(depth_rmse_diff_list))
    res['depth_abs_rel_diff'] = str(np.mean(depth_abs_rel_diff_list))
    res['depth_squ_rel_diff'] = str(np.mean(depth_squ_rel_diff_list))

    res['pos_3d'] = str(np.mean(pos_3d_list))
    res['extr_r'] = str(np.mean(extr_r_list)*(360/(2*pi)))
    res['extr_r_med'] = str(np.median(extr_r_list)*(360/(2*pi)))
    res['extr_r_sym'] = str(np.mean(extr_r_sym_list)*(360/(2*pi)))
    res['extr_r_sym_med'] = str(np.median(extr_r_sym_list)*(360/(2*pi)))
    res['r_sym_mode'] = rot_symmetrie
    res['extr_x'] = str([np.mean(extr_x_list), np.var(extr_x_list)])
    res['extr_y'] = str([np.mean(extr_y_list), np.var(extr_y_list)])
    res['extr_z'] = str([np.mean(extr_z_list), np.var(extr_z_list)])
    res['extr_s'] = str([np.mean(extr_s_list), np.var(extr_s_list)])
    # res['match_id_list'] = match_id_dict

    path = os.path.join(EVAL_DIR, FLAGS.split + '_ep{}.json'.format(epoch))
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)

    # Plot rotation histogram (Temporary)
    n_bins = 72
    print('histogramm...')
    fig, axs = plt.subplots(5, 1, constrained_layout=True)
    axs[0].hist(extr_r_list, bins=n_bins, range=(0., pi))
    axs[0].set_title('rotation errors')
    axs[1].hist(extr_r_gt_org_list, bins=n_bins, range=(-1., 1.))
    axs[1].set_title('all gt rotation (org)')
    axs[2].hist(extr_r_pred_org_list, bins=n_bins, range=(-1., 1.))
    axs[2].set_title('all predicted rotation (org)')
    axs[3].hist(extr_r_gt_big_err_org_list, bins=n_bins, range=(-1., 1.))
    axs[3].set_title('gt rotation with big errors')
    axs[4].hist(extr_r_pred_big_err_org_list, bins=n_bins, range=(-1., 1.))
    axs[4].set_title('pred rotation with big errors')
    hist_path = os.path.join(EVAL_DIR, FLAGS.split + '_ep{}_hist.png'.format(epoch))
    plt.savefig(hist_path)
    # plt.show()


def eval_runtime(net, ops):

    num_batches = (DATA_VAL.get_size() // CONFIG['training']['batch_size'])+1

    time_start = datetime.now()
    time_start_str = '{:%d.%m_%H:%M}'.format(time_start)
    print('Start Time:', time_start_str)

    for _ in range(num_batches):

        input_batch = next(ops['iterators']['val'])
        input_batch = net.get_input(input_batch)
        output_batch = net([input_batch['rgb_in']])

    time_stop = datetime.now()
    time_delta = time_stop - time_start
    time_stop_str = '{:%d.%m_%H:%M}'.format(time_stop)
    print('Stop Time:', time_stop_str)

    print('Time Delta [total, per image]:', time_delta, time_delta/(num_batches*CONFIG['training']['batch_size']))

def eval_3d_reconstruction(epoch, input_data_dir, ops, single_objs=False):
    if single_objs:
        iou_array = np.zeros(DATA_VAL.get_size()*3)
        valid_matches = np.zeros(DATA_VAL.get_size()*3)
    else:
        iou_array = np.zeros(DATA_VAL.get_size())

    batch_size = CONFIG['training']['batch_size']
    num_batches = (DATA_VAL.get_size() // CONFIG['training']['batch_size'])

    for batch in range(num_batches):

        input_batch = next(ops['iterators']['val'])
        input_batch = MODEL.get_input(input_batch)

        #Compute output from model:
        if r_and_c:
            latents = optimized_latents_20[batch, :, :] #The optimized latents need to have already been computed before
            latents = np.expand_dims(latents, axis=0)
            latents = np.expand_dims(latents, axis=2)
        else:
            output_batch = MODEL([input_batch['rgb_in']])
            latents = tf.concat([output_batch['z_shape_mean'], output_batch['z_tex_mean'], output_batch['z_s_mean'], output_batch['z_pos_x_mean'],  output_batch['z_pos_y_mean'],  output_batch['z_pos_z_mean'], output_batch['z_rot_mean']], axis=-1).numpy()

        for i in range(batch_size):
            print("3D IoU eval on scene {}".format(batch*batch_size+i))

            latents_scene = latents[i, :, 0, :]

            if single_objs:
                # In the GT obj files, the y and z axes are swapped, hence the different bounding box parameters for the different function calls
                voxels_pred = pred_sdf_to_voxels(MODEL.renderer, latents_scene, [-3, 3, -3, 3, -1, 2], 20, single_objs=True, gen_plots=False)
                voxels_gt, obj_centers_gt = point_cloud_to_voxels(input_data_dir+'obj_files/CLEVR_new_{}_0.obj'.format(str(batch*batch_size+i).zfill(6)), [-3, 3, -1, 2, -3, 3], 20, single_objs=True, objs_permutation=np.array([0, 1, 2]), gen_plots=False)

                obj_centers_pred = latents_scene[:, 16:19]
                unused_objs = [0, 1, 2]
                permutation = []
                matching_thresh = 0.4   # The maximum L2 distance between GT and prediction object centers below which the objects are considered a good match
                for obj_gt_id in range(3):
                    curr_obj_center_gt = obj_centers_gt[obj_gt_id, :]
                    matching_obj = -1
                    min_l2_dist = np.inf
                    for obj_pred_id in unused_objs:
                        curr_obj_center_pred = obj_centers_pred[obj_pred_id, :]
                        l2_distance = np.sum(np.square(curr_obj_center_gt-curr_obj_center_pred))
                        if l2_distance < min_l2_dist and l2_distance < matching_thresh:
                            matching_obj = obj_pred_id
                            min_l2_dist = l2_distance
                    permutation.append(matching_obj)
                    if matching_obj > -1:
                        unused_objs.pop(unused_objs.index(matching_obj))

                # print(permutation)
                #
                # plt.imshow(output_batch['rgb_pred'][i, 0, ...])
                # plt.show()
                # plt.imshow(input_batch['rgb_gt'][i, 0, ...])
                # plt.show()
                #
                # _ = point_cloud_to_voxels(input_data_dir+'obj_files/CLEVR_new_{}_0.obj'.format(str(batch*batch_size+i).zfill(6)), [-3, 3, 0, 3, -3, 3], 20, gen_plots=True)
                # _ = pred_sdf_to_voxels(MODEL.renderer, latents_scene, [-3, 3, -3, 3, 0, 3], 20, gen_plots=True)
                #
                # for obj_gt_id in range(3):
                #     fig = plt.figure(dpi=400)
                #     ax = fig.add_subplot(111, projection='3d')
                #     ax.set_xlabel('x')
                #     ax.set_ylabel('y')
                #     ax.set_zlabel('z')
                #     voxels = (voxels_gt[:, :, :, obj_gt_id] != 0)
                #     ax.voxels(voxels, linewidth=0.1)
                #     plt.show()
                #
                #     fig = plt.figure(dpi=400)
                #     ax = fig.add_subplot(111, projection='3d')
                #     ax.set_xlabel('x')
                #     ax.set_ylabel('y')
                #     ax.set_zlabel('z')
                #     voxels = (voxels_pred[:, :, :, permutation[obj_gt_id]] != 0)
                #     ax.voxels(voxels, linewidth=0.1)
                #     plt.show()

                for obj in range(3):
                    if permutation[obj] > -1: # valid match
                        intersection = voxels_gt[:, :, :, obj]*voxels_pred[:, :, :, permutation[obj]]
                        union = voxels_gt[:, :, :, obj]+voxels_pred[:, :, :, permutation[obj]]
                        union[union>0.] = 1.

                        num_voxels_intersection = np.sum(intersection)
                        num_voxels_union = np.sum(union)
                        iou = num_voxels_intersection/num_voxels_union

                        iou_array[(batch*batch_size*3)+(i*3)+obj] = iou
                        valid_matches[(batch*batch_size*3)+(i*3)+obj] = 1.
                    else:
                        iou_array[(batch*batch_size*3)+(i*3)+obj] = 0.
                        valid_matches[(batch*batch_size*3)+(i*3)+obj] = 0.

                # permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
                # best_avg_iou = 0.
                # best_iou_vals = [0, 0, 0]
                # for perm in permutations:
                #     voxels_gt_ = voxels_gt[:, :, :, perm]
                #     avg_iou = 0.
                #     iou_vals = [0, 0, 0]
                #     for obj in range(3):
                #         intersection = voxels_gt_[:, :, :, obj]*voxels_pred[:, :, :, obj]
                #         union = voxels_gt_[:, :, :, obj]+voxels_pred[:, :, :, obj]
                #         union[union>0.] = 1.
                #
                #         num_voxels_intersection = np.sum(intersection)
                #         num_voxels_union = np.sum(union)
                #         iou = num_voxels_intersection/num_voxels_union
                #
                #         avg_iou += iou/3
                #         iou_vals[obj] = iou
                #     if avg_iou > best_avg_iou:
                #         best_avg_iou = avg_iou
                #         best_iou_vals = iou_vals
                # for j in range(len(best_iou_vals)):
                #         iou_array[(batch*batch_size*3)+(i*3)+j] = best_iou_vals[j]
                print("IoU = {}".format(iou_array[(batch*batch_size*3)+(i*3) : (batch*batch_size*3)+(i*3)+3]))

            else:

                # plt.imshow(output_batch['rgb_pred'][i, 0, ...])
                # plt.show()
                # plt.imshow(input_batch['rgb_gt'][i, 0, ...])
                # plt.show()

                # In the GT obj files, the y and z axes are swapped, hence the different bounding box parameters for the different function calls
                voxels_pred = pred_sdf_to_voxels(MODEL.renderer, latents_scene, [-3, 3, -3, 3, 0, 3], 20, gen_plots=False)
                voxels_gt = point_cloud_to_voxels(input_data_dir+'obj_files/CLEVR_new_{}_0.obj'.format(str(batch*batch_size+i).zfill(6)), [-3, 3, 0, 3, -3, 3], 20, gen_plots=False)

                intersection = voxels_gt*voxels_pred
                union = voxels_gt+voxels_pred
                union[union>0.] = 1.

                num_voxels_intersection = np.sum(intersection)
                num_voxels_union = np.sum(union)
                iou = num_voxels_intersection/num_voxels_union

                iou_array[batch*batch_size+i] = iou

                print("IoU = {}".format(iou))

    if single_objs:
        # Around (0, 0, 0) the objects will always trivially intersect; We only consider an object correctly detected
        # and represented if the prediction and the ground truth are close to one another.
        # The mean, stddev, max and min are computed over the correctly detected objects, rather than all of them.

        num_objs_detected = str(np.sum(valid_matches))
        print("Number of objects detected: {}/3000".format(num_objs_detected))
        iou_list_detected_objs = []
        for i in range(np.shape(iou_array)[0]):
            if valid_matches[i] == 1.:
                iou_list_detected_objs.append(iou_array[i])
        iou_array_detected_objs = np.array(iou_list_detected_objs)

        print("Mean IoU: {}".format(np.mean(iou_array_detected_objs)))
        print("Stddev IoU: {}".format(np.std(iou_array_detected_objs)))
        print("Max IoU: {}".format(np.max(iou_array_detected_objs)))
        print("Min IoU: {}".format(np.min(iou_array_detected_objs)))

        res = {}
        res['num_objs_detected'] = num_objs_detected
        res['mean_iou'] = np.mean(iou_array_detected_objs)
        res['stddev_iou'] = np.std(iou_array_detected_objs)
        res['max_iou'] = np.max(iou_array_detected_objs)
        res['min_iou'] = np.min(iou_array_detected_objs)
        res['all_iou_vals'] = []
        for val in iou_array:
            res['all_iou_vals'].append(val)


    else:
        print("Mean IoU: {}".format(np.mean(iou_array)))
        print("Stddev IoU: {}".format(np.std(iou_array)))
        print("Max IoU: {}".format(np.max(iou_array)))
        print("Min IoU: {}".format(np.min(iou_array)))

        res = {}
        res['mean_iou'] = np.mean(iou_array)
        res['stddev_iou'] = np.std(iou_array)
        res['max_iou'] = np.max(iou_array)
        res['min_iou'] = np.min(iou_array)
        res['all_iou_vals'] = []
        for val in iou_array:
            res['all_iou_vals'].append(val)

    path = os.path.join(EVAL_DIR, FLAGS.split + '_ep{}_3d_reconstruction.json'.format(epoch))
    if single_objs: path = os.path.join(EVAL_DIR, FLAGS.split + '_ep{}_3d_reconstruction_single_objs.json'.format(epoch))
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)


def generate_obj_meshes(sess, ops, epoch):

    match_list = {}

    json_path = os.path.join(EVAL_DIR, FLAGS.split + '_ep{}.json'.format(epoch))
    with open(os.path.join(json_path)) as json_file:
        eval_stats_file = json.load(json_file)

        for n, m in eval_stats_file["match_id_list"].items():
            match_list[n] = m

    num_batches = (DATA_VAL.get_size() // CONFIG['training']['batch_size'])+1

    for _ in range(num_batches):

        handle_train = sess.run(ops['iterators']['val'].string_handle())

        output_batch, inputs_val = sess.run([ops['outputs'], ops['inputs_pl']],
                                           feed_dict={ops['handle_pl']: handle_train})

        for i in range(CONFIG['training']['batch_size']):

            name = get_scene_name(inputs_val, i)
            if name not in match_list.keys():
                continue
            m_id_list = match_list[name]

            z = np.expand_dims(output_batch['z_mean'][i], axis=0)
            z_extr = output_batch['z_ext'][i]
            for n in range(z.shape[1]):
                m_id = -1
                for match in m_id_list:
                    if match[0] == n:
                        m_id = match[1]
                if m_id == -1:
                    continue

                mesh_obj_path = os.path.join(EVAL_DIR, 'obj', FLAGS.split + '_mesh_ep{}_{}_obj{}-{}.off'.format(
                    epoch, name, n, m_id))
                if os.path.exists(mesh_obj_path):
                    return

                z_shape = z[:, n, 0, :CONFIG['model'][model_base_name]['dim_latent_split'][0]]

                pred_volume_val = sess.run(ops['pred_volume'], feed_dict={ops['z_shape_pl']: z_shape})

                viz.show_mesh(pred_volume_val, path=None, path_obj=mesh_obj_path)

                s = z_extr[n, 0, 0]
                scale_off_output(mesh_obj_path, os.path.join(EVAL_DIR, 'obj_scaled'), s)


def point_cloud_to_voxels(obj_file, bounding_box, resolution, single_objs=False, objs_permutation=None, gen_plots=False):
    #param resolution: scalar number of voxels per unit length
    #param bounding_box: [min_x, max_x, min_y, max_y, min_z, max_z]; only integer sizes for now

    [min_x, max_x, min_y, max_y, min_z, max_z] = bounding_box
    array_size = [(max_x-min_x)*resolution, (max_y-min_y)*resolution, (max_z-min_z)*resolution]
    if single_objs:
        num_objs = np.shape(objs_permutation)[0]
        array_size = [(max_x-min_x)*resolution, (max_y-min_y)*resolution, (max_z-min_z)*resolution, num_objs]
    voxel_array = np.zeros(array_size)

    if single_objs: #Get the mean vertex of each object in the scene
        f = open(obj_file, 'r')
        mean_vertices = np.zeros([num_objs+1, 3])
        num_vertices_obj = 0
        curr_obj = -1
        for line in f:
            if curr_obj < 1: # The background mesh is the first object in the file; We want to ignore the vertices belonging to it
                if line.startswith('v '):
                    continue
                if line.startswith('g ') or line.startswith('o '):
                    curr_obj += 1
            else:
                if line.startswith('v '): # The current line is a vertex of an object
                    vertex = np.fromstring(line[2:], dtype=float, sep=' ')
                    mean_vertices[curr_obj, :] += vertex
                    num_vertices_obj += 1
                if line.startswith('g ') or line.startswith('o '): # A new object/  group starts
                    mean_vertices[curr_obj] /= num_vertices_obj
                    num_vertices_obj = 0
                    curr_obj += 1
        mean_vertices[curr_obj] /= num_vertices_obj #Divide the sum of the vertices of the last object by its number of vertices

    f = open(obj_file, 'r')
    points_in_bb = 0
    all_pts = 0
    curr_obj = -1
    for line in f:
        if curr_obj < 1: # The background mesh is the first object in the file; We want to ignore the vertices belonging to it
            if line.startswith('v '):
                continue
            if line.startswith('g ') or line.startswith('o '): # A new object/ group starts
                curr_obj += 1
        else:
            if line.startswith('v '): # The current line is a vertex of an object
                all_pts += 1

                vertex = np.fromstring(line[2:], dtype=float, sep=' ')
                if single_objs:
                    vertex -= mean_vertices[curr_obj]
                voxel_x = int(((vertex[0]-min_x)/(max_x-min_x))*array_size[0])
                if voxel_x < 0 or voxel_x >= array_size[0]: continue
                voxel_y = int(((vertex[1]-min_y)/(max_y-min_y))*array_size[1])
                if voxel_y < 0 or voxel_y >= array_size[1]: continue
                voxel_z = int(((vertex[2]-min_z)/(max_z-min_z))*array_size[2])
                if voxel_z < 0 or voxel_z >= array_size[2]: continue


                if voxel_x < 0 or voxel_x >= array_size[0]:
                    print("Warning: GT vertex outside bounding box")
                    exit(-1)
                    continue
                if voxel_y < 0 or voxel_y >= array_size[1]:
                    print("Warning: GT vertex outside bounding box")
                    exit(-1)
                    continue
                if voxel_z < 0 or voxel_z >= array_size[2]:
                    print("Warning: GT vertex outside bounding box")
                    exit(-1)
                    continue


                points_in_bb += 1
                if single_objs:
                    voxel_array[voxel_x, voxel_y, voxel_z, curr_obj-1] = curr_obj
                else:
                    voxel_array[voxel_x, voxel_y, voxel_z] = curr_obj
            if line.startswith('g ') or line.startswith('o '): # A new object/  group starts
                curr_obj += 1

    # Connect the voxels belonging to each object in y-direction:
    if single_objs:
        for obj in range(array_size[3]):
            for x in range(array_size[0]):
                for z in range(array_size[2]):
                    inside_obj = 0 #The number of the object we are inside of, 0 if outside
                    for y in range(array_size[1]):
                        curr_voxel_val = voxel_array[x, y, z, obj]
                        #Entering an objects interior
                        if curr_voxel_val > 0 and inside_obj == 0 and voxel_array[x, y+1, z, obj] == 0. and curr_voxel_val in voxel_array[x,y+1:,z, obj]:
                            inside_obj = curr_voxel_val
                            continue
                        #Leaving an objects interior
                        if curr_voxel_val > 0 and inside_obj > 0 and voxel_array[x, y+1, z, obj] == 0.:
                            inside_obj = 0
                            continue
                        #Filling an objects interior
                        if inside_obj > 0: voxel_array[x, y, z, obj] = inside_obj
    else:
        for x in range(array_size[0]):
            for z in range(array_size[2]):
                inside_obj = 0 #The number of the object we are inside of, 0 if outside
                for y in range(array_size[1]):
                    curr_voxel_val = voxel_array[x, y, z]
                    #Entering an objects interior
                    if curr_voxel_val > 0 and inside_obj == 0 and voxel_array[x, y+1, z] == 0. and curr_voxel_val in voxel_array[x,y+1:,z]:
                        inside_obj = curr_voxel_val
                        continue
                    #Leaving an objects interior
                    if curr_voxel_val > 0 and inside_obj > 0 and voxel_array[x, y+1, z] == 0.:
                        inside_obj = 0
                        continue
                    #Filling an objects interior
                    if inside_obj > 0: voxel_array[x, y, z] = inside_obj

    #For each object and every y-value, set all voxels inside the convex hull of the already occupied voxels to occupied:
    from skimage.morphology import convex_hull_image
    if single_objs:
        for obj in range(array_size[3]):
            for y in range(array_size[1]):
                obj_slice = (voxel_array[:, y, :, obj] > 0)
                convex_hull = convex_hull_image(obj_slice)

                voxel_array[:, y, :, obj] = voxel_array[:, y, :, obj] - (voxel_array[:, y, :, obj] * convex_hull.astype(int))
                voxel_array[:, y, :, obj] = voxel_array[:, y, :, obj] + ((obj+1) * convex_hull.astype(int))
    else:
        for obj_indx in range(1,4):
            for y in range(array_size[1]):
                obj_slice = (voxel_array[:, y, :] == obj_indx)
                convex_hull = convex_hull_image(obj_slice)

                voxel_array[:, y, :] = voxel_array[:, y, :] - (voxel_array[:, y, :] * convex_hull.astype(int))
                voxel_array[:, y, :] = voxel_array[:, y, :] + (obj_indx * convex_hull.astype(int))

    # Flip the scene around the x-axis:
    if single_objs:
        voxel_array_ = np.zeros([array_size[0], array_size[2], array_size[1], array_size[3]])
        for obj in range(array_size[3]):
            for z in range(array_size[1]):
                voxel_array_[:, :, z, obj] = voxel_array[:, z, ::-1, objs_permutation[obj]]
    else:
        voxel_array_ = np.zeros([array_size[0], array_size[2], array_size[1]])
        for z in range(array_size[1]):
            voxel_array_[:, :, z] = voxel_array[:, z, ::-1]
    voxel_array = voxel_array_

    #Set all occupied voxels to 1:
    voxel_array = (voxel_array > 0).astype(int)

    #print("Done")

    if gen_plots:
        if single_objs:
            for obj in range(array_size[3]):
                print(mean_vertices[1:, [0, 2, 1]][obj, :])

                fig = plt.figure(dpi=400)
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                voxels = (voxel_array[:, :, :, obj] != 0)
                ax.voxels(voxels, linewidth=0.1)
                plt.show()
        else:
            fig = plt.figure(dpi=400)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            voxels = (voxel_array != 0)
            ax.voxels(voxels, linewidth=0.1)
            plt.show()

        # for y in range(array_size[1]):
        #     plt.imshow(voxel_array[:, y, :])
        #     plt.show()

    #print(np.sum(voxel_array))

    perc_occupied = np.sum(voxel_array)/voxel_array.size
    perc_pts_in_bb = points_in_bb/all_pts

    if single_objs:
        mean_vertices = mean_vertices[1:, [0, 2, 1]]
        mean_vertices[:, 1] = -mean_vertices[:, 1]
        return voxel_array, mean_vertices
    return voxel_array


def pred_sdf_to_voxels(renderer, latent_vectors, bounding_box, resolution, single_objs=False, gen_plots=False):
    #param latent_vectors: [num_objects, size_shape+size_tex+size_ext]
    #param resolution: scalar number of voxels per unit length
    #param bounding_box: [min_x, max_x, min_y, max_y, min_z, max_z]; only integer sizes for now

    shape_latents = latent_vectors[:, :8]
    ext_latents = latent_vectors[:, -5:] # [scale, t_x, t_y, t_z, alpha_z]

    [min_x, max_x, min_y, max_y, min_z, max_z] = bounding_box
    array_size = [(max_x-min_x)*resolution, (max_y-min_y)*resolution, (max_z-min_z)*resolution, np.shape(latent_vectors)[0]]
    voxel_array_tmp = np.ones(array_size)
    voxel_size = 1/resolution

    world_coord_pts = np.zeros([array_size[0]*array_size[1]*array_size[2], 3])
    for x in range(array_size[0]):
        for y in range(array_size[1]):
            for z in range(array_size[2]):
                world_coord_pts[x*array_size[1]*array_size[2]+y*array_size[2]+z, :] = np.array([min_x+x*voxel_size, min_y+y*voxel_size, min_z+z*voxel_size])

    for obj in range(np.shape(latent_vectors)[0]):
        ext_latent = ext_latents[obj:obj+1, :].astype('float32')
        s_obj, t_obj, R_obj = renderer.get_obj_trans(ext_latent)

        t_obj = t_obj.numpy()

        obj_coord_pts = np.copy(np.transpose(world_coord_pts))
        if not single_objs:
            obj_coord_pts -= tf.tile(t_obj[0, 0, :, :], [1, array_size[0]*array_size[1]*array_size[2]]).numpy()
        obj_coord_pts = obj_coord_pts*(s_obj[0, 0, 0, 0]).numpy()
        obj_coord_pts = np.matmul(R_obj[0, ...], obj_coord_pts)
        #print(s_obj[0, 0, 0, 0])
        #print(t_obj[0, 0, :, 0])

        sdf_vals = renderer.get_sdf_values(tf.expand_dims(np.transpose(obj_coord_pts), axis=0), tf.expand_dims(shape_latents[obj, :], axis=0))

        # pos = 0
        # for x in range(array_size[0]):
        #     for y in range(array_size[1]):
        #         for z in range(array_size[2]):
        #             voxel_array_tmp[x, y, z, obj] = sdf_vals[0, pos, 0]
        #             pos += 1
        voxel_array_tmp[:, :, :, obj] = sdf_vals[0, :, 0].numpy().reshape([(max_x-min_x)*resolution, (max_y-min_y)*resolution, (max_z-min_z)*resolution])

        #print("done with obj {}".format(obj))

    # voxel_array = np.zeros([(max_x-min_x)*resolution, (max_y-min_y)*resolution, (max_z-min_z)*resolution])
    # for x in range(array_size[0]):
    #     for y in range(array_size[1]):
    #         for z in range(array_size[2]):
    #             for obj in range(array_size[3]):
    #                 if voxel_array_tmp[x, y, z, obj] <= 0: #If the point of the voxel is inside object obj
    #                     voxel_array[x, y, z] = 1.

    if single_objs:
        voxel_array = (voxel_array_tmp <= 0).astype(int)    #Set all voxels inside objects to 1, all others to 0
    else:
        voxel_array = np.min(voxel_array_tmp, axis=-1)  #For a voxel, if the min over the SDFs of all objects in the scene is <= 0, it is inside an object
        voxel_array = (voxel_array <= 0).astype(int)    #Set all voxels inside objects to 1, all others to 0


    #print("Done with all objects in the scene")
    frac_occupied = np.sum(voxel_array)/(array_size[0]*array_size[1]*array_size[2])

    if gen_plots:
        import matplotlib.pyplot as plt
        if single_objs:
            for obj in range(np.shape(voxel_array)[-1]):
                fig = plt.figure(dpi=400)
                ax = fig.add_subplot(111, projection='3d')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                voxels = (voxel_array[:, :, :, obj] != 0)
                ax.voxels(voxels, linewidth=0.1)
                plt.show()
        else:
            fig = plt.figure(dpi=400)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            voxels = (voxel_array != 0)
            ax.voxels(voxels, linewidth=0.1)
            plt.show()

        #for y in range(array_size[1]):
        #    plt.imshow(voxel_array[:, y, :])
        #    plt.show()

    return voxel_array

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

    cnfg_file = os.path.join(LOG_DIR, FLAGS.config + '.pkl')
    CONFIG = load_config(cnfg_file)
    if 'anti_aliasing' not in CONFIG['model']['mosnet']:
        CONFIG['model']['mosnet']['anti_aliasing'] = False

    model_file = os.path.join(LOG_DIR, model_base_name + '.py')
    model_module = load_module_from_log(model_base_name, model_file)
    MODEL = model_module.get_model(CONFIG, model_ext_name)

    # Create evaluation directory
    EVAL_DIR = os.path.join(LOG_DIR, FLAGS.eval_dir)
    if not os.path.exists(EVAL_DIR):
        os.mkdir(EVAL_DIR)

    # GT background
    bg_img, bg_depth = data_provider.load_bg_img(FLAGS.data_dir, CONFIG['data']['img_size'])
    MODEL.set_gt_bg(depth=bg_depth)

    # Open Log-file
    LOG_FILE = LogFile(os.path.join(LOG_DIR, 'log_eval.txt'))
    LOG_FILE.write(str(FLAGS))
    LOG_FILE.write('{:%d.%m_%H:%M}'.format(datetime.now()))

    # Load data
    DATA_VAL = data_provider.DataSetMultiObj(FLAGS.data_dir, FLAGS.split, CONFIG['data'])

    r_and_c = True

    if r_and_c:
        optimized_latents_21 = np.zeros([1000, 3, 21]) #The optimized latents with rotation as sin, cos; Used by eval_stats
        optimized_latents_20 = np.zeros([1000, 3, 20]) #The optimized latents with rotation as single scalar; Used for voxel eval


        CONFIG['training']['batch_size'] = 1

        if 'anti_aliasing' not in CONFIG['model']['mosnet']:
            CONFIG['model']['mosnet']['anti_aliasing'] = False

        CONFIG['r-and-c'] = {}                      # Holds the config parameters specific to the render-and-compare optimization
        for k, v in CONFIG['training'].items():
            CONFIG['r-and-c'][k] = v


        CONFIG['r-and-c']['post-processing'] = False        # Whether to delete superfluous object slots in post-processing
        CONFIG['r-and-c']['produce_dataset'] = False        # Whether to produce an optimized pseudo-ground-truth dataset
        # based on the training set to train a model in a supervised manner
        CONFIG['r-and-c']['loss_eval'] = False              # Evaluation;  Only if produce_dataset = False

        CONFIG['r-and-c']['batch_size'] = 1 # Optimize one image at a time; We want to minimize the loss_total which is defined over the entire batch
        CONFIG['r-and-c']['learning_rate_shape'] = 0.
        CONFIG['r-and-c']['learning_rate_tex'] = 0.001      # Learning rate for the texture
        CONFIG['r-and-c']['learning_rate_ext'] = 0.01       # Learning rate for the extrinsics
        CONFIG['r-and-c']['eps_final'] = 0.00000000000000001   # Convergence parameter for final output
        CONFIG['r-and-c']['eps'] = 0.0001                   # Convergence parameter for blurred images
        CONFIG['r-and-c']['num_examples'] = 200              # The number of examples in the validation set we want to render and compare for
        CONFIG['r-and-c']['optimize_rand_obj'] = False       # Whether to optimize a slot from random if the encoder output produces too large a loss
        CONFIG['r-and-c']['rand_seed'] = 42                 # Random seed
        CONFIG['r-and-c']['loss_thresh'] = 0.01             # Loss threshold above which a random vector is optimized instead of the encoder output
        CONFIG['r-and-c']['print_timing_info'] = False       # Whether to print how long the optimization steps took
        CONFIG['r-and-c']['smoothing_kernels_rand'] = [16., 4., 0.]     # The sizes of the smoothing kernels when optimizing from random initialization, in order of usage
        CONFIG['r-and-c']['smoothing_kernels_enc'] = [0.]               # The sizes of the smoothing kernels when optimizing encoder output
        CONFIG['r-and-c']['print_loss_vals'] = False
        CONFIG['r-and-c']['max_num_iterations'] = 50        # The maximum number of iterations with the current smoothing kernel
        CONFIG['r-and-c']['optimization_order'] = ['all']   # The order in which the shape, texture and extrinsics parts of the latents are optimized;
        # Can be any permutation of ['ext', 'tex', 'shape'] or ['all'] to optimize all at once


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

        params = {}
        weights = CONFIG['model'][model_base_name]['l-weights']
        for k, w in weights.items():
            params['w-' + k] = w
        params['w-z-reg'] = tf.constant(0.025)

        render_and_compare_opt = Render_and_compare_optimization(MODEL, CONFIG)

    main()
    LOG_FILE.write('--')

    exit(0)
