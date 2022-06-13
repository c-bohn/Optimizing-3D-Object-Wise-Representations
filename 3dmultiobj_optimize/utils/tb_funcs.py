import sys
from utils.tf_funcs import *


# ---- Handle individual summary entries

def loss_to_summary(name, value, epoch):
    """
    summary writer:     <tf object
    name:               <string>
    value:              <float>
    epoch:              <int>
    """
    tf.summary.scalar(name, value, step=epoch)


def stats_to_summary(name, value, epoch, hist=0):
    tf.summary.scalar(name, tf.reduce_mean(value), step=epoch)
    tf.summary.scalar(name + '_std', tf.math.reduce_std(value), step=epoch)
    # tf.summary.histogram(name, value, step=epoch)


def img_to_summary(name, imgs, epoch):
    if len(imgs.shape) == 6:  # multiple objects
        _, s, n, _, _, _ = imgs.shape
        imgs = tf.concat([tf.concat([imgs[:, j, i] for j in range(s)], axis=1) for i in range(n)], axis=2)
    if len(imgs.shape) == 5:  # multiple images
        bs, n, h, w, d = imgs.shape
        imgs = tf.transpose(a=imgs, perm=[0, 2, 1, 3, 4])
        imgs = tf.reshape(imgs, (-1, h, n * w, d))
    if imgs.dtype != tf.uint8:
        imgs = tf.cast(255 * imgs, tf.uint8)
    tf.summary.image(name, imgs, step=epoch, max_outputs=1)


def depth_to_summary(name, depth, epoch):
    if tf.shape(depth)[-1] != 1:
        depth = tf.expand_dims(depth, -1)
    depth_min = tf.math.reduce_min(input_tensor=depth)
    depth_max = tf.math.reduce_max(input_tensor=depth)
    depth = (depth - depth_min) / (depth_max - depth_min)

    img_to_summary(name, depth, epoch)


# ---- Manage entire model output

def summarize_all_losses(loss_dict, epoch, mode, log_file=None):
    for k, v in loss_dict.items():
        loss_flag = 'loss' in k or ('l_' == k[:2])
        if loss_flag:
            loss_to_summary('loss/' + mode + '/' + k, v, epoch)
        elif 'ctrl' in k:
            loss_to_summary('loss-ctrl/' + k.replace('-ctrl', ''), v, epoch)

        if log_file is not None and loss_flag:
            log_file.write(k + ': \t' + str(v))
        elif loss_flag:
            print(k + ': ' + str(v))


def summarize_all_simple(dict, name, epoch):
    for k, v in dict.items():
        tf.summary.scalar(name+'/' + k, tf.reduce_mean(v), step=epoch)


def summarize_all_latents(latents_dict, epoch):
    for k, v in latents_dict.items():
        if 'z_' == k[:2]:
            if ('_mean' in k or '_var' in k) and 'gt' not in k and tf.reduce_sum(v) != 0.:
                stats_to_summary('repr/' + k, v, epoch)


def summarize_all_images(data_dict, epoch, types=[]):

    for t in types:
        name = t
        if 'obj' in t:
            name = 'v_'+t

        if t not in data_dict.keys():
            continue

        if 'rgb' in t:
            if name == 'rgb_in':
                name = '0_input'
            img_to_summary(name, data_dict[t], epoch)

        elif 'normal' in t:
            img_to_summary(name, data_dict[t], epoch)

        elif 'diff' in t:
            diff_img = slice_coloring(data_dict[t])
            img_to_summary(t, diff_img, epoch)

        elif 'depth' in t:
            depth_to_summary(name, data_dict[t], epoch)

        elif 'msk' in t:
            img_to_summary(name, data_dict[t], epoch)

        # - Object Slices
        elif 'slice_gt' in t:
            img_to_summary(t, data_dict['slice_gt'], epoch)
        elif 'slice_pred' in t:
            shape_org = data_dict['slice_pred'].shape   # (BS, 3, W, H, 1)
            assert len(shape_org) == 5, '[Summary] Unknown shape for \'slice_pred\''
            pred_slice = tf.reshape(data_dict['slice_pred'], (-1, shape_org[2], shape_org[3], shape_org[4]))
            pred_slice = slice_coloring(pred_slice)
            # pred_slice = tf.image.flip_left_right(tf.image.flip_up_down(pred_slice))
            pred_slice = tf.image.flip_up_down(pred_slice)
            pred_slice = tf.reshape(pred_slice, (-1, shape_org[1], shape_org[2], shape_org[3], 3))
            img_to_summary(t, pred_slice, epoch)


def summarize_all(summary_writer, epoch, loss_dict=None, data_dict=None, eval_dict=None, params=None, imgs_types=[],
                  log_file=None, mode='train'):
    with summary_writer.as_default():
        if loss_dict is not None:
            summarize_all_losses(loss_dict, epoch, mode, log_file)
        if data_dict is not None:
            summarize_all_latents(data_dict, epoch)
            summarize_all_images(data_dict, epoch, imgs_types)
        if eval_dict is not None:
            summarize_all_simple(eval_dict, 'val', epoch)
        if params is not None:
            summarize_all_simple(params, 'params', epoch)
