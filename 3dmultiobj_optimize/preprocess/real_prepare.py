import os
import argparse
import json
import pyexr
import random

import numpy as np
import pyrealsense2 as rs
from PIL import Image, ImageEnhance
from shutil import copyfile
from scipy.io import savemat, loadmat
from skimage.transform import resize

SIZE = 64
SIZE_ORG = (640, 480)

# 110956 (~6k), 113220 (~9k), 115157 (~4k)

def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--in_dir', default='/is/sg2/celich/Pictures/RealSense/20200709_v1/', help='Data dir')
    # parser.add_argument('--out_dir', default='/is/sg2/celich/Pictures/RealSense/realsense_07-09_v1/', help='Data dir')
    parser.add_argument('--in_dir', default='/is/rg/ev/scratch/celich/data/RealSense/new_20200709_110956/', help='Data dir')
    parser.add_argument('--out_dir', default='/is/rg/ev/scratch/celich/data/RealSense/new_20200709_110956_out/', help='Data dir')
    parser.add_argument('--start_idx', default=0, type=int, help='First index for naming files')
    FLAGS = parser.parse_args()
    return FLAGS


def viz(img, path=None):
    img_viz = np.asarray(img)
    # img_viz = np.clip(img_viz, 0., 750.)
    depth_scale = 0.001
    img_viz = np.clip(img_viz, 0., 750.*depth_scale)
    # img_viz = (img_viz-np.min(img_viz)) / (np.max(img_viz) - np.min(img_viz))
    img_viz = (img_viz - 270.*depth_scale) / (np.max(img_viz) - 270.*depth_scale)
    img_viz = np.clip(img_viz, 0., 1.)

    img_viz = 255. * (img_viz / np.max(img_viz))
    img_viz = img_viz.astype(np.uint8)
    img_viz = Image.fromarray(img_viz)
    if path is None:
        img_viz.show()
    else:
        img_viz.save(path)


def select_imgs(in_dir, out_dir):
    """
    :param dir:  path to folder containing image stream
    :return:  [list of image names]
    """

    if not os.path.exists(in_dir):
        print('- Input directory does not exist; {}'.format(in_dir))
        return []
    print('- Input directory: {}'.format(in_dir))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('- Make output directory; {}'.format(out_dir))
    else:
        print('- Output directory already exist; {}'.format(out_dir))

    sub_dirs = ['images_org', 'depth_org']
    for sub_dir in sub_dirs:
        if not os.path.exists(os.path.join(out_dir, sub_dir)):
            os.mkdir(os.path.join(out_dir, sub_dir))

    name_list = [[]]
    th = 4.
    th_max = 25.
    th2_mean = 75.
    th2_max = 50.
    id_next_image = 5

    in_dir_depth = in_dir  #.replace('v1', 'v1_raw')

    rgb_all_names = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f)) and '.png' in f and 'Color' in f]
    rgb_all_names.sort()
    depth_all_names = [f for f in os.listdir(in_dir_depth) if os.path.isfile(os.path.join(in_dir_depth, f)) and '.raw' in f and 'Depth' in f]
    depth_all_names.sort()
    # assert(len(rgb_all_names) == len(depth_all_names))
    rgb_all_names = rgb_all_names[:200]
    depth_all_names = depth_all_names[:200]

    rgb_all_imgs = [np.array(Image.open(os.path.join(in_dir, name))) for name in rgb_all_names]

    print('Loaded images:', len(rgb_all_names), len(depth_all_names))

    rgb_list_final = []
    depth_list_final = []
    rgb_list_tmp = []
    depth_list_tmp = []
    cur_id = FLAGS.start_idx
    for i in range(len(rgb_all_imgs)-id_next_image):
        print(i)
        rgb_name = rgb_all_names[i]
        depth_name = depth_all_names[i]
        rgb_id_tmp = int(rgb_name.split('.')[0].split('_')[-1])
        depth_id_tmp = int(depth_name.split('.')[0].split('_')[-1])
        # assert(depth_id_tmp+1 == rgb_id_tmp or depth_id_tmp == rgb_id_tmp)
        if not (depth_id_tmp+1 == rgb_id_tmp or depth_id_tmp == rgb_id_tmp):
            print('[WARNING] Id mismatch: depth ', depth_id_tmp, ', rgb ', rgb_id_tmp)

        cur_img = rgb_all_imgs[i]
        next_img = rgb_all_imgs[i+id_next_image]

        diff_mean = np.mean(np.square(cur_img.astype("float") - next_img.astype("float")))
        diff_max = np.max(np.abs(cur_img.astype("float") - next_img.astype("float")))
        if diff_mean > th or diff_max > th_max:
            continue

        if len(rgb_list_tmp) > 0:
            prev_img = rgb_list_tmp[-1]   # last image in last set
            diff_mean_2 = np.mean(np.square(cur_img.astype("float") - prev_img.astype("float")))
            diff_max_2 = np.max(np.abs(cur_img.astype("float") - prev_img.astype("float")))

            # new scene
            if diff_mean_2 > th2_mean or diff_max_2 > th2_max:

                name_rgb = 'rs_{:04d}_0.png'.format(cur_id)
                cur_id += 1

                rgb, depth = merge_imgs(rgb_list_tmp, depth_list_tmp)

                rgb.save(os.path.join(out_dir, 'images_org', name_rgb))
                viz(depth, os.path.join(out_dir, 'depth_org', name_rgb))

                rgb_list_final.append(rgb)
                depth_list_final.append(depth)

                rgb_list_tmp = []
                depth_list_tmp = []
                name_list.append([])

        # load depth image
        raw_depth = open(os.path.join(in_dir_depth, depth_all_names[i]), 'rb').read()
        depth = Image.frombytes('I;16L', SIZE_ORG, raw_depth, 'raw')

        rgb_list_tmp.append(cur_img)
        depth_list_tmp.append(depth)
        name_list[-1].append(rgb_all_names[i])

    # output for Matlab
    imgs = np.stack(rgb_list_final, axis=-1)
    depths = np.stack(depth_list_final, axis=-1)
    mdic = {"images": imgs, "rawDepths": depths}
    savemat(os.path.join(out_dir, 'depth_org', 'init.mat'), mdic)

    return rgb_list_final, depth_list_final, name_list


def select_imgs_2(in_dir, out_dir):
    """
    :param dir:  path to folder containing image stream
    :return:  [list of image names]
    """

    if not os.path.exists(in_dir):
        print('- Input directory does not exist; {}'.format(in_dir))
        return []
    print('- Input directory: {}'.format(in_dir))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('- Make output directory; {}'.format(out_dir))
    else:
        print('- Output directory already exist; {}'.format(out_dir))

    sub_dirs = ['images_org', 'depth_org']
    for sub_dir in sub_dirs:
        if not os.path.exists(os.path.join(out_dir, sub_dir)):
            os.mkdir(os.path.join(out_dir, sub_dir))

    name_list = [[]]
    th = 4.
    th_max = 25.
    th2_mean = 75.
    th2_max = 50.
    id_next_image = 5

    print('- Load Depth data')
    depth_all_names = [f for f in os.listdir(in_dir)
                       if os.path.isfile(os.path.join(in_dir, f)) and '.npy' in f and 'depth_' in f]
    # depth_all_images = [np.load(os.path.join(in_dir, name)) for name in depth_all_names]
    # depth_all_images = np.concatenate(depth_all_images, axis=0)

    def load_depth(name):
        depth_start_idx = int(((name.split('.')[0]).split('_')[1])[1:])
        depth_end_idx = int(((name.split('.')[0]).split('_')[2])[1:])
        depth_part_images = np.load(os.path.join(in_dir, name))

        return depth_part_images, depth_start_idx, depth_end_idx

    depth_np_id = 0
    depth_offset_idx = 0
    depth_part_images, depth_start_idx, depth_end_idx = load_depth(depth_all_names[depth_np_id])
    # print(depth_all_names[1])
    # depth_part_images, depth_start_idx, depth_end_idx = load_depth(depth_all_names[1])
    # viz(depth_part_images[0])

    print('- Load RGB data')
    rgb_all_names = [f for f in os.listdir(in_dir)
                     if os.path.isfile(os.path.join(in_dir, f)) and '.png' in f and 'col_' in f]
    rgb_all_names.sort()
    # rgb_all_names = rgb_all_names[:2000]

    rgb_all_imgs = [np.array(Image.open(os.path.join(in_dir, name))) for name in rgb_all_names]

    print('Loaded data:', len(rgb_all_imgs), depth_part_images.shape[0])

    rgb_list_final = []
    depth_list_final = []
    rgb_list_tmp = []
    depth_list_tmp = []
    cur_id = FLAGS.start_idx
    for i in range(len(rgb_all_imgs)-id_next_image):
        if i%100 == 0:
            print(i)

        if i-depth_offset_idx >= depth_part_images.shape[0]:
            depth_np_id += 1
            depth_offset_idx += depth_part_images.shape[0]
            print(depth_all_names[depth_np_id], depth_np_id, depth_offset_idx)
            depth_part_images, depth_start_idx, depth_end_idx = load_depth(depth_all_names[depth_np_id])
            # viz(depth_part_images[0])

        cur_img = rgb_all_imgs[i]
        next_img = rgb_all_imgs[i+id_next_image]

        diff_mean = np.mean(np.square(cur_img.astype("float") - next_img.astype("float")))
        diff_max = np.max(np.abs(cur_img.astype("float") - next_img.astype("float")))
        if diff_mean > th or diff_max > th_max:
            continue

        if len(rgb_list_tmp) > 0:
            prev_img = rgb_list_tmp[-1]   # last image in last set
            diff_mean_2 = np.mean(np.square(cur_img.astype("float") - prev_img.astype("float")))
            diff_max_2 = np.max(np.abs(cur_img.astype("float") - prev_img.astype("float")))

            # new scene
            if diff_mean_2 > th2_mean or diff_max_2 > th2_max:

                name_rgb = 'rs_{:04d}_0.png'.format(cur_id)
                cur_id += 1

                rgb, depth = merge_imgs(rgb_list_tmp, depth_list_tmp)

                rgb.save(os.path.join(out_dir, 'images_org', name_rgb))
                viz(depth, os.path.join(out_dir, 'depth_org', name_rgb))

                rgb_list_final.append(rgb)
                depth_list_final.append(depth)

                rgb_list_tmp = []
                depth_list_tmp = []
                name_list.append([])

        # load depth image
        # raw_depth = open(os.path.join(in_dir_depth, depth_all_names[i]), 'rb').read()
        # depth = Image.frombytes('I;16L', SIZE_ORG, raw_depth, 'raw')
        depth = depth_part_images[i - depth_offset_idx]

        rgb_list_tmp.append(cur_img)
        depth_list_tmp.append(depth)
        name_list[-1].append(rgb_all_names[i])

    # last scene
    name_rgb = 'rs_{:04d}_0.png'.format(cur_id)
    rgb, depth = merge_imgs(rgb_list_tmp, depth_list_tmp)
    rgb.save(os.path.join(out_dir, 'images_org', name_rgb))
    viz(depth, os.path.join(out_dir, 'depth_org', name_rgb))
    rgb_list_final.append(rgb)
    depth_list_final.append(depth)

    # output for Matlab
    imgs = np.stack(rgb_list_final, axis=-1)
    depths = np.stack(depth_list_final, axis=-1)
    mdic = {"images": imgs, "rawDepths": depths}
    savemat(os.path.join(out_dir, 'depth_org', 'init.mat'), mdic)

    return rgb_list_final, depth_list_final, name_list


def merge_imgs(rgb_list_tmp, depth_list_tmp):

    # id = int(len(rgb_list_tmp)/2)

    # RGB Image
    img = np.mean(np.stack(rgb_list_tmp, axis=0), axis=0)
    img = Image.fromarray(np.uint8(img))

    # Depth
    d_clip = 750.
    depths = np.stack(depth_list_tmp, axis=0)  # (N_img, H, W) for scene i
    msks = np.where(depths > 0, 1, 0)
    msk_sum = np.sum(msks, axis=0)
    depth = np.where(msk_sum != 0, np.sum(depths, axis=0) / msk_sum, 0)
    depth = np.where(depth <= d_clip, depth, 0)
    depth = np.clip(depth, 0., d_clip)

    return img, depth


def complete_data(out_dir, name_list):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('Make output directory; {}'.format(out_dir))
    else:
        print('Output directory already exist; {}'.format(out_dir))

    sub_dirs = ['instances', 'scenes']
    for sub_dir in sub_dirs:
        if not os.path.exists(os.path.join(out_dir, sub_dir)):
            os.mkdir(os.path.join(out_dir, sub_dir))

    scenes_names = {}
    for i in range(len(name_list)):

        name_rgb = 'rs_{:04d}_0.png'.format(i)

        scenes_names[name_rgb.replace('.png', '')] = name_list[i]

        # Instance (TMP)
        tmp_png = Image.fromarray(np.uint8(np.zeros((64, 64, 3))))
        tmp_png.save(os.path.join(out_dir, 'instances',  name_rgb))

        single_obj_dict = {
                  "size": 1,
                  "rotation": [0.],
                  "pixel_coords": [[32, 32, 4.]],
                  "3d_coords": [[0., 0., 1.0]]
                }
        scene_info = {
            "objects": [single_obj_dict for _ in range(3)]
        }
        name_json = name_rgb.replace('_0.png', '.json')
        with open(os.path.join(out_dir, 'scenes', name_json), 'w') as f:
            json.dump(scene_info, f, indent=2)

    with open(os.path.join(out_dir, 'scene_names.json'), 'w') as f:
        json.dump(scenes_names, f, indent=2)


def get_matlab_data(out_dir, depth_ext):

    mdic = loadmat(os.path.join(out_dir, 'depth_org', 'filled' + depth_ext + '.mat'))
    imgs = mdic['rgb']
    depths = mdic['depth']

    sub_dirs = ['images', 'depth_prev'+depth_ext, 'depth_viz'+depth_ext]
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(out_dir, sub_dir)
        if not os.path.exists(sub_dir_path):
            os.mkdir(sub_dir_path)
            print('Make output directory; {}'.format(sub_dir_path))
        else:
            print('Output directory already exist; {}'.format(sub_dir_path))

    for i in range(imgs.shape[3]):

        name = 'rs_{:04d}_0'.format(i)

        # RGB Image
        img = imgs[:, :, :, i]
        img = Image.fromarray(np.uint8(img))
        img.save(os.path.join(out_dir, 'images', name+'.png'))

        # Depth
        depth = depths[:, :, i]
        pyexr.write(os.path.join(out_dir, 'depth_prev'+depth_ext, name+'.exr'), depth)

        viz(depth, os.path.join(out_dir, 'depth_viz'+depth_ext, name+'.png'))


def preprocess_imgs(dir, depth_ext, test_ratio=1.):

    img_in_dir = os.path.join(dir, 'images')
    img_out_dir = os.path.join(dir, 'images_bright')

    depth_in_dir = os.path.join(dir, 'depth_prev'+depth_ext)
    depth_out_dir = os.path.join(dir, 'depth')

    for out_dir in [img_out_dir, depth_out_dir]:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            print('Make output directory; {}'.format(out_dir))
        else:
            print('Output directory already exist; {}'.format(out_dir))

    # Images
    # left = 40
    # top = 0
    # right = 420
    # bottom = 380
    left = 50
    top = 0
    right = 410
    bottom = 360
    assert(left-right == top-bottom)

    img_name_list = [f for f in os.listdir(img_in_dir) if os.path.isfile(os.path.join(img_in_dir, f)) and '.png' in f]

    for name in img_name_list:

        img = Image.open(os.path.join(img_in_dir, name))

        img = img.crop((left, top, right, bottom))  # current original size was (380, 460)
        # img.save(os.path.join(out_dir, name))

        # img = img.resize((SIZE, SIZE), resample=Image.BILINEAR)

        enhancer_contrast = ImageEnhance.Contrast(img)
        img = enhancer_contrast.enhance(0.75)
        #
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.3)

        img.save(os.path.join(img_out_dir, name))

    # Train/ val split
    img_list = [f for f in os.listdir(img_out_dir) if os.path.isfile(os.path.join(img_out_dir, f)) and '.png' in f]
    random.shuffle(img_list)
    n_imgs = len(img_list)
    n_test = int(test_ratio * n_imgs)
    if n_test == 0 or n_test == n_imgs:
        train_img_list_out = img_list
        test_img_list_out = img_list
    else:
        train_img_list_out = img_list[:-n_test]
        test_img_list_out = img_list[-n_test:]

    # Write txt files with img paths for train/ val split
    with open(os.path.join(img_out_dir, 'train_images.txt'), 'w') as filehandle:
        filehandle.writelines("%s\n" % img for img in train_img_list_out)
    with open(os.path.join(img_out_dir, 'val_images.txt'), 'w') as filehandle:
        filehandle.writelines("%s\n" % img for img in test_img_list_out)

    # Depth
    depth_name_list = [f for f in os.listdir(depth_in_dir) if os.path.isfile(os.path.join(depth_in_dir, f)) and '.exr' in f]

    for name in depth_name_list:

        file = pyexr.open(os.path.join(depth_in_dir, name))
        depth = file.get()

        depth = depth[top:bottom, left:right]

        # # resize
        # d_max = np.max(depth)
        # d_min = np.min(depth)
        #
        # depth = 255. * (depth - d_min)/(d_max - d_min)
        # depth = np.tile(depth.astype(np.uint8), (1, 1, 3))
        # depth = Image.fromarray(depth)
        #
        # depth = depth.resize((SIZE, SIZE), resample=Image.BILINEAR)
        #
        # depth = np.float32((1. / 255) * np.array(depth))[:, :, 0:1]
        # # depth = depth * (d_max - d_min) + d_min
        # # depth = depth * (14. - 4.) + 4.
        # depth = depth * (12. - 2.) + 2.

        pyexr.write(os.path.join(depth_out_dir, name), depth)

    print('{} images processed.'.format(len(img_name_list)))


def combine_folders(out_dir, dir_ext_list):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print('- Make output directory; {}'.format(out_dir))
    else:
        print('- Output directory already exist; {}'.format(out_dir))

    sub_dirs = ['images', 'depth_prev_col', 'depth_viz_col', 'instances', 'scenes']
    for sub_dir in sub_dirs:
        if not os.path.exists(os.path.join(out_dir, sub_dir)):
            os.mkdir(os.path.join(out_dir, sub_dir))

    data_ext = {'images': '_0.png',
                'depth_prev_col': '_0.exr',
                'depth_viz_col': '_0.png',
                'instances': '_0.png',
                'scenes': '.json'}

    cur_id = 0
    for dir_ext in dir_ext_list:
        in_dir = '_'.join([out_dir, dir_ext])
        if not os.path.exists(in_dir):
            print('- Input directory does not exist; {}'.format(in_dir))
            exit(1)
        print('- Input directory: {}'.format(in_dir))

        rgb_all_names = [f[:-6] for f in os.listdir(os.path.join(in_dir, 'images')) if
                         os.path.isfile(os.path.join(in_dir, 'images', f)) and '.png' in f]

        for name in rgb_all_names:

            name_new = 'rs_{:04d}'.format(cur_id)
            cur_id += 1

            for sub_dir in sub_dirs:
                src = os.path.join(in_dir, sub_dir, name+data_ext[sub_dir])
                dst = os.path.join(out_dir, sub_dir, name_new+data_ext[sub_dir])
                copyfile(src, dst)


if __name__ == "__main__":

    FLAGS = parse_arguments()

    # print('1. Select Images')
    # rgb_list, depth_list, name_list = select_imgs(FLAGS.in_dir, FLAGS.out_dir)
    # for i, n in enumerate(name_list):
    #     print(i, len(n), n)
    #
    # print('2. Complete missing data')
    # complete_data(FLAGS.out_dir, name_list)

    print('1.b Select Images')
    rgb_list, depth_list, name_list = select_imgs_2(FLAGS.in_dir, FLAGS.out_dir)

    print('2. Complete missing data')
    complete_data(FLAGS.out_dir, name_list)

    depth_ext = '_col'
    # depth_ext = '_cbf'

    # print('3. Get Matlab Data')
    # get_matlab_data(FLAGS.out_dir, depth_ext)
    # #
    # print('4. Basic Pre-processing')
    # preprocess_imgs(FLAGS.out_dir, depth_ext)

    # print('5. Combine Image Folders')
    # # global_out_dir = '_'.join(FLAGS.out_dir.split('_')[:-1])
    # # dir_ext_list = ['v1', 'v2', 'v3']
    # global_out_dir = '_'.join(FLAGS.out_dir.split('_')[:-2])
    # dir_ext_list = ['110956_out', '113220_out', '115157_out']
    # combine_folders(global_out_dir, dir_ext_list)
    # preprocess_imgs(global_out_dir, depth_ext, test_ratio=0.2)

    exit(0)
