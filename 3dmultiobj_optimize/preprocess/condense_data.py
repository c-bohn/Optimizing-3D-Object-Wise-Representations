import os
import argparse
#import pyexr
import cv2
import json
import numpy as np

from PIL import Image
from termcolor import colored
from tqdm import *
from math import pi

# to delete
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--type', default='train', help='from [train, val, (test?)]')
    parser.add_argument('--n_obj', type=int, default=3)
    parser.add_argument('--n_img', type=int, default=-1)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--depth_max', type=float, default=12.)
    FLAGS = parser.parse_args()
    return FLAGS


def exr2numpy(path, maxvalue, normalize=False, dist_to_depth=True):
    """ converts 1-channel exr-data to 2D numpy arrays """

    if not os.path.isfile(path):
        print('path not found..', path)
        exit(1)

    #file = pyexr.open(path)
    #img = file.get()
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    img = np.expand_dims(img, axis=-1)
    

    # Project depth values onto optical axis:
    #
    #           ----------
    #           |       /
    #        Z  |      /
    #           |     /  D
    #           | x  /
    #      ------------------ image plane
    #           |  /
    #        f  | /  d
    #           |/
    #           O
    #
    # D : Depth values returned by the cycles z-pass
    # f : Focal length
    # x : Pixel position on the image plane
    #
    # The sought depth value Z can be computed as follows:
    #
    # d = sqrt(x^2 + y^2 + f^2)
    # Z = f * D / d

    if dist_to_depth:
        # Sensor dimensions and pixel size
        camera_sensor_width = 32.
        camera_lens = 35.

        depth_values = img
        width, height, _ = depth_values.shape

        sensor_width = camera_sensor_width / 1000.0
        sensor_height = sensor_width * (height / width)
        px_size = sensor_width / width

        # Create grid with real world positions for each pixel
        img_coord = (np.indices(depth_values.shape).astype(np.float32) + 0.5) * px_size
        img_coord[0] -= sensor_height / 2.0
        img_coord[1] -= sensor_width / 2.0

        # Extend the 2D pixel position grid in 3rd dimension. The camera center is in the origin.
        # All pixels lie on the plane at z = focal length.
        f = camera_lens / 1000.0
        img_coord = np.concatenate((img_coord, np.ones(depth_values.shape)[np.newaxis, ...] * f))

        # Calculate transformed depth values
        depth = f * depth_values / np.linalg.norm(img_coord, axis=0)
    else:
        depth = img

    # old pre-processing
    # normalize and clip depth
    data = np.array(depth)[:, :, 0:1]
    data[data > maxvalue] = maxvalue

    if normalize:
        data /= np.max(data)

    img = data.astype(np.float32)

    return img


def prepare_bg_data(data_dir):

    if not os.path.exists(data_dir):
        print(colored('Input directory does not exist - {}'.format(data_dir), 'red'))
        exit(0)

    # Background Data
    # RGB
    rgb_path = os.path.join(data_dir, 'img-'+str(FLAGS.img_size)+'.png')
    rgb = Image.open(rgb_path)

    # Depth
    depth_path = os.path.join(data_dir, 'depth-'+str(FLAGS.img_size)+'.exr')
    dtd = not ('real' in depth_path)
    depth = exr2numpy(depth_path, maxvalue=FLAGS.depth_max, dist_to_depth=dtd)

    np.save(os.path.join(data_dir, 'rgb-'+str(FLAGS.img_size)), rgb)
    np.save(os.path.join(data_dir, 'depth-'+str(FLAGS.img_size)), depth)


if __name__ == "__main__":

    FLAGS = parse_arguments()
    data_dir = FLAGS.data_dir

    if not os.path.exists(data_dir):
        print(colored('(ERROR) Input directory does not exist - {}'.format(data_dir), 'red'))
        exit(0)

    out_dir = os.path.join(data_dir, 'input_' + FLAGS.type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('Created output directory {}'.format(out_dir))
    else:
        print('Output directory exists; potentially overwriting contents ({}).'.format(out_dir))

    # ----------------------------------------------------------------------------------------
    # -- 2D Data

    rgb_dir = os.path.join(data_dir, 'images_bright')

    # Get list of all scenes
    with open(os.path.join(rgb_dir, FLAGS.type + '_images.txt'), 'r') as f:
        datalist = f.read().splitlines()
    # datalist = datalist[:10]

    scene_names = []
    rgb_imgs = []
    depth_imgs = []
    msk_imgs = []

    for s in tqdm(datalist):
        name = s[:-6]

        # RGB
        rgb_path = os.path.join(rgb_dir, s)
        rgb = Image.open(rgb_path)

        # Depth
        depth_path = rgb_path.replace('images_bright', 'depth').replace('.png', '.exr')
        dtd = not('real' in depth_path)
        depth = exr2numpy(depth_path, maxvalue=FLAGS.depth_max, dist_to_depth=dtd)

        # Instance Masks
        instances_path = rgb_path.replace('images_bright', 'instances')
        instance = np.asarray(Image.open(instances_path))
        un_inst = np.unique(np.reshape(instance, (-1, 4)), axis=0)
        msk_list = []
        for i in range(FLAGS.n_obj + 1):
            if i < len(un_inst):
                inst_col = un_inst[i]
                if np.all(inst_col == [64, 64, 64, 255]):  # background
                    continue
                msk = np.ones(instance.shape[:2])
                for c in range(3):
                    msk *= instance[:, :, c] == inst_col[c]
            else:
                msk = np.zeros(instance.shape[:2])
            msk_list.append(msk)
        msk = np.asarray(msk_list)

        if name not in scene_names:
            scene_names.append(name)
            rgb_imgs.append([])
            depth_imgs.append([])
            msk_imgs.append([])
        if FLAGS.n_img != -1 and len(rgb_imgs[-1]) == FLAGS.n_img:
            continue
        rgb_imgs[-1].append(np.array(rgb)[:, :, :3])      # ignore alpha channel
        depth_imgs[-1].append(np.array(depth)[:, :, 0])     # all color channel have same depth value
        msk_imgs[-1].append(msk)
    rgb_imgs = np.array(rgb_imgs)
    depth_imgs = np.array(depth_imgs)
    msk_imgs = np.transpose(np.asarray(msk_imgs, dtype=bool), (0, 2, 1, 3, 4))

    # Save data as numpy array
    np.save(os.path.join(out_dir, 'rgb'), rgb_imgs)
    np.save(os.path.join(out_dir, 'depth'), depth_imgs)
    np.save(os.path.join(out_dir, 'mask'), msk_imgs)

    with open(os.path.join(out_dir, 'scene_names.txt'), 'w') as filehandle:
        for scene in scene_names:
            filehandle.write('%s\n' % scene)

    print('(DONE) Prepared 2D data for {} scenes.'.format(rgb_imgs.shape[0]))

    # ----------------------------------------------------------------------------------------
    # -- 3D Extrinsic Data

    scenes_dir = os.path.join(data_dir, 'scenes')

    obj_extrs = []
    try:
        for s in scene_names:
            with open(os.path.join(scenes_dir, s + '.json')) as json_file:
                data = json.load(json_file)
                obj_extrs.append([])
                for i in range(FLAGS.n_obj):
                    if i >= len(data['objects']):
                        eps = 0.00001
                        obj_extrs[-1].append(eps * np.ones_like(obj_extrs[-1][-1]))
                        continue
                    obj = data['objects'][i]
                    size = 1.25 * obj['size']
                    pos = obj['3d_coords']
                    rot = [-r / pi for r in obj['rotation']]
                    z_extr = [[[size] + pos[i] + [rot[i]]] for i in range(len(pos))]
                    z_extr = np.asarray(z_extr)
                    if FLAGS.n_img > -1:
                        z_extr = z_extr[0:FLAGS.n_img]
                    obj_extrs[-1].append(np.squeeze(z_extr))
        obj_extrs = np.asarray(obj_extrs)  # (n_scenes, n_obj, n_views, 5)
        if len(obj_extrs.shape) == 3:
            obj_extrs = np.expand_dims(obj_extrs, axis=2)

        np.save(os.path.join(out_dir, 'obj_extrs'), obj_extrs)
        print('(DONE) Prepared 3D extrinsic data for {} scenes.'.format(obj_extrs.shape[0]))
    except FileNotFoundError:
        print('(WARNING) No object extrinsic data found.')

    # ----------------------------------------------------------------------------------------
    # -- 3D Shapes
    # Object names and slice views (for visualization)

    obj_dir = os.path.join(data_dir, 'objects')

    obj_names = [[] for _ in range(len(scene_names))]
    obj_slice_imgs = [[] for _ in range(len(scene_names))]
    try:
        with open(os.path.join(obj_dir, 'slices', 'slices.txt'), 'r', encoding='utf-8-sig') as f:
            slice_list = f.read().splitlines()
        for s in slice_list:
            obj_name = s[:-6]
            scene_name = '_'.join(obj_name.split('_')[:3])
            if scene_name not in scene_names:
                continue
            scene_id = scene_names.index(scene_name)
            img = Image.open(os.path.join(obj_dir, 'slices', s))
            obj_slice_imgs[scene_id].append(np.array(img))
            if 'x.png' in s:
                obj_names[scene_id].append(obj_name)

        # Save data
        obj_slice_imgs = np.asarray(obj_slice_imgs)
        np.save(os.path.join(out_dir, 'obj_slices'), obj_slice_imgs)
        with open(os.path.join(out_dir, 'obj_names.txt'), 'w') as filehandle:
            for scene_objs in obj_names:
                filehandle.write('%s\n' % scene_objs)

        print('(DONE) Prepared 3D shape data (slices viz) for {} scenes.'.format(obj_slice_imgs.shape[0]))
    except FileNotFoundError:
        print('(WARNING) No 3d object shape data (names+slices) found.')

    obj_sdf_smpls = [0. for _ in range(len(scene_names))]
    sdf_sample_dir = os.path.join(obj_dir, 'sampled')
    try:
        # with open(os.path.join(obj_dir, 'objects', 'sampled', 'samples.txt'), 'r', encoding='utf-8-sig') as f:
        with open(os.path.join(sdf_sample_dir, 'samples.txt'), 'r') as f:
            sample_list = f.read().splitlines()
        for s in sample_list:
            obj_name = s.split(',')[1]
            scene_name = '_'.join(obj_name.split('_')[:3])
            if scene_name not in scene_names:
                continue
            scene_id = scene_names.index(scene_name)
            sdf_path = os.path.join(sdf_sample_dir, obj_name)
            obj_sdf = np.float32(np.load(sdf_path)['arr_0'])
            n_total_sampled_pnts = obj_sdf.shape[0]
            obj_sdf_smpls[scene_id] = obj_sdf

            # # ----------
            # n_on = int(n_total_sampled_pnts / 5)
            # n_off = n_total_sampled_pnts - n_on
            #
            # obj_sdf = obj_sdf[:, 3:]
            # obj_sdf_on_surface = obj_sdf[:int(n_total_sampled_pnts / 5)]
            # obj_sdf_off_surface = obj_sdf[int(n_total_sampled_pnts / 5):]
            #
            # obj_sdf_on_surface = np.abs(obj_sdf_on_surface)
            # obj_sdf_off_surface = np.abs(obj_sdf_off_surface)
            # print('on surface', np.min(obj_sdf_on_surface), np.mean(obj_sdf_on_surface), np.max(obj_sdf_on_surface))
            # print('off surface', np.min(obj_sdf_off_surface), np.mean(obj_sdf_off_surface), np.max(obj_sdf_off_surface))
            # #-----------

        # Save data
        obj_sdf_smpls = np.asarray(obj_sdf_smpls)
        np.save(os.path.join(out_dir, 'obj_sdf_smpls'), obj_sdf_smpls)

        print('(DONE) Prepared 3D shape data (sdf samples) for {} scenes.'.format(obj_sdf_smpls.shape[0]))
    except FileNotFoundError:
        print('(WARNING) No 3d object shape data (sdf samples) found.')

    # ----------------------------------------------------------------------------------------
    # -- Background Data
    bg_dir = os.path.join(os.path.dirname(data_dir), 'bg_exrdepth')
    if os.path.exists(bg_dir):
        prepare_bg_data(bg_dir)
        print('(DONE) Background data prepared.')
    else:
        print('(WARNING) Background directory does not exist - {}'.format(bg_dir))
        exit(0)

    # ----------------------------------------------------------------------------------------

    exit(0)
