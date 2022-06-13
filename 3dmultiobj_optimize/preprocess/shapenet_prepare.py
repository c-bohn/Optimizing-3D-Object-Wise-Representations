import os
import argparse
import datetime
from multiprocessing import Pool
from itertools import product
from shutil import copyfile

import numpy as np
import math
from PIL import Image, ImageEnhance
import pymesh
import trimesh
import random

NUM_THREADS = 6


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Data dir')
    parser.add_argument('--test_ratio', default=0.1, type=float, help='Test ratio')
    parser.add_argument('--scale_size', default=64, type=int, help='Size of output image')
    FLAGS = parser.parse_args()
    return FLAGS


def rigid_transform(mesh, offset, rotation_axis, rotation_angle, output_path):

    offset = np.array(offset)
    axis = np.array(rotation_axis)
    angle = math.radians(rotation_angle)
    rot = pymesh.Quaternion.fromAxisAngle(axis, angle)
    rot = rot.to_matrix()

    vertices = mesh.vertices
    bbox = mesh.bbox
    centroid = 0.5 * (bbox[0] + bbox[1])
    vertices = np.dot(rot, (vertices - centroid).T).T + centroid + offset

    pymesh.save_mesh_raw(output_path, vertices, mesh.faces, mesh.voxels)


def process_obj(args):
    name, in_dir, out_dir = args

    in_path = os.path.join(in_dir, name)
    out_path = os.path.join(out_dir, name).replace('.obj', '.off')

    if os.path.exists(out_path):
        return out_path

    mesh = pymesh.load_mesh(in_path)

    rigid_transform(mesh, 0., [1, 0, 0], 90, out_path)
    # pymesh.save_mesh(out_path, mesh)

    # Remove '# Generated with PyMesh' line from file
    with open(out_path, "r") as f:
        lines = f.readlines()
    with open(out_path, "w") as f:
        for line in lines:
            if 'PyMesh' in line:
                continue
            f.write(line)

    mesh = trimesh.load(out_path)
    print('Watertight?', mesh.is_watertight)
    if not mesh.is_watertight:
        broken_before = trimesh.repair.broken_faces(mesh)
        is_watertight = mesh.fill_holes()
        broken_after = trimesh.repair.broken_faces(mesh)
        print('  #broken: {}, {}'.format(len(broken_before), len(broken_after)))

    return out_path


def process_img(args):
    name, in_dir, out_dir = args

    img = Image.open(os.path.join(in_dir, name))

    W, H = img.size

    if not name == 'all.png' and (W != FLAGS.scale_size or H != FLAGS.scale_size):
        img = img.resize((FLAGS.scale_size, FLAGS.scale_size))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.8)

    out_path = os.path.join(out_dir, name)
    img.save(out_path)

    return name


def fill_missing_depth_images(data_dir, img_list):

    parent_dir = os.path.dirname(data_dir)
    src_basic_depth = os.path.join(parent_dir, 'bg_exrdepth', 'depth-64.exr')

    for f in img_list:
        print(f)
        dst = os.path.join(data_dir, 'depth', f.replace('.png', '.exr'))
        print(dst)
        copyfile(src_basic_depth, dst)


if __name__ == "__main__":

    FLAGS = parse_arguments()

    # ----------------------------------------------------------------------------------------
    # Image processing

    in_dir = os.path.join(FLAGS.data_dir, 'images')

    if not os.path.exists(in_dir):
        print('Input image directory does not exist; {}'.format(in_dir))
        exit(1)

    out_dir = os.path.join(FLAGS.data_dir, 'images_bright')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('Created output directory {}'.format(out_dir))
    else:
        print('Output directory exists; potentially overwriting contents ({}).'.format(out_dir))

    obj_list = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f)) and '.png' in f]
    obj_list.sort()

    # Image pre-processing (eg. crop+rescale, brightening)
    args = product(obj_list, [in_dir], [out_dir])
    with Pool(processes=NUM_THREADS) as pool:
        img_list_out = pool.map(process_img, args)

    train_imgs = []
    val_imgs = []

    # Create train/ val split (so far, only validated on novel views)
    # for obj in obj_list:
    #     obj_dir = os.path.join(in_dir, obj)
    for obj in ['']:
        obj_dir = in_dir

        view_list = [f for f in os.listdir(obj_dir)
                     if os.path.isfile(os.path.join(obj_dir, f)) and '.png' in f and not 'all.png' in f]
        # random.shuffle(view_list)
        view_list_tmp = [f for f in view_list
                       if os.path.isfile(os.path.join(FLAGS.data_dir, 'depth', f.replace('.png', '.exr')))]
        view_list_mising_depth = [f for f in view_list
                       if not os.path.isfile(os.path.join(FLAGS.data_dir, 'depth', f.replace('.png', '.exr')))]
        view_list = view_list_tmp + view_list_mising_depth

        # Train/ val split
        n_imgs = len(view_list)
        n_test = int(FLAGS.test_ratio * n_imgs)
        if n_test == 0:
            train_split = view_list
            val_split = []
        else:
            train_split = view_list[:-n_test]
            val_split = view_list[-n_test:]

        train_imgs.extend([os.path.join(obj, view) for view in train_split])
        val_imgs.extend([os.path.join(obj, view) for view in val_split])

    # Write txt files with img paths for train/ val split
    with open(os.path.join(out_dir, 'train_images.txt'), 'w') as filehandle:
        filehandle.writelines("%s\n" % img for img in train_imgs)
    with open(os.path.join(out_dir, 'val_images.txt'), 'w') as filehandle:
        filehandle.writelines("%s\n" % img for img in val_imgs)

    with open(os.path.join(out_dir, 'missig_depth.txt'), 'w') as filehandle:
        filehandle.writelines("%s\n" % img for img in view_list_mising_depth)

    # Combine all images in single image for visualization
    n_imgs = [5, 5]
    img_list = [[] for _ in range(n_imgs[0])]
    for i in range(n_imgs[0] * n_imgs[1]):
        if i < len(view_list):
            img = Image.open(os.path.join(out_dir, view_list[i]))
        else:
            img = np.zeros_like(img_list[0][0])
        img_list[int(i / n_imgs[1])].append(np.asarray(img))
    for x in range(n_imgs[0]):
        img_list[x] = np.hstack(img_list[x])
    img_list = np.vstack(img_list)

    imgs_comb = Image.fromarray(img_list)
    imgs_comb.save(os.path.join(out_dir, 'all.png'))

    # # Depth maps - fill-missing data
    # fill_missing_depth_images(FLAGS.data_dir, view_list_mising_depth)

    # Depth maps - remove alpha
    depth_dir = os.path.join(FLAGS.data_dir, 'depth')
    if os.path.exists(depth_dir):
        depth_list = [f for f in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, f)) and '-0001.png' in f]
        depth_list.sort()

        for name in depth_list:
            if '-0001.png' in name:
                depth_path = os.path.join(depth_dir, name)
                img = Image.open(depth_path)
                img.putalpha(256)
                img.save(depth_path)
                os.rename(depth_path, depth_path.replace('-0001.png', '.png'))

    # depth_dir = os.path.join(FLAGS.data_dir, 'depth_exr')
    depth_dir = os.path.join(FLAGS.data_dir, 'depth')
    # depth_list = [f for f in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, f)) and '-0001.exr' in f]
    depth_list = [f for f in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, f))]
    depth_list.sort()

    for name in depth_list:
        if '-0001.exr' in name:
            depth_path = os.path.join(depth_dir, name)
            os.rename(depth_path, depth_path.replace('-0001.exr', '.exr'))
        if '_0001.exr' in name:
            depth_path = os.path.join(depth_dir, name)
            os.rename(depth_path, depth_path.replace('_0001.exr', '.exr'))

    # ----------------------------------------------------------------------------------------
    # Mesh (OFF files) processing
    in_dir = os.path.join(FLAGS.data_dir, 'objects', 'original_watertight')
    out_dir = os.path.join(FLAGS.data_dir, 'objects', 'triangular')

    if not os.path.exists(in_dir):
        print('Input object directory does not exist; {}'.format(in_dir))
        exit(0)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('Created output directory {}'.format(out_dir))
    else:
        print('Output directory exists; potentially overwriting contents ({}).'.format(out_dir))

    obj_list = [f for f in os.listdir(in_dir) if (os.path.isfile(os.path.join(in_dir, f)) and not '_tmp.obj' in f)]
    obj_list.sort()
    # obj_list = [f.replace('_v00000.png', '.obj') for f in view_list]

    DEBUG = False
    if DEBUG:
        for obj_name in obj_list[0:1]:
            print(obj_name)
            obj_path_out = process_obj((obj_name, in_dir, out_dir))
            mesh = trimesh.load(os.path.join(out_dir, obj_name).replace('.obj', '.off'))
            # print('Watertight?', mesh.is_watertight)
            # mesh.show()
        exit(0)

    # Get triangular mesh representations
    print(datetime.datetime.now().time())
    args = product(obj_list, [in_dir], [out_dir])
    with Pool(processes=NUM_THREADS) as pool:
        obj_list_out = pool.map(process_obj, args)
    print(datetime.datetime.now().time())

    print('{} objects processed.'.format(len(obj_list_out)))

    exit(0)
