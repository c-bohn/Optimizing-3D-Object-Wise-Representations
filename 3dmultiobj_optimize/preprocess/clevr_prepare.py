import os
import argparse
import datetime
from multiprocessing import Pool
from itertools import product

from PIL import Image, ImageEnhance
#import pymesh
import trimesh
import numpy as np

NUM_THREADS = 6

CROP_SIZE = 128
SCALE_SIZE = 64


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/is/sg2/celich/projects/DataSets/clevr-dataset-gen/outputs/objects1_small_64', help='Data dir')
    parser.add_argument('--proc_obj', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--test_ratio', type=float, default=-1)
    FLAGS = parser.parse_args()
    return FLAGS


def process_img(args):
    name, in_dir, out_dir = args

    img = Image.open(os.path.join(in_dir, name))

    W, H = img.size

    if W!=SCALE_SIZE or H!=SCALE_SIZE:
        img = img.crop(box=((W-CROP_SIZE)/2, (H-CROP_SIZE)/2, W-(W-CROP_SIZE)/2, H-(H-CROP_SIZE)/2))
        img = img.resize((SCALE_SIZE, SCALE_SIZE))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.8)         # 1.8 std; 1.2 for scenes with more lightning
    # enhancer = ImageEnhance.Contrast(img)
    # img = enhancer.enhance(1.2)

    # noise_rgb = 255*np.concatenate([np.random.normal(0., 0.025, [1, 1, 3]),[[[0]]]], axis=-1)
    # img += noise_rgb
    # img = Image.fromarray(np.uint8(img))

    out_path = os.path.join(out_dir, name)
    img.save(out_path)

    return name


def process_obj_off(args):
    name, in_dir, out_dir = args

    in_path = os.path.join(in_dir, name)
    out_path = os.path.join(out_dir, name)

    if os.path.exists(out_path):
        return out_path

    # Make triangular mesh (naive approach)
    vertice_lines = []
    faces_lines = []
    with open(in_path, "r") as f_in:
        f = f_in.readline()
        print(in_path, f)
        assert f == 'OFF\n', 'No OFF file format.'

        n_vertices, n_faces, n_edges = f_in.readline().split()
        assert n_edges == '0', 'Unknown number of edges, {}'.format(n_edges)

        for i in range(int(n_vertices)):
            vertice_lines.append(f_in.readline())

        for i in range(int(n_faces)):
            l = f_in.readline().split()
            if l[0] == '3':
                assert len(l) == 4, 'Incorrect format for file: {}, {}'.format(l, name)
                faces_lines.append(l[0] + ' ' + l[1] + ' ' + l[2] + ' ' + l[3] + '\n')
            elif l[0] == '4':
                assert len(l) == 5, 'Incorrect format for file: {}, {}'.format(l, name)
                faces_lines.append('3 ' + l[1] + ' ' + l[2] + ' ' + l[3] + '\n')
                faces_lines.append('3 ' + l[1] + ' ' + l[3] + ' ' + l[4] + '\n')
            else:
                print(
                    'process_obj_off: Only meshes that have faces with 4 vertices are considered, but this has {}. {}'.format(
                        l[0], in_dir))
                exit(1)

    with open(out_path, "w") as f_out:
        f_out.write('OFF\n')
        f_out.write(str(len(vertice_lines)) + ' ' + str(len(faces_lines)) + ' 0\n')

        for l in vertice_lines:
            f_out.write(l)
        for l in faces_lines:
            f_out.write(l)

    # Edge Collapsing (fewer edges/ vertices)
    mesh = pymesh.load_mesh(out_path)
    mesh_n_vertices_before = mesh.num_vertices
    mesh_n_faces_before = mesh.num_faces
    if 'Cube' in out_path:
        tol = 0.02
    elif 'Cylinder' in out_path:
        tol = 0.015
    else:
        tol = 0.05
    mesh, info = pymesh.collapse_short_edges(mesh, tol, preserve_feature=True)

    print('After edge collapsing: vertices {}, faces {}'.format(mesh.num_vertices/mesh_n_vertices_before,
                                                                mesh.num_faces/mesh_n_faces_before))
    pymesh.save_mesh(out_path, mesh)

    # Remove '# Generated with PyMesh' line from file
    with open(out_path, "r") as f:
        lines = f.readlines()
    with open(out_path, "w") as f:
        for line in lines:
            if 'PyMesh' in line:
                continue
            f.write(line)

    return out_path


if __name__ == "__main__":

    FLAGS = parse_arguments()

    # ----------------------------------------------------------------------------------------
    # Image processing
    img_folders = ['images']
    for folder in img_folders:
        in_dir = os.path.join(FLAGS.data_dir, folder)
        out_dir = os.path.join(FLAGS.data_dir, folder+'_bright')

        if not os.path.exists(in_dir):
            print('Input directory does not exist; {}'.format(in_dir))
            continue

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print('Created output directory {}'.format(out_dir))
        else:
            print('Output directory exists; potentially overwriting contents ({}).'.format(out_dir))

        print(out_dir)

        img_list = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f)) and '.png' in f]
        img_list.sort()
        if FLAGS.n_samples!=-1 and FLAGS.n_samples<len(img_list):
            img_list = img_list[:FLAGS.n_samples]

        DEBUG = False
        if DEBUG:
            img_path_out = process_img((img_list[0], in_dir, out_dir))
            exit(0)

        # Image pre-processing (eg. crop+rescale, brightening)
        args = product(img_list, [in_dir], [out_dir])
        with Pool(processes=NUM_THREADS) as pool:
            img_list_out = pool.map(process_img, args)

        # Train/ val split
        n_imgs = len(img_list)
        n_test = int(FLAGS.test_ratio * n_imgs)
        if n_test%2==1:
            n_test-=1
        if n_test == 0:
            train_img_list_out = img_list_out
            test_img_list_out = [0]
        else:
            train_img_list_out = img_list_out[:-n_test]
            test_img_list_out = img_list_out[-n_test:]

        # Write txt files with img paths for train/ val split
        with open(os.path.join(out_dir, 'train_images.txt'), 'w') as filehandle:
            filehandle.writelines("%s\n" % img for img in train_img_list_out)
        with open(os.path.join(out_dir, 'val_images.txt'), 'w') as filehandle:
            filehandle.writelines("%s\n" % img for img in test_img_list_out)

        # Combine all images in single image for visualization
        n_imgs = [3, 6]
        img_list = [[] for _ in range(n_imgs[0])]
        for i in range(n_imgs[0]*n_imgs[1]):
            if i < len(img_list_out):
                img = Image.open(os.path.join(out_dir, img_list_out[i]))
            else:
                img = np.zeros_like(img_list[0][0])
            img_list[int(i/n_imgs[1])].append(np.asarray(img))
        for x in range(n_imgs[0]):
            img_list[x] = np.hstack(img_list[x])
        img_list = np.vstack(img_list)

        imgs_comb = Image.fromarray(img_list)
        imgs_comb.save(os.path.join(out_dir, 'all.png'))

        print('{} images processed.'.format(len(img_list_out)))

    # Depth maps - remove alpha
    depth_dir = os.path.join(FLAGS.data_dir, 'depth')
    depth_list = [f for f in os.listdir(depth_dir) if os.path.isfile(os.path.join(depth_dir, f)) and ('.png' in f or '.exr' in f)]
    depth_list.sort()
    if FLAGS.n_samples != -1 and FLAGS.n_samples < len(depth_list):
        depth_list = depth_list[:FLAGS.n_samples]
    for name in depth_list:
        if '_0001.png' in name:
            img = Image.open(os.path.join(depth_dir, name))
            img.putalpha(255)
            img.save(os.path.join(depth_dir, name.replace('_0001.png', '.png')))
            if '_0001.png' in name:
                os.remove(os.path.join(depth_dir, name))
    for name in depth_list:
        if '_0001.exr' in name:
            depth_path = os.path.join(depth_dir, name)
            os.rename(depth_path, depth_path.replace('_0001.exr', '.exr'))

    # # ----------------------------------------------------------------------------------------
    # # Mesh (OFF files) processing

    if FLAGS.proc_obj == 1:
        in_dir = os.path.join(FLAGS.data_dir, 'objects', 'original')
        out_dir = os.path.join(FLAGS.data_dir, 'objects', 'triangular')

        if not os.path.exists(in_dir):
            print('Input directory does not exist; {}'.format(in_dir))
            exit(1)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print('Created output directory {}'.format(out_dir))
        else:
            print('Output directory exists; potentially overwriting contents ({}).'.format(out_dir))

        obj_list = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f)) and '.off' in f]
        obj_list.sort()
        if FLAGS.n_samples>0:
            obj_list = obj_list[:FLAGS.n_samples]

        DEBUG = False
        if DEBUG:
            for obj_name in obj_list[0:1]:
                print(obj_name)
                obj_path_out = process_obj_off((obj_name, in_dir, out_dir))
                mesh = trimesh.load(os.path.join(out_dir, obj_name.replace('.ply', '.off')))
                print('Watertight?', mesh.is_watertight)
                mesh.show()

            exit(0)

        # Get triangular mesh representations
        print(datetime.datetime.now().time())
        args = product(obj_list, [in_dir], [out_dir])
        with Pool(processes=NUM_THREADS) as pool:
            obj_list_out = pool.map(process_obj_off, args)
        print(datetime.datetime.now().time())

        print('{} objects processed.'.format(len(obj_list_out)))

    exit(0)
