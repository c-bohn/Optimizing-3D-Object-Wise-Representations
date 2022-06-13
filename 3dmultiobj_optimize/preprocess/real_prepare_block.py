import os
import argparse
import datetime
import pyexr
import json
import random
import numpy as np

from multiprocessing import Pool
from itertools import product
from PIL import Image, ImageEnhance


NUM_THREADS = 1  # 6

IMG_SIZE = 64


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='C:/Users/Cathrin/Documents/MPI/datasets/real/bird_blocksworld', help='Data dir')
    FLAGS = parser.parse_args()
    return FLAGS


def crop_img(img):
    th = 0.45*255  #  0.5 * 255
    W, H = img.size
    off = 30
    min_border = 15
    W_new = W -2*off

    # img.show()
    img_np = np.asarray(img)
    img_r = np.tile(img_np[:, :, 0:1], (1, 1, 3))
    img_g = np.tile(img_np[:, :, 1:2], (1, 1, 3))
    img_b = np.tile(img_np[:, :, 2:3], (1, 1, 3))
    img_min = np.tile(np.min(img_np, axis=-1, keepdims=True), (1, 1, 3))
    img_th = 255 * np.less(img_min, th).astype(np.uint8)

    img_th_horizontal = np.max(img_th, axis=0, keepdims=True)
    img_th_vertical = np.max(img_th, axis=1, keepdims=True)
    img_th_l = np.argmax(img_th_horizontal[..., 0], axis=1)[0]
    img_th_r = W - np.argmax(np.fliplr(img_th_horizontal[..., 0]), axis=1)[0]
    img_th_u = np.argmax(img_th_vertical[..., 0], axis=0)[0]
    img_th_d = H - np.argmax(np.flipud(img_th_vertical[..., 0]), axis=0)[0]

    cont_w = img_th_r-img_th_l
    cont_h = img_th_d-img_th_u

    # print('H: {}, W: {}'.format(H, W))
    # print('- Left: {}, Right: {}, Up: {}, Down: {}'.format(img_th_l, img_th_r, img_th_u, img_th_d))
    # print('Content W: {}, H: {}'.format(cont_w, cont_h))

    img_th[:, img_th_l] = [[255, 0, 0]]
    img_th[:, img_th_r-1] = [[255, 0, 0]]
    img_th[img_th_u, :] = [[0, 255, 0]]
    img_th[img_th_d-1, :] = [[0, 255, 0]]
    img_stack = np.concatenate([img_np, img_r, img_g, img_b, img_min, img_th], axis=0)
    # Image.fromarray(img_stack).show()

    if img_th_l >= off+min_border and img_th_r+min_border <= off+W_new:
        img = img.crop(box=(off, 2 * off, W - off, H))
    elif cont_w <= W_new:  # places bb in the middle of cropped image
        border = (W_new-cont_w)/2.
        l = max([0, min([img_th_l-border, W-W_new])])
        img = img.crop(box=(l, 2 * off, l + W_new, H))
    else:
        img_th_horizontal_sum = np.sum(img_th, axis=0)[:, 0]
        l_best=-1
        cnt_best = 0
        for l in range(W-W_new):
            cnt = np.sum(img_th_horizontal_sum[l: l+W_new])
            if cnt > cnt_best:
                cnt_best = cnt
                l_best = l
        img = img.crop(box=(l_best, 2 * off, l_best + W_new, H))

    return img


def process_img(args):
    name, in_dir, out_dir = args

    img_full = Image.open(os.path.join(in_dir, name))
    W, H = img_full.size

    id = str(int((name.split('_')[-1]).split('.')[0])).zfill(3)
    name_new = "_".join([name.split('_')[0], id+'.png'])

    # crop with box (left, upper, right, lower)
    imgs = []
    # imgs.append(img_full.crop(box=(0, 0, W, W)))      # first image in video sequence
    imgs.append(img_full.crop(box=(0, H-W, W, H)))      # last image in video sequence

    for i, img in enumerate(imgs):

        img = crop_img(img)
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # enhancer = ImageEnhance.Brightness(img)
        # img = enhancer.enhance(1.8)

        out_path = os.path.join(out_dir, name_new.replace('.png', '_' + str(i) + '.png'))
        img.save(out_path)

    return name


if __name__ == "__main__":

    FLAGS = parse_arguments()

    if not os.path.exists(FLAGS.data_dir):
        print('Data directory does not exist; {}'.format(FLAGS.data_dir))

    # ----------------------------------------------------------------------------------------
    # Image processing

    in_dir = os.path.join(FLAGS.data_dir, 'block_camera_combined_scaled2')
    out_dir = os.path.join(FLAGS.data_dir, 'images_bright')

    if not os.path.exists(in_dir):
        print('Input directory does not exist; {}'.format(in_dir))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('Created output directory {}'.format(out_dir))
    else:
        print('Output directory exists; potentially overwriting contents ({}).'.format(out_dir))

    img_list = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f)) and
                '.png' in f and 'img_concat' in f]
    img_list.sort()
    # img_list = img_list[:10]

    DEBUG = False
    if DEBUG:
        img_path_out = process_img((img_list[25], in_dir, out_dir))
        exit(0)

    # Image pre-processing
    args = product(img_list, [in_dir], [out_dir])
    with Pool(processes=NUM_THREADS) as pool:
        _ = pool.map(process_img, args)

    img_list_name = [f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f)) and '_0.png' in f]
    # img_list_name.sort()
    random.shuffle(img_list_name)

    # Write txt files with img paths
    with open(os.path.join(out_dir, 'val_images.txt'), 'w') as filehandle:
        filehandle.writelines("%s\n" % img for img in img_list_name)

    # Combine all images in single image for visualization
    n_imgs = [3,6]
    img_list = [[] for _ in range(n_imgs[0])]
    for i in range(n_imgs[0] * n_imgs[1]):
        if i < len(img_list_name):
            img = Image.open(os.path.join(out_dir, img_list_name[i]))
        else:
            img = np.zeros_like(img_list[0][0])
        img_list[int(i / n_imgs[1])].append(np.asarray(img))
    for x in range(n_imgs[0]):
        img_list[x] = np.hstack(img_list[x])
    img_list = np.vstack(img_list)

    imgs_comb = Image.fromarray(img_list)
    imgs_comb.save(os.path.join(out_dir, 'all.png'))

    print('{} images processed.'.format(len(img_list_name)))

    # ----------------------------------------------------------------------------------------
    # Empty depth + normals + scene info

    sub_dirs = ['depth', 'instances', 'scenes']
    for dir in sub_dirs:
        out_dir = os.path.join(FLAGS.data_dir, dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    depth_basic_folder = os.path.join(os.path.dirname(FLAGS.data_dir), 'bg_exrdepth')
    depth_basic = pyexr.open(os.path.join(depth_basic_folder, 'depth-64.exr'))
    depth_basic = depth_basic.get()

    inst_basic = Image.fromarray(np.uint8(np.zeros((64, 64, 3))))

    scenes_names = {}
    for img_name in img_list_name:
        img_name = img_name.replace('.png', '')

        inst_basic.save(os.path.join(FLAGS.data_dir, 'instances',  img_name+'.png'))
        pyexr.write(os.path.join(FLAGS.data_dir, 'depth',  img_name+'.exr'), depth_basic)

        single_obj_dict = {
                  "size": 1,
                  "rotation": [0.],
                  "pixel_coords": [[32, 32, 4.]],
                  "3d_coords": [[0., 0., 1.0]]
                }
        scene_info = {
            "objects": [single_obj_dict for _ in range(3)]
        }
        name_json = img_name.replace('_1', '')+'.json'
        with open(os.path.join(FLAGS.data_dir, 'scenes', name_json), 'w') as f:
            json.dump(scene_info, f, indent=2)

    exit(0)
