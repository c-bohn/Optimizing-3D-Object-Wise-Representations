import os
import numpy as np
from PIL import Image
import itertools
import importlib    # for (test) main method
import json
import math
from tqdm import *
#import pyexr
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
plt.style.use('classic')

# import utils.viz as viz

# --------------------------------------------------

DATASET_TYPES = ['train', 'val', 'test']


def load_bg_img(data_dir, size):
    bg_dir = os.path.join(os.path.dirname(data_dir), 'bg_exrdepth')

    rgb = np.load(os.path.join(bg_dir, 'rgb-'+str(size)+'.npy'))
    depth = np.load(os.path.join(bg_dir, 'depth-'+str(size)+'.npy'))

    return [rgb, depth]


class DataSet:

    def __init__(self, data_dir, type, cnfg):
        """
        Initialize dataset object
        :param data_dir:    string, directory of data
        :param type:        string, type of dataset (from DATASET_TYPES)
        :param cnfg:        dict, dataset configuration
        """

        if not os.path.exists(data_dir):
            print('DataSet.__init__(): Directory {} does not exist.'.format(data_dir))
            exit(1)
        assert type in DATASET_TYPES

        self.data_dir = os.path.join(data_dir, 'input_'+type)
        self.type = type

        self.img_size = cnfg['img_size']
        self.n_samples = cnfg['n_samples']
        self.max_n_obj = cnfg['max_n_obj']
        self.n_images = cnfg['n_images']
        self.depth_max = cnfg['depth_max']

    # --------------------------------
    # --- Load Dataset:

    def load_data(self):
        """
        Load image dataset. (used during initialization)
        """

        with open(os.path.join(self.data_dir, 'scene_names.txt'), 'r') as f:
            scene_names = f.read().splitlines()
        rgb_imgs = np.load(os.path.join(self.data_dir, 'rgb.npy'))

        return scene_names, rgb_imgs

    # --------------------------------
    # --- General functions:

    def get_size(self):
        """
        :return:    int, number of data samples
        """
        return len(self.scene_names)

    # --- Prepare input for model:

    def prepare_input(self, idx, rgb_noise=False):
        """
        Convert data to input format for network
        :param idx:           int,        index of object sample
        :return: smpl_id:         (), int16
                 samples:         (N, 4), float32   -> (x,y,z,d)
        """

        scene_name = self.scene_names[idx]
        scene_id = np.int32(scene_name.split('_')[-1]) * np.ones(1, dtype=np.int32)

        rgb_in = np.copy(self.rgb_imgs[idx])
        rgb_in = np.float32((1. / 255) * rgb_in)

        # Data Augmentation
        if rgb_noise and self.type == 'train':

            # gaussian noise (only input)
            eps = 0.01
            rgb_in += np.random.normal(0., eps, rgb_in.shape)
            rgb_in = np.clip(rgb_in, 0., 1.)

        assert (scene_id.dtype == np.int32)
        assert (rgb_in.dtype == np.float32)

        return scene_id, rgb_in

    def generator(self):
        """
        :return:    Generator for data sample ids
        """
        for i in itertools.cycle(self.ids):
            yield (i)


class DataSetShapeDec(DataSet):

    def __init__(self, data_dir, type, cnfg):
        super(DataSetShapeDec, self).__init__(data_dir, type, cnfg)

        # # Load Data
        scene_names, rgb_imgs, obj_names, obj_slices, obj_sdf_smpls = self.load_data()
        # - general info
        self.scene_names = scene_names
        self.ids = np.arange(self.get_size())
        # - 2D data
        self.rgb_imgs = rgb_imgs
        # - 3D object data
        self.obj_names = obj_names
        self.obj_slices = obj_slices
        self.obj_sdf_smpls = obj_sdf_smpls

        print('Dataset (ShapeDec) loaded: type {}, size {}'.format(self.type, self.get_size()))

    def load_data(self):
        scene_names, rgb_imgs = super(DataSetShapeDec, self).load_data()

        with open(os.path.join(self.data_dir, 'obj_names.txt'), 'r') as f:
            obj_names = f.read().splitlines()
        obj_slice_imgs = np.load(os.path.join(self.data_dir, 'obj_slices.npy'))
        obj_sdf_smpls = np.load(os.path.join(self.data_dir, 'obj_sdf_smpls.npy'))

        return scene_names, rgb_imgs, obj_names, obj_slice_imgs, obj_sdf_smpls

    def prepare_input(self, idx):
        scene_id, rgb = super(DataSetShapeDec, self).prepare_input(idx)

        obj_name = self.obj_names[idx][0]  # only one object atm (i.e. pre-training of shape decoder)
        n_on = int(self.n_samples / 5)
        n_off = self.n_samples - n_on

        obj_sdf = self.obj_sdf_smpls[idx]
        n_total_sampled_pnts = obj_sdf.shape[0]
        assert (n_total_sampled_pnts % 5 == 0)

        # sampling subset of pnt/sdf value pairs
        obj_sdf_on_surface = obj_sdf[np.random.choice(range(int(n_total_sampled_pnts / 5)),
                                                      n_on, replace=False)]
        obj_sdf_off_surface = obj_sdf[np.random.choice(range(int(n_total_sampled_pnts / 5), n_total_sampled_pnts),
                                                       n_off, replace=False)]
        obj_sdf = np.concatenate([obj_sdf_on_surface, obj_sdf_off_surface])

        # slice visualization
        obj_slices = self.obj_slices[idx]

        assert (obj_sdf.dtype == np.float32)
        assert (obj_slices.dtype == np.uint8)

        return scene_id, rgb, obj_sdf, obj_slices


class DataSetMultiObj(DataSet):

    def __init__(self, data_dir, type, cnfg):
        super(DataSetMultiObj, self).__init__(data_dir, type, cnfg)

        # # Load Data
        scene_names, rgb_imgs, depth_imgs, msk_imgs, obj_extrs = self.load_data()
        # - general info
        self.scene_names = scene_names
        self.ids = np.arange(self.get_size())
        # - 2D data
        self.rgb_imgs = rgb_imgs
        self.depth_imgs = depth_imgs
        self.msk_imgs = msk_imgs
        # - 3D object data
        self.obj_extrs = obj_extrs

        print('Dataset (MultiObj) loaded: type {}, size {}'.format(self.type, self.get_size()))

    def load_data(self):
        scene_names, rgb_imgs = super(DataSetMultiObj, self).load_data()

        depth_imgs = np.load(os.path.join(self.data_dir, 'depth.npy'))
        msk_imgs = np.load(os.path.join(self.data_dir, 'mask.npy'))
        obj_extrs = np.load(os.path.join(self.data_dir, 'obj_extrs.npy'))

        return scene_names, rgb_imgs, depth_imgs, msk_imgs, obj_extrs

    def prepare_input(self, idx):
        scene_id, rgb_in = super(DataSetMultiObj, self).prepare_input(idx, rgb_noise=True)

        rgb_out = np.float32((1. / 255) * np.copy(self.rgb_imgs[idx]))

        depth = np.copy(self.depth_imgs[idx])
        msk = np.float32(np.copy(self.msk_imgs[idx]))

        obj_extr = np.float32(self.obj_extrs[idx])

        assert (rgb_out.dtype == np.float32)
        assert (depth.dtype == np.float32)
        assert (msk.dtype == np.float32)

        return scene_id, rgb_in, rgb_out, depth, msk, obj_extr


# --------------------------------------------------

def load_cnfg(name):
    cnfg_name = name
    cnfg_file = os.path.join('config', cnfg_name+'.py')
    spec = importlib.util.spec_from_file_location(cnfg_name, cnfg_file)
    cnfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cnfg)
    return cnfg


def helper_show_imgs(imgs):
    for i, img in enumerate(imgs):
        if img.dtype == np.float32 or np.max(img) <= 1.:
            img = 255. * img
        if img.shape[-1] > 3:
            img = np.expand_dims(img, axis=-1)
        if img.shape[-1] == 1:
            img = np.tile(img, (1, 1, 3))
        imgs[i] = img
    img = np.concatenate(imgs, axis=1).astype(np.uint8)
    Image.fromarray(img).show()


def test_clevr_shapedec(data_dir):
    cnfg = load_cnfg('cnfg_deepsdf_clevr')

    dataset = DataSetShapeDec(data_dir, 'train', cnfg.data)
    # dataset.shuffle_data()
    data_gen = dataset.generator()

    # Iterate over data
    n = 0
    for id in data_gen:
        inputs = dataset.prepare_input(id)
        scene_id, rgb, obj_sdf, obj_slices = inputs
        print('Sample {}: {}'.format(scene_id, dataset.scene_names[scene_id[0]]))

        imgs = [rgb[0]] + [obj_slices[i] for i in range(3)]
        helper_show_imgs(imgs)

        n += 1
        if n >= 1:
            exit(0)


def test_clevr_multiobj(data_dir):
    cnfg = load_cnfg('cnfg_mosnet-org_obj3_clevr')

    dataset = DataSetMultiObj(data_dir, 'train', cnfg.data)
    # dataset.shuffle_data()
    data_gen = dataset.generator()

    # Iterate over data
    n = 0
    for id in data_gen:
        inputs = dataset.prepare_input(id)
        scene_id, rgb_in, rgb_out, depth, msk, obj_extrs = inputs
        print('Sample {}: {}'.format(scene_id, dataset.scene_names[scene_id[0]]))

        imgs = [rgb_in[0], rgb_out[0], depth[0]] + [msk[i, 0] for i in range(dataset.max_n_obj)]
        helper_show_imgs(imgs)

        n += 1
        if n >= 1:
            exit(0)


if __name__ == "__main__":

    test_clevr_shapedec('Z:/datasets/ICCV21/clevr/obj_gt-sdf')
    # test_clevr_multiobj('Z:/datasets/ICCV21/clevr/objects3_train')