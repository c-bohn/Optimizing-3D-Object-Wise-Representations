#!/usr/bin/env python
#
#
# Copyright 2019 Joerg Stueckler <joerg.stueckler at tuebingen dot mpg dot de> (Max Planck Institute for Intelligent Systems). All Rights Reserved.
# Adapted by Cathrin Elich
#
#

import os
import argparse
import numpy as np
import trimesh

# from preprocess.clevr_config import *

LABEL_MAP = ['cube',  'sphere',  'cylinder', 'unknown']

def get_category(name):
    if 'Cube' in name:
        return 0
    elif 'Sphere' in name:
        return 1
    elif 'Cylinder' in name:
        return 2
    else:
        return -1


def sample_meshes(input, output, num_on_surface_samples=1000, num_off_surface_samples=4000, num_sample_sets=100):
    filelist_on_surface = []
    filelist_off_surface = []
    filelist = []

    input_list = [f for f in os.listdir(input) if '.off' in f]

    for i, filename in enumerate(input_list):

        filepath_input = os.path.join(input, filename)

        if not '.off' in filepath_input:
            continue
        print(filepath_input, i+1, '/', len(input_list))

        mesh = trimesh.load(filepath_input)
        # mesh.show()

        for k in range(0, num_sample_sets):

            if num_sample_sets>1:
                print("sample %d/%d" % (k, num_sample_sets))

            filename_on = filename[:-4] + '_samples_on_surface%d.npz' % k
            filename_off = filename[:-4] + '_samples_off_surface%d.npz' % k
            filename_all = filename[:-4] + '_samples%d.npz' % k
            filelist_on_surface.append(filename_on)
            filelist_off_surface.append(filename_off)
            filelist.append(filename_all)

            filename_abs = os.path.join(output, filename[:-4] + '_samples%d.npz' % k)
            if os.path.exists(filename_abs):
                continue

            # sample points in the volume (off surface)
            # points_off_surface = np.random.uniform(-1.0, 1.0, (num_off_surface_samples, 3))
            # assert (num_off_surface_samples % num_on_surface_samples == 0)
            N_PNTS = 1000
            assert (num_off_surface_samples % N_PNTS == 0)
            assert (num_on_surface_samples % N_PNTS == 0)
            n_on = int(num_on_surface_samples / N_PNTS)
            n_off = int(num_off_surface_samples / N_PNTS)

            def handle_nan(sdf_samples):
                while np.isnan(sdf_samples).any():
                    print('NAN', np.count_nonzero(np.isnan(sdf_samples)))
                    points_on_surface_add = trimesh.sample.sample_surface(mesh, num_on_surface_samples)
                    sdf_samples_add = - trimesh.proximity.signed_distance(mesh, points_on_surface_add[0])
                    sdf_samples[np.isnan(sdf_samples)] = sdf_samples_add[np.isnan(sdf_samples)]
                return sdf_samples

            # sample points on the mesh (on surface)
            try:
                # points_on_surface = trimesh.sample.sample_surface(mesh, num_on_surface_samples)
                # # -> trimesh function yields negative values for points outside of mesh and vise versa
                # # sdf_samples_on_surface = - trimesh.proximity.signed_distance(mesh, points_on_surface[0])

                print('--sample on surface')
                sdf_samples_near_surface = []
                points_near_surface = []
                for i in range(n_on):
                    if i % 5 == 0:
                        print('--{}/{}'.format(i+1, n_on))
                    points_on = trimesh.sample.sample_surface(mesh, N_PNTS)
                    # add some noise
                    pnts_near1 = points_on[0] + np.random.normal(0., 0.00025, points_on[0].shape)
                    pnts_near2 = points_on[0] + np.random.normal(0., 0.0025, points_on[0].shape)
                    smpls1 = - trimesh.proximity.signed_distance(mesh, pnts_near1)
                    smpls2 = - trimesh.proximity.signed_distance(mesh, pnts_near2)

                    # handle nan results
                    handle_nan(smpls1)
                    handle_nan(smpls2)
                    smpls = np.concatenate([smpls1, smpls2], axis=0)
                    pnts_near = np.concatenate([pnts_near1, pnts_near2], axis=0)

                    sdf_samples_near_surface.append(smpls)
                    points_near_surface.append(pnts_near)
                sdf_samples_near_surface = np.reshape(np.asarray(sdf_samples_near_surface), (2*num_on_surface_samples))
                points_near_surface = np.reshape(points_near_surface, (2*num_on_surface_samples, 3))

                print('--sample ramdomly')
                sdf_samples_off_surface = []
                points_off_surface = []
                for i in range(n_off):
                    if i % 5 == 0:
                        print('--{}/{}'.format(i+1, n_off))
                    points_off_surface.append(np.random.uniform(-1.0, 1.0, (N_PNTS, 3)))
                    smpls = - trimesh.proximity.signed_distance(mesh, points_off_surface[i])
                    sdf_samples_off_surface.append(smpls)
                sdf_samples_off_surface = np.reshape(np.asarray(sdf_samples_off_surface), (num_off_surface_samples))
                points_off_surface = np.reshape(points_off_surface, (num_off_surface_samples, 3))

                sdf_samples_near_surface = np.expand_dims(sdf_samples_near_surface, axis=1)
                sdf_samples_off_surface = np.expand_dims(sdf_samples_off_surface, axis=1)

                # samples_on_surface = np.concatenate([points_on_surface[0], sdf_samples_on_surface], axis=1)
                samples_near_surface = np.concatenate([points_near_surface, sdf_samples_near_surface], axis=1)
                samples_off_surface = np.concatenate([points_off_surface, sdf_samples_off_surface], axis=1)
                sdf_samples_all = np.concatenate([samples_near_surface, samples_off_surface], axis=0)

                if sdf_samples_near_surface.size < num_on_surface_samples or \
                        sdf_samples_off_surface.size < num_off_surface_samples:
                    print("too few samples")
                    continue
                if np.isnan(sdf_samples_all).any():
                    print("nan samples")
                    continue
                if np.isinf(sdf_samples_all).any():
                    print("inf samples")
                    continue

                filename_all = os.path.join(output, filename[:-4] + '_samples%d.npz' % k)
                np.savez(filename_all, sdf_samples_all)

            except ValueError as error:
                print("Sampling failed ({})".format(error))

    return [filelist_on_surface, filelist_off_surface, filelist]


def write_filelist(filelist, path):
    with open(path, 'w') as f:
        for v in filelist:
            key = get_category(v)
            f.write(str(key) + ',' + v + ',%s' % (LABEL_MAP[key]) + os.linesep)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate a dataset of samples on the meshes on-surface and off-surface.')
    parser.add_argument('--input', type=str, help='The input base directory containing OFF files.')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input directory does not exist.')
        exit(1)

    out_dir = os.path.join(os.path.dirname(args.input), 'sampled_sdf_near_more')
    if not os.path.exists(out_dir):
        print('Create output directory {}.'.format(out_dir))
        os.makedirs(out_dir)
    else:
        print('Output directory {} exists; potentially overwriting contents.'.format(out_dir))

    [filelist_on_surface, filelist_off_surface, filelist] = \
            sample_meshes(args.input, out_dir, num_on_surface_samples=50000, num_off_surface_samples=400000, num_sample_sets=1)
        # sample_meshes(args.input, out_dir, num_on_surface_samples=15000, num_off_surface_samples=5000, num_sample_sets=1)

    # write out the filelists
    write_filelist(filelist, os.path.join(out_dir, 'samples.txt'))
