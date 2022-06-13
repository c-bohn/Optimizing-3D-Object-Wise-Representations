
import os
import argparse

import numpy as np
import trimesh
from PIL import Image

import utils.viz as viz


SIZE = 64


def rec_meshes(input, outdir):

    filelist = []

    for filename in os.listdir(input):

        filepath = os.path.join(input, filename)
        outfile_rel = filename[:-4] + '.png'
        outfile_abs = os.path.join(outdir, filename[:-4] + '.png')
        print(filepath)

        slices_names = ['x', 'y', 'z']

        if not '.off' in filepath:
            continue
        if os.path.exists(outfile_abs):
            for slice in slices_names:
                slice_outfile = outfile_rel.replace('.png', '_' + slice + '.png')
                filelist.append(slice_outfile)
            continue

        mesh = trimesh.load(filepath)

        def gen_slice(axis, slice_pos=0.):
            N, M = np.meshgrid(np.linspace(-1., 1., SIZE), np.linspace(-1., 1., SIZE))

            N = np.reshape(N, [-1])
            M = np.reshape(M, [-1])
            N = np.expand_dims(N, axis=1)
            M = np.expand_dims(M, axis=1)

            if axis == 'x':
                slice = np.concatenate([slice_pos * np.ones_like(N), M, N], axis=1)
            elif axis == 'y':
                slice = np.concatenate([M, slice_pos * np.ones_like(N), N], axis=1)
            else:
                slice = np.concatenate([M, N, slice_pos * np.ones_like(N)], axis=1)

            return slice

        slice_coords=[]
        for s in np.linspace(-1., 1., SIZE):
            slice = gen_slice('y', slice_pos=s)
            slice_coords.append(slice)

        sdf_list = []
        for i, slice in enumerate(slice_coords):
            print('Slice No. {}/{}'.format(i+1, SIZE))
            sdf_values = - trimesh.proximity.signed_distance(mesh, slice)

            sdf_values = np.reshape(np.asarray(sdf_values), (-1))

            if np.isnan(sdf_values).any():
                print("nan samples", np.sum(np.isnan(sdf_values)))
                sdf_values[np.isnan(sdf_values)] = 0.
                # continue

            if np.isinf(sdf_values).any():
                print("inf samples")
                continue

            sdf_values = np.reshape(sdf_values, [SIZE, SIZE])
            sdf_list.append(sdf_values)

        sdf_vol = np.stack(sdf_list, axis=0)
        viz.show_mesh(sdf_vol, outfile_abs)

    return []


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate slices of the SDF values of meshes.')
    parser.add_argument('--input', type=str, help='The input base directory containing OFF files.')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input directory does not exist.')
        exit(1)

    out_dir = os.path.join(os.path.dirname(args.input), 'meshes')
    if not os.path.exists(out_dir):
        print('Create output directory {}.'.format(out_dir))
        os.makedirs(out_dir)
    else:
        print('Output directory {} exists; potentially overwriting contents.'.format(out_dir))

    filelist = rec_meshes(args.input, out_dir)

    exit()





