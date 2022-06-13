#!/usr/bin/env python
#
#
# Copyright 2019 Joerg Stueckler <joerg.stueckler at tuebingen dot mpg dot de> (Max Planck Institute for Intelligent Systems). All Rights Reserved.
#
#

import os
import argparse

import numpy as np
import trimesh
from PIL import Image

import utils.viz as viz


IMG_SIZE = 64


def slice_meshes(input, outdir):

    filelist = []

    for filename in os.listdir(input):

        filepath = os.path.join(input, filename)
        outfile_rel = filename[:-4] + '.png'
        outfile_abs = os.path.join(outdir, filename[:-4] + '.png')
        print(filepath)

        slices_names = ['x', 'y', 'z']

        if not '.off' in filepath:
            continue
        if os.path.exists(outfile_abs.replace('.png', '_z.png')):
            for slice in slices_names:
                slice_outfile = outfile_rel.replace('.png', '_' + slice + '.png')
                filelist.append(slice_outfile)
            continue

        mesh = trimesh.load(filepath)

        X, Y = np.meshgrid(np.linspace(-1., 1., IMG_SIZE), np.linspace(1., -1., IMG_SIZE))  # order w.r.t. PIL.Image
        X = np.reshape(X, [-1])
        Y = np.reshape(Y, [-1])
        X = np.expand_dims(X, axis=1)
        Y = np.expand_dims(Y, axis=1)
        x_slices = np.concatenate([np.zeros_like(X), X, Y], axis=1)
        y_slices = np.concatenate([X, np.zeros_like(X), Y], axis=1)
        z_slices = np.concatenate([X, Y, np.zeros_like(X)], axis=1)
        slices = [x_slices, y_slices, z_slices]

        for i, slice in enumerate(slices):

            # sdf_values = - trimesh.proximity.signed_distance(mesh, slice)
            n = 8
            assert ((IMG_SIZE**2)%n == 0)
            batch_size = int((IMG_SIZE ** 2) / n)
            sdf_values = []
            for j in range(n):
                print('-- {}/{}, {}'.format(j, n, sum([len(sdf_values[i]) for i in range(len(sdf_values))])))
                sdf_values.append(- trimesh.proximity.signed_distance(mesh, slice[batch_size*j:batch_size*(j+1)]))
            sdf_values = np.reshape(np.asarray(sdf_values), (-1))

            if sdf_values.size < 1000:
                print("too few samples ({})".format(sdf_values.size ))
                continue

            if np.isnan(sdf_values).any():
                print("nan samples", np.sum(np.isnan(sdf_values)))
                sdf_values[np.isnan(sdf_values)] = 0.
                # continue

            if np.isinf(sdf_values).any():
                print("inf samples")
                continue

            sdf_values = np.reshape(sdf_values, [IMG_SIZE, IMG_SIZE])

            sdf_color = viz.slice_coloring(sdf_values)
            img = Image.fromarray(np.uint8(255*sdf_color))

            img.save(outfile_abs.replace('.png', '_' + slices_names[i]+'.png'))

            filelist.append(outfile_rel.replace('.png', '_' + slices_names[i]+'.png'))

    return filelist


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate slices of the SDF values of meshes.')
    parser.add_argument('--input', type=str, help='The input base directory containing OFF files.')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input directory does not exist.')
        exit(1)

    out_dir = os.path.join(os.path.dirname(args.input), 'slices')
    if not os.path.exists(out_dir):
        print('Create output directory {}.'.format(out_dir))
        os.makedirs(out_dir)
    else:
        print('Output directory {} exists; potentially overwriting contents.'.format(out_dir))

    filelist = slice_meshes(args.input, out_dir)

    with open(os.path.join(out_dir, 'slices.txt'), 'w') as f:
        for path in filelist:
            f.write(path + os.linesep)





