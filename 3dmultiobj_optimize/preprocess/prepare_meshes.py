#!/usr/bin/env python
#
#
# Copyright 2019 Joerg Stueckler <joerg.stueckler at tuebingen dot mpg dot de> (Max Planck Institute for Intelligent Systems). All Rights Reserved.
# Adapted by Cathrin Elich
#
#

import argparse

from scale_off import *
import clevr_config as data_config
# from preprocess.scale_off import *
# import preprocess.clevr_config as data_config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rescale and center OFF meshes.')
    parser.add_argument('--input', type=str, help='The input base directory with objects of different types containing OFF files.')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('Input directory does not exist.')
        exit(1)

    out_dir = os.path.join(os.path.dirname(args.input), 'scaled')
    if not os.path.exists(out_dir):
        print('Create output directory {}.'.format(out_dir))
        os.makedirs(out_dir)
    else:
        print('Output directory {} exists; potentially overwriting contents.'.format(out_dir))

    filelists = scale_off(args.input, out_dir)
    filelists.sort()

    # write out the filelist
    f = open(os.path.join(out_dir, 'all_meshes.txt'), 'w')
    for v in filelists:
        key = data_config.get_category(v)
        f.write(str(key) + ',' + v + ',%s' % (data_config.LABEL_MAP[key]) + os.linesep)
    f.close()

