#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:56:21 2018

@author: jiahuei
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time, argparse, numpy as np
import utils, net_params
from tqdm import tqdm
#import shutil

pjoin = os.path.join


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='lol')
    
    parser.add_argument(
        '--recursive_scan', '-r', action='store_true',
        help='Recursively scan through all the subdirectories.')
    parser.add_argument(
        '--network_name', '-n', type=str, default='resnet_v1_50',
        help='The name of the CNN used for duplicate detection.')
    parser.add_argument(
        '--difference_threshold', '-t', type=int, default=10,
        help='Any image pair with smaller difference than threshold '
                'will be considered as duplicates. '
                'Can be any value between (and including) 1 to 99.')
    parser.add_argument(
        '--image_size', '-s', type=int, default=0,
        help='The size of the resized input image to the CNN.')
    parser.add_argument(
        '--cache_resized_img', '-c', action='store_true',
        help='Cache the resized images. Speeds up subsequent scans.')
    parser.add_argument(
        '--gpu_id', '-g', type=int, default=0,
        help='The GPU ID that TensorFlow should use to run the network.')
    
    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    
    net_name = args.network_name
    in_size = args.image_size
    emb_dtype = np.float16
    
    recursive_list = args.recursive_scan
    cache_resized_img = args.cache_resized_img
    gpu_id = str(args.gpu_id)
    print('INFO: Using GPU #{}.'.format(gpu_id))
    
    dist_threshold = args.difference_threshold
    if dist_threshold < 1 or dist_threshold > 100:
        print('INFO: Using the default difference threshold of `10`.')
        dist_threshold = 10
    dist_threshold = float(dist_threshold) / 1000
    
    dir_file = pjoin(os.path.dirname(__file__), 'directory_file.txt')
    if os.path.isfile(dir_file):
        with open(dir_file, 'r') as ff:
            dirs = [l.strip() for l in ff.readlines()]
    else:
        raise ValueError(
            '`directory_file.txt` file missing. '
            'It should contain the list of directories to scan.')
    
    # Maybe download checkpoint file
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    network = net_params.get_net_params(net_name)
    
    if in_size < network['default_input_size'] or in_size > 1024:
        print('INFO: Using the default input size for `{}`.'.format(
                                                            network['name']))
        in_size = network['default_input_size']
    
    cname = '{}_@_{}_@_{}'.format(network['name'], network['end_point'], in_size)
    print('INFO: Using `{}` for duplicate detection.'.format(cname))
    time.sleep(0.1)
    utils.maybe_get_ckpt_file(network)
    
    # Iterate through directories
    img_paths = utils.list_files(dirs, recursive_list)
    print('INFO: Found {:,d} images.'.format(len(img_paths)))
    time.sleep(0.2)
    
    # Read and resize the images
    imgs = [utils.read_resize(f, in_size, cache_resized_img) for f in tqdm(img_paths)]
    
    # Run the CNN to retrieve the embeddings
    embeds = utils.get_embeds(network, imgs)
    
    # Compare
    dup_fnames = utils.get_duplicates(embeds, img_paths, dist_threshold, emb_dtype)
    
    print('\n\nResults:\n')
    for k, v in dup_fnames.iteritems():
        for img in v:
            try:
                #shutil.move(pjoin(gallery_dir, img), pjoin('/mnt/8TBHDD/flickr/curved_text/duplicate', img))
                print(k, img)
            except IOError:
                print('error {} {}'.format(k, img))
    print('\nINFO: Duplicate detection completed.')
    print('============================================================\n')
    
