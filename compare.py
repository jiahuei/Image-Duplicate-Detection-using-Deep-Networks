#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:56:21 2018

@author: jiahuei
"""
import os, time, argparse, numpy as np
import logging
from ops import utils
from tqdm import tqdm
from ops.pretrained_cnn import get_cnn_default_input_size

pjoin = os.path.join
info = logging.info


def create_parser():
    _parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='A parser')
    
    _parser.add_argument(
        '--recursive_scan', '-r', action='store_true',
        help='Recursively scan through all the subdirectories.')
    _parser.add_argument(
        '--cnn_name', '-n', type=str, default='resnet_v1_50',
        help='The name of the CNN used for duplicate detection.')
    _parser.add_argument(
        '--difference_threshold', '-t', type=int, default=10,
        help='Any image pair with smaller difference than threshold will be considered as duplicates. '
             'Can be any value between (and including) 1 to 99.')
    _parser.add_argument(
        '--short_side_len', '-s', type=int, default=0,
        help='The short-side size of the input image.')
    _parser.add_argument(
        '--cache_resized_img', '-c', action='store_true',
        help='Cache the resized images. Speeds up subsequent scans.')
    _parser.add_argument(
        '--gpu_id', '-g', type=int, default=0,
        help='The GPU ID that TensorFlow should use to run the network.')
    
    return _parser


if __name__ == '__main__':
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate the choice of CNN
    def_input_size = get_cnn_default_input_size(cnn_name=args.cnn_name, is_training=False)
    
    recursive_list = args.recursive_scan
    cache_resized_img = args.cache_resized_img
    
    dist_threshold = args.difference_threshold
    if dist_threshold < 1 or dist_threshold > 100:
        info('Using the default difference threshold of `10`.')
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
    
    # Get CNN arguments
    short_side_len = args.short_side_len
    cnn_name = args.cnn_name
    print('Using `{}` for duplicate detection.'.format(cnn_name))
    if short_side_len < def_input_size[0] or short_side_len > 1024:
        short_side_len = def_input_size[0]
    info('Input size used: {}'.format(short_side_len))
    
    gpu_id = str(args.gpu_id)
    info('Using GPU #{}.'.format(gpu_id))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    # Iterate through directories
    img_paths = utils.list_files(dirs, recursive_list)
    info('Found {:,d} images.'.format(len(img_paths)))
    time.sleep(0.2)
    
    # Read and resize the images
    imgs = [utils.read_resize(f, short_side_len, cache_resized_img) for f in tqdm(img_paths)]
    
    # Run the CNN to retrieve the embeddings
    embeds = utils.get_embeds(cnn_name, imgs)
    
    # Compare
    dup_fnames = utils.get_duplicates(embeds, img_paths, dist_threshold, np.float16)
    
    print('\n\nResults:\n')
    for k, v in dup_fnames.iteritems():
        for img in v:
            try:
                # shutil.move(pjoin(gallery_dir, img), pjoin('/mnt/8TBHDD/flickr/curved_text/duplicate', img))
                print(k, img)
            except IOError:
                print('error {} {}'.format(k, img))
    print('\nINFO: Duplicate detection completed.')
    print('============================================================\n')
