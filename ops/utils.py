# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:38:59 2019

@author: jiahuei

Utility functions.

"""
import os
import re
import numpy as np
from tqdm import tqdm
from PIL import Image
from ops.pretrained_cnn import CNNModel

try:
    from natsort import natsorted, ns
except ImportError:
    natsorted = None
# Image.MAX_IMAGE_PIXELS = None
# By default, PIL limit is around 89 Mpix (~ 9459 ** 2)


pjoin = os.path.join
SKIP = ['/', '\\', 'C:\\']
IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.gif']


def _atoi(text):
    return int(text) if text.isdigit() else text


def _natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [_atoi(c) for c in re.split(r'(\d+)', text)]


def _sort(l):
    if natsorted is not None:
        return natsorted(l, alg=ns.IGNORECASE)
    else:
        return sorted(l, key=_natural_keys)


def _skip(d):
    if d in SKIP or not os.path.isdir(d) or '$' in d:
        print('INFO: Directory `{}` skipped.'.format(d))
        return True
    return False


def get_date_taken(path):
    try:
        # noinspection PyProtectedMember
        return Image.open(path)._getexif()[36867]  # DateTimeOriginal
    except (AttributeError, KeyError, TypeError):
        return '0'
    except IOError:
        return None


def list_files(dirs, out_dir=None, out_filename=None, verbose=True):
    if not isinstance(dirs, list):
        raise ValueError('`dirs` must be a list.')
    file_paths = []
    for rootdir in dirs:
        if _skip(rootdir):
            continue
        # for root, subdirs, files in os.walk(unicode(rootdir, 'utf-8')):
        for root, subdirs, files in os.walk(rootdir):
            if _skip(root):
                continue
            if verbose:
                print('INFO: Reading directory: {}'.format(root))
            for f in _sort(files):
                fpath = pjoin(root, f)
                fext = os.path.splitext(fpath)[1]
                if fext.lower() not in IMAGE_EXT:
                    continue
                file_paths.append(fpath)
    return file_paths


def read_resize(img_path, short_side_len, save_resized_imgs=True):
    """
    # Check if the resized images already exists
    # If yes, read and avoid resizing operation
    """
    img_dir, img_name = os.path.split(img_path)
    img_dir_new = img_dir + '_resized_{}'.format(short_side_len)
    img_name, _ = os.path.splitext(img_name)
    img_save_name = pjoin(img_dir_new, img_name + '.jpg')
    
    resize_img_exists = os.path.isfile(img_save_name)
    if resize_img_exists:
        img = Image.open(img_save_name)
    else:
        img = Image.open(img_path)
        ratio = short_side_len / min(img.size)
        new_size = [int(img.size[i] * ratio) for i in range(2)]
        img = img.resize(new_size, Image.BILINEAR)
    
    def _try_convert(image):
        try:
            image = image.convert(mode='RGB')
            return np.array(image)
        except:
            raise ValueError(err_mssg.format(img_path))
    
    img_arr = np.array(img)
    err_mssg = 'Corrupted or unsupported image file: `{}`.'
    if len(img_arr.shape) == 4:
        if img.mode == 'RGBA':
            png = img.copy()
            png.load()  # required for png.split()
            img = Image.new("RGB", png.size, (255, 255, 255))
            img.paste(png, mask=png.split()[3])
            img_arr = np.array(img)
        else:
            img_arr = _try_convert(img)
    elif len(img_arr.shape) == 3:
        if img_arr.shape[-1] == 3:
            pass
        elif img_arr.shape[-1] == 1:
            img_arr = np.concatenate([img_arr] * 3, axis=2)
        else:
            img_arr = _try_convert(img)
    elif len(img_arr.shape) == 2:
        img_arr = np.stack([img_arr] * 3, axis=2)
    else:
        img_arr = _try_convert(img)
    if save_resized_imgs and not resize_img_exists:
        if not os.path.exists(img_dir_new):
            os.makedirs(img_dir_new)
        img.save(img_save_name)
    return img_arr


def _shape(tensor):
    return list(tensor.shape)


def get_embeds(cnn_name, imgs):
    """
    """
    print('INFO: Building TensorFlow graph.')
    
    # Setup input pipeline & Build model
    model = CNNModel(cnn_name,
                     cnn_feat_map_name=None,
                     include_top=False,
                     trainable=False,
                     batch_norm_momentum=0.997,
                     input_shape=None,
                     pooling='avg',
                     cnn_kwargs=None,
                     layer_kwargs=None,
                     name='CNNModel')
    # Get embeddings
    embeddings = []
    for img in tqdm(imgs, desc='Running CNN'):
        img = np.expand_dims(img, axis=0)
        embeddings.append(np.squeeze(model(img)))
    embeddings = np.stack(embeddings, axis=0)
    return embeddings


def save_embeds(save_dir, embeds, net_params, in_size):
    suffix = 'embeds_@_{}_@_{}_@_{}'.format(
        net_params['name'], net_params['end_point'], in_size)
    np.save(pjoin(save_dir, suffix), embeds)
    print('INFO: Embeddings saved to disk.')


def get_duplicates(embeds, fnames, dist_threshold, dtype=np.float16):
    # fnames = [os.path.basename(f) for f in fnames]
    num_imgs = embeds.shape[0]
    embeds = embeds.astype(dtype)
    dup_fnames = {}
    
    # If the number of imgs is large, then use loop to conserve memory
    if embeds.nbytes * num_imgs > 2 * (1024 ** 3):
        mask = np.ones(shape=[num_imgs], dtype=dtype)
        for i in tqdm(range(num_imgs), desc='Comparing images'):
            mask[i] = 0
            diff = embeds[i, :] - embeds
            dist = np.sqrt(np.sum(np.square(diff), axis=1))
            duplicates = dist < dist_threshold
            duplicates = duplicates.astype(dtype) * mask
            dups = np.nonzero(duplicates)[0]
            if len(dups) == 0:
                continue
            else:
                dup_fnames[fnames[i]] = [fnames[d] for d in dups]
    else:
        prb_embeds = np.expand_dims(embeds, axis=1)
        gal_embeds = np.expand_dims(embeds, axis=0)
        diff = prb_embeds - gal_embeds
        diff = diff.astype(dtype)
        np.square(diff, out=diff)
        dist = np.sqrt(np.sum(diff, axis=2))
        mask = np.tril(np.ones_like(dist), k=0)
        dist += mask
        duplicates = dist < dist_threshold
        for i in range(num_imgs):
            dups = np.nonzero(duplicates[i, :])[0]
            if len(dups) == 0:
                continue
            else:
                dup_fnames[fnames[i]] = [fnames[d] for d in dups]
    return dup_fnames
