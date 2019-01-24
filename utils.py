# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:38:59 2019

@author: jiahuei

Utility functions.

"""
import os, math, time, re, requests, tarfile
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
#Image.MAX_IMAGE_PIXELS = None
# By default, PIL limit is around 89 Mpix (~ 9459 ** 2)


_EXT = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
pjoin = os.path.join

try:
    from natsort import natsorted, ns
except ImportError:
    natsorted = None


def _download_from_url(url):
    """
    Downloads file from URL, streaming large files.
    """
    #url = 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz'
    response = requests.get(url, stream=True)
    #chunk_size = 1024 * 512         # 512 kB
    chunk_size = 1024 ** 2          # 1 MB
    fname = url.split('/')[-1]
    curr_dir = os.path.dirname(__file__)
    if response.ok:
        print('INFO: Downloading `{}`'.format(fname))
    else:
        print('ERROR: Download error. Server response: {}'.format(response))
        return False
    time.sleep(0.2)
    
    if not os.path.exists(pjoin(curr_dir, 'ckpt')):
        os.makedirs(pjoin(curr_dir, 'ckpt'))
    tar_gz_path = pjoin(curr_dir, 'ckpt', fname)
    
    # Case-insensitive Dictionary of Response Headers.
    # The length of the request body in octets (8-bit bytes).
    file_size = int(response.headers['Content-Length'])
    num_iters = math.ceil(file_size / chunk_size)
    tqdm_kwargs = dict(desc = 'Download progress',
                       total = num_iters,
                       unit = 'MB')
    with open(tar_gz_path, 'wb') as handle:
        for chunk in tqdm(response.iter_content(chunk_size), **tqdm_kwargs):
            if not chunk: break
            handle.write(chunk)
    return tar_gz_path


def _extract_tar_gz(fpath):
    """
    Extracts the ckpt file from the tar.gz file into the containing directory.
    """
    tar = tarfile.open(fpath, 'r:gz')
    for m in tar.getmembers():
        if m.name.endswith('.ckpt'):
            break
    opath = os.path.split(fpath)[0]
    tar.extractall(path=opath, members=[m])   # members=None to extract all
    tar.close()


def maybe_get_ckpt_file(net_params, remove_tar=True):
    """
    Download, extract, remove.
    """
    if os.path.isfile(net_params['ckpt_path']):
        pass
    else:
        url = net_params['url']
        tar_gz_path = _download_from_url(url)
        if not tar_gz_path:
            _download_from_url(url)     # Retry one more time
        _extract_tar_gz(tar_gz_path)
        if remove_tar: os.remove(tar_gz_path)


def _atoi(text):
    return int(text) if text.isdigit() else text


def _natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [ _atoi(c) for c in re.split('(\d+)', text) ]


def list_files(dirs, recursive_list=False):
    def _sort(l):
        if natsorted:
            return natsorted(l, alg=ns.IGNORECASE)
        else:
            return sorted(l, key=_natural_keys)
    file_paths = []
    if recursive_list:
        for rootdir in dirs:
            for root, subdirs, files in os.walk(rootdir):
                print('INFO: Reading directory: {}'.format(root))
                file_paths += [pjoin(root, f) for f in  _sort(files)]
    else:
        for rootdir in dirs:
            print('INFO: Reading directory: {}'.format(rootdir))
            for f in _sort(os.listdir(rootdir)):
                file_paths.append(pjoin(rootdir, f))
    # Filter out the non-image files
    img_paths = []
    for f in file_paths:
        _, file_extension = os.path.splitext(f)
        if file_extension in _EXT:
            img_paths.append(f)
    print('INFO: Image file listing complete.')
    return img_paths


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
    
    img_arr = np.array(img)
    err_mssg = 'Corrupted or unsupported image file.'
    if len(img_arr.shape) == 3:
        if img_arr.shape[-1] == 3:
            pass
        elif img_arr.shape[-1] == 1:
            img_arr = np.concatenate([img_arr] * 3, axis=2)
        else:
            raise ValueError(err_mssg)
    elif len(img_arr.shape) == 2:
        img_arr = np.stack([img_arr] * 3, axis=2)
    else:
        raise ValueError(err_mssg)
    if save_resized_imgs and not resize_img_exists:
        if not os.path.exists(img_dir_new):
            os.makedirs(img_dir_new)
        img.save(img_save_name)
    return img_arr


def _shape(tensor):
    return tensor.get_shape().as_list()


def get_embeds(net_params, imgs):
    """
    """
    print('INFO: Building TensorFlow graph.')
	
    # Setup input pipeline & Build model
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(4896)
        
        cnn_fn = net_params['cnn_fn']
        image_prepro_fn = net_params['prepro_fn']
        
        image = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        images = tf.expand_dims(image_prepro_fn(image), axis=0)
        
        try:
            _, end_points = cnn_fn(images, global_pool=True)
        except:
            _, end_points = cnn_fn(images)
        #print(end_points.keys())
        embed = tf.squeeze(end_points[net_params['end_point']])
        print('INFO: Embedding has shape: {}'.format(_shape(embed)))
        embed = tf.nn.l2_normalize(embed, -1)
        
        #init_fn = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=None)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=g)
    
    with sess:
        # Restore whole model
        saver.restore(sess, net_params['ckpt_path'])
        g.finalize()
        print('')
        
        # Get embeddings
        embeddings = [sess.run(embed, {image: img}) for img in tqdm(
                                        imgs, desc='Running CNN')]
        embeddings = np.stack(embeddings, axis=0)
        sess.close()
    return embeddings


def save_embeds(save_dir, embeds, net_params, in_size):
    suffix = 'embeds_@_{}_@_{}_@_{}'.format(
                        net_params['name'], net_params['end_point'], in_size)
    np.save(pjoin(save_dir, suffix), embeds)
    print('INFO: Embeddings saved to disk.')


def get_duplicates(embeds, fnames, dist_threshold, dtype=np.float16):
    #fnames = [os.path.basename(f) for f in fnames]
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
        np.square(diff, out=diff)
        dist = np.sqrt(np.sum(diff, axis=2))
        mask = np.tril(np.ones_like(dist), k=0)
        duplicates = dist < dist_threshold
        duplicates = duplicates.astype(dtype) * mask
        for i in range(num_imgs):
            dups = np.nonzero(duplicates[i, :])[0]
            if len(dups) == 0:
                continue
            else:
                dup_fnames[fnames[i]] = [fnames[d] for d in dups]
    return dup_fnames





