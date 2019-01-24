# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:21:36 2019

@author: jiahuei

Network parameters, preprocessing functions, etc.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from nets import nets_factory


_VGG_MEANS = np.array([[[123.68, 116.78, 103.94]]]).astype(np.float32)
pjoin = os.path.join


def _vgg_prepro(image):
    image = tf.to_float(image) - _VGG_MEANS
    return image


def _inception_prepro(image):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


# all ResNet and VGG variants
resnet_params = dict(
                    end_point = 'global_pool',
                    prepro_fn = _vgg_prepro,
                    num_classes = None,
                    default_input_size = 224)

# Inception V1 and V2
inception_params_a = dict(
                    end_point = 'global_pool',
                    prepro_fn = _inception_prepro,
                    num_classes = None,
                    default_input_size = 224)

# Inception V3 and V4, Inception-ResNet-V2
inception_params_b = dict(
                    end_point = 'global_pool',
                    prepro_fn = _inception_prepro,
                    num_classes = None,
                    default_input_size = 299)

all_net_params = dict(
                vgg_16 = dict(
                    name = 'vgg_16',
                    ckpt_path = 'vgg_16.ckpt',
                    url = 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
                    **resnet_params),
                resnet_v1_50 = dict(
                    name = 'resnet_v1_50',
                    ckpt_path = 'resnet_v1_50.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
                    **resnet_params),
                resnet_v1_101 = dict(
                    name = 'resnet_v1_101',
                    ckpt_path = 'resnet_v1_101.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
                    **resnet_params),
                resnet_v1_152 = dict(
                    name = 'resnet_v1_152',
                    ckpt_path = 'resnet_v1_152.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',
                    **resnet_params),
                resnet_v2_50 = dict(
                    name = 'resnet_v2_50',
                    ckpt_path = 'resnet_v2_50.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
                    **resnet_params),
                resnet_v2_101 = dict(
                    name = 'resnet_v2_101',
                    ckpt_path = 'resnet_v2_101.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
                    **resnet_params),
                resnet_v2_152 = dict(
                    name = 'resnet_v2_152',
                    ckpt_path = 'resnet_v2_152.ckpt',
                    url = 'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
                    **resnet_params),
                inception_v1 = dict(
                    name = 'inception_v1',
                    ckpt_path = 'inception_v1.ckpt',
                    url = 'http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz',
                    **inception_params_a),
                inception_v2 = dict(
                    name = 'inception_v2',
                    ckpt_path = 'inception_v2.ckpt',
                    url = 'http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz',
                    **inception_params_a),
                inception_v3 = dict(
                    name = 'inception_v3',
                    ckpt_path = 'inception_v3.ckpt',
                    url = 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz',
                    **inception_params_b),
                inception_v4 = dict(
                    name = 'inception_v4',
                    ckpt_path = 'inception_v4.ckpt',
                    url = 'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
                    **inception_params_b),
                inception_resnet_v2 = dict(
                    name = 'inception_resnet_v2',
                    ckpt_path = 'inception_resnet_v2_2016_08_30.ckpt',
                    url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz',
                    **inception_params_b),
                )


def get_net_params(net_name):
    net_params = all_net_params[net_name]
    cnn_fn = nets_factory.get_network_fn(
                        net_params['name'],
                        num_classes=net_params['num_classes'],
                        weight_decay=0.0,
                        is_training=False)
    net_params['cnn_fn'] = cnn_fn
    curr_dir = os.path.dirname(__file__)
    net_params['ckpt_path'] = pjoin(curr_dir, 'ckpt', net_params['ckpt_path'])
    return net_params



