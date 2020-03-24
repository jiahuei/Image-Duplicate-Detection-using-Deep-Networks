"""
Based on this commit: https://github.com/keras-team/keras-applications/tree/71acdcd98088501247f4b514b7cbbdf8182a05a4

Named `keras_app` to avoid conflict with `keras_applications`.
# tensorflow_core/python/keras/applications/__init__.py", line 22, in <module>
#     import keras_applications

Enables dynamic setting of underlying Keras module.
"""

__version__ = '1.0.8'

import tensorflow as tf

_KERAS_BACKEND = tf.keras.backend
_KERAS_LAYERS = tf.keras.layers
_KERAS_MODELS = tf.keras.models
_KERAS_UTILS = tf.keras.utils


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


# from ops.keras_app import resnext
from ops.keras_app import masked_mobilenet
from ops.keras_app import efficientnet
from ops.keras_app import imagenet_utils
