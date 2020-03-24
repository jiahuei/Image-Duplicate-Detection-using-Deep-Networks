# -*- coding: utf-8 -*-
"""
Created on 19 Dec 2019 23:57:54

@author: jiahuei

`TFHUB_CNN_LIST` last updated on 6 Feb 2020,
selected from page source using regex `imagenet\​\/[\S]+\​\/feature_vector`

Old TF Slim model zoo page:
https://github.com/tensorflow/models/tree/master/research/slim

For MobileNetV2:
    The values of `layer_i` is equal to `layer_i/output`.

    tfhub_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4'
    hub_args = dict(return_endpoints=True)
    layer = tfhub.KerasLayer(tfhub_url, trainable=False, arguments=hub_args)
    cnn_outputs = layer(np.random.normal(size=[1, 224, 224, 3]).astype(np.float32))
    out = []
    for i in range(2, 19):
        k = 'MobilenetV2/layer_{}'.format(i)
        out.append(
            (cnn_outputs[k].numpy(), cnn_outputs[k + '/output'].numpy())
            )
    print(all([np.all(x == y) for x, y in out]))
    > True
"""
import logging
import tensorflow as tf
import tensorflow_hub as tfhub
# from ops import models_v0 as mops
from ops import keras_app as kapp

assert tf.version.VERSION.startswith('2.')

imagenet_preprocess_input = kapp.imagenet_utils.preprocess_input
TFHUB_CNN_LIST_TF1 = ['amoebanet_a_n18_f448',
                      'efficientnet_b0',
                      'efficientnet_b1',
                      'efficientnet_b2',
                      'efficientnet_b3',
                      'efficientnet_b4',
                      'efficientnet_b5',
                      'efficientnet_b6',
                      'efficientnet_b7',
                      ]
TFHUB_CNN_LIST_TF2 = ['inception_v1',
                      'inception_v2',
                      'inception_v3',
                      'resnet_v1_50',
                      'resnet_v1_101',
                      'resnet_v1_152',
                      'resnet_v2_50',
                      'resnet_v2_101',
                      'resnet_v2_152',
                      'inception_resnet_v2',
                      'nasnet_large',
                      'nasnet_mobile',
                      'pnasnet_large',
                      'mobilenet_v1_025_128',
                      'mobilenet_v1_025_160',
                      'mobilenet_v1_025_192',
                      'mobilenet_v1_025_224',
                      'mobilenet_v1_050_128',
                      'mobilenet_v1_050_160',
                      'mobilenet_v1_050_192',
                      'mobilenet_v1_050_224',
                      'mobilenet_v1_075_128',
                      'mobilenet_v1_075_160',
                      'mobilenet_v1_075_192',
                      'mobilenet_v1_075_224',
                      'mobilenet_v1_100_128',
                      'mobilenet_v1_100_160',
                      'mobilenet_v1_100_192',
                      'mobilenet_v1_100_224',
                      'mobilenet_v2_035_96',
                      'mobilenet_v2_035_128',
                      'mobilenet_v2_035_160',
                      'mobilenet_v2_035_192',
                      'mobilenet_v2_035_224',
                      'mobilenet_v2_050_96',
                      'mobilenet_v2_050_128',
                      'mobilenet_v2_050_160',
                      'mobilenet_v2_050_192',
                      'mobilenet_v2_050_224',
                      'mobilenet_v2_075_96',
                      'mobilenet_v2_075_128',
                      'mobilenet_v2_075_160',
                      'mobilenet_v2_075_192',
                      'mobilenet_v2_075_224',
                      'mobilenet_v2_100_96',
                      'mobilenet_v2_100_128',
                      'mobilenet_v2_100_160',
                      'mobilenet_v2_100_192',
                      'mobilenet_v2_100_224',
                      'mobilenet_v2_130_224',
                      'mobilenet_v2_140_224',
                      ]
KERAS_CNN_LIST = ['vgg_16',
                  'vgg_19',
                  'densenet_121',
                  'densenet_169',
                  'densenet_201',
                  'nasnet_large',
                  'nasnet_mobile',
                  'efficientnet_b0',
                  'efficientnet_b1',
                  'efficientnet_b2',
                  'efficientnet_b3',
                  'efficientnet_b4',
                  'efficientnet_b5',
                  'efficientnet_b6',
                  'efficientnet_b7',
                  'mobilenet_v1_025_128',
                  'mobilenet_v1_025_160',
                  'mobilenet_v1_025_192',
                  'mobilenet_v1_025_224',
                  'mobilenet_v1_050_128',
                  'mobilenet_v1_050_160',
                  'mobilenet_v1_050_192',
                  'mobilenet_v1_050_224',
                  'mobilenet_v1_075_128',
                  'mobilenet_v1_075_160',
                  'mobilenet_v1_075_192',
                  'mobilenet_v1_075_224',
                  'mobilenet_v1_100_128',
                  'mobilenet_v1_100_160',
                  'mobilenet_v1_100_192',
                  'mobilenet_v1_100_224',
                  'mobilenet_v2_035_96',
                  'mobilenet_v2_035_128',
                  'mobilenet_v2_035_160',
                  'mobilenet_v2_035_192',
                  'mobilenet_v2_035_224',
                  'mobilenet_v2_050_96',
                  'mobilenet_v2_050_128',
                  'mobilenet_v2_050_160',
                  'mobilenet_v2_050_192',
                  'mobilenet_v2_050_224',
                  'mobilenet_v2_075_96',
                  'mobilenet_v2_075_128',
                  'mobilenet_v2_075_160',
                  'mobilenet_v2_075_192',
                  'mobilenet_v2_075_224',
                  'mobilenet_v2_100_96',
                  'mobilenet_v2_100_128',
                  'mobilenet_v2_100_160',
                  'mobilenet_v2_100_192',
                  'mobilenet_v2_100_224',
                  'mobilenet_v2_130_224',
                  'mobilenet_v2_140_224',
                  ]


# TODO: enable BN freeze
class PretrainedCNN(tf.keras.layers.Layer):
    def __init__(self,
                 cnn_name,
                 cnn_feat_map_name=None,
                 include_top=False,
                 trainable=False,
                 name='PretrainedCNN',
                 **kwargs):
        """

        :param cnn_name: Name of the CNN to be built.
        :param cnn_feat_map_name: Either `a list of strings`, or `None`, or `all`.
            If `None`, will return feature vector + empty dict.
            If `all`, will return feature vector + all endpoints as dict.
            If `a list of strings`, will return feature vector + specified endpoints as dict.
        :param include_top: Whether to include the fully-connected layer at the top of the network.
        :param trainable: If True, will update batch norm variables, and mark weights as trainable.
        :param name: Name scope of this module.
        """
        super().__init__(name=name, **kwargs)
        self.cnn_name = cnn_name
        self.cnn_feat_map_name = cnn_feat_map_name
        if not isinstance(cnn_feat_map_name, list):
            assert cnn_feat_map_name in [None, 'all'], \
                '`cnn_feat_map_name` must be `a list of strings` or `None` or `all`.'
        self.return_all_maps = isinstance(self.cnn_feat_map_name, str) and self.cnn_feat_map_name.lower() == 'all'
        self.include_top = include_top
        self.trainable = trainable
        assert isinstance(trainable, bool)
        self.cnn_layer = self.get_cnn_layer()
    
    def forward(self, images, training):
        assert not isinstance(images, list)
        return self.__call__(inputs=[images], training=training)
    
    @staticmethod
    def get_cnn_feat_map(cnn_outputs, feat_map_name):
        if feat_map_name not in cnn_outputs:
            _err = '\n'.join(['{} --- {}'.format(k, v.shape.as_list())
                              for k, v in sorted(cnn_outputs.items())])
            _err = 'Invalid feature map name: `{}`. Available choices: \n{}'.format(
                feat_map_name, _err)
            raise ValueError(_err)
        return cnn_outputs[feat_map_name]


class TFHubCNN(PretrainedCNN):
    """
    self.cnn_layer is a KerasLayer and is different from a standard Keras layer. It has no `layers` attribute.
    """
    
    def __init__(self,
                 cnn_name,
                 cnn_feat_map_name=None,
                 include_top=False,
                 trainable=False,
                 batch_norm_momentum=0.997,
                 cnn_kwargs=None,
                 layer_kwargs=None,
                 name='TFHubCNN'):
        self.batch_norm_momentum = batch_norm_momentum
        self.cnn_kwargs = {} if cnn_kwargs is None else cnn_kwargs
        kwargs = {} if layer_kwargs is None else layer_kwargs
        super().__init__(
            cnn_name=cnn_name,
            cnn_feat_map_name=cnn_feat_map_name,
            include_top=include_top,
            trainable=trainable,
            name=name,
            **kwargs)
    
    # noinspection PyMethodOverriding
    def call(self, inputs, training):
        assert isinstance(inputs, list)  # Standardise across `call` methods
        images = inputs[0]
        
        # Some CNNs cannot accept non-default image size
        # Using parentheses to break expressions on multiple lines, as alternative to backslash
        # https://github.com/tensorflow/tensorflow/issues/33962#issuecomment-575804633
        if (self.cnn_name.endswith('nasnet_large')
                or self.cnn_name.startswith('mobilenet_v2')
                or self.cnn_name == 'nasnet_mobile'):
            assert images.shape[1:3] == get_cnn_default_input_size(self.cnn_name)
        
        images = self.preprocess_inputs(images)
        cnn_outputs = self.cnn_layer(images, training)
        preds = cnn_outputs['default']
        
        # Some feature map name processing
        def _process_key(k):
            name_parts = k.split('/')
            if len(name_parts) > 1:
                return '/'.join(name_parts[1:])
            else:
                return k
        
        cnn_outputs = {_process_key(k): v for k, v in cnn_outputs.items()}
        final_outputs = {}
        
        # Standardise output: Keras models only predict 1000 classes
        # This also enables the use of `decode_predictions`, like so:
        # `tf.keras.applications.imagenet_utils.decode_predictions(probabilities.numpy())`
        # Class-0 is background class
        if self.include_top:
            if preds.shape[-1] == 1001:
                preds = preds[:, 1:]
            if self.return_all_maps:
                final_outputs['logits'] = preds
            preds = tf.nn.softmax(preds)
        
        if self.cnn_feat_map_name is None:
            pass
        elif self.return_all_maps:
            cnn_outputs.update(final_outputs)
            final_outputs.update(cnn_outputs)
        else:
            assert isinstance(self.cnn_feat_map_name, list)
            for n in self.cnn_feat_map_name:
                final_outputs[n] = self.get_cnn_feat_map(cnn_outputs=cnn_outputs, feat_map_name=n)
        return preds, final_outputs
    
    @staticmethod
    def preprocess_inputs(images):
        """
        Images are expected to have values in the range [0, 255].
        :return:
        """
        # Cast to float
        if images.dtype != tf.dtypes.float32:
            images = tf.cast(images, tf.dtypes.float32)
        return tf.math.divide(images, 255.)
    
    def get_cnn_layer(self):
        """
        :return: TF Hub CNN Keras layer.
        """
        # assert self.cnn_name in TFHUB_CNN_LIST_TF1 + TFHUB_CNN_LIST_TF2
        assert self.cnn_name in TFHUB_CNN_LIST_TF2, '`{}` not available as TF-Hub layer.'.format(self.cnn_name)
        if self.include_top:
            mode = 'classification'
        else:
            mode = 'feature_vector'
        hub_url_map = dict(
            amoebanet_a_n18_f448='https://tfhub.dev/google/imagenet/amoebanet_a_n18_f448/{}/1'.format(mode)
        )
        cnn_name_or_url = self.cnn_name
        if cnn_name_or_url.startswith(('inception_v',
                                       'resnet_v',
                                       'inception_resnet_v',
                                       'mobilenet_v',
                                       'nasnet',
                                       'pnasnet')):
            tfhub_url = 'https://tfhub.dev/google/imagenet/{}/{}/4'.format(cnn_name_or_url, mode)
        elif cnn_name_or_url.startswith('efficientnet_b'):
            eff_type = cnn_name_or_url.split('_b')[1]
            if self.include_top:
                tfhub_url = 'https://tfhub.dev/google/efficientnet/b{}/classification/1'.format(eff_type)
            else:
                tfhub_url = 'https://tfhub.dev/google/efficientnet/b{}/feature-vector/1'.format(eff_type)
        elif cnn_name_or_url in hub_url_map:
            tfhub_url = hub_url_map[cnn_name_or_url]
        else:
            tfhub_url = cnn_name_or_url
        _arguments = dict(batch_norm_momentum=self.batch_norm_momentum, return_endpoints=True)
        self.cnn_kwargs.update(dict(trainable=self.trainable, arguments=_arguments))
        layer = tfhub.KerasLayer(tfhub_url, **self.cnn_kwargs)
        return layer


class KerasCNN(PretrainedCNN):
    """
    self.cnn_layer is a standard Keras layer. It has `layers` attribute.
    """
    
    def __init__(self,
                 cnn_name,
                 cnn_feat_map_name=None,
                 input_shape=(None, None, 3),
                 include_top=False,
                 trainable=False,
                 pooling='avg',
                 cnn_kwargs=None,
                 layer_kwargs=None,
                 name='KerasCNN'):
        self.cnn_input_shape = input_shape
        self.pooling = pooling
        self.cnn_kwargs = {} if cnn_kwargs is None else cnn_kwargs
        kwargs = {} if layer_kwargs is None else layer_kwargs
        super().__init__(
            cnn_name=cnn_name,
            cnn_feat_map_name=cnn_feat_map_name,
            include_top=include_top,
            trainable=trainable,
            name=name,
            **kwargs)
        # Build keras model
        inputs = self.cnn_layer.input
        preds = self.cnn_layer.layers[-1].output
        if self.cnn_feat_map_name is None:
            self.model_layer = tf.keras.Model(inputs=inputs, outputs=(preds, {}))
        else:
            all_layers = {l.name: l.output for l in self.cnn_layer.layers}
            if self.return_all_maps:
                logging.warning('Retrieving a large number of layers: {}'.format(len(all_layers)))
                self.model_layer = tf.keras.Model(inputs=inputs, outputs=(preds, all_layers))
            else:
                assert isinstance(self.cnn_feat_map_name, list)
                cnn_outputs = {}
                for n in self.cnn_feat_map_name:
                    cnn_outputs[n] = self.get_cnn_feat_map(cnn_outputs=all_layers, feat_map_name=n)
                self.model_layer = tf.keras.Model(inputs=inputs, outputs=(preds, cnn_outputs))
    
    # noinspection PyMethodOverriding
    def call(self, inputs, training):
        assert isinstance(inputs, list)  # Standardise across `call` methods
        images = inputs[0]
        images = self.preprocess_inputs(images)
        outputs = self.model_layer(images, training)
        return outputs
    
    def preprocess_inputs(self, images):
        """
        Images are expected to have values in the range [0, 255].
        :return:
        """
        cnn_name = self.cnn_name
        
        def _get_prepro_fn(mode):
            return lambda x, data_format=None, **kwargs: imagenet_preprocess_input(
                x, data_format, mode=mode, **kwargs)
        
        if cnn_name.startswith('vgg'):
            # preprocess_fn = tf.keras.applications.vgg16.preprocess_input  # actually uses imagenet_utils
            preprocess_fn = imagenet_preprocess_input
        elif cnn_name.startswith(('densenet', 'efficientnet_b')):
            # preprocess_fn = tf.keras.applications.densenet.preprocess_input
            preprocess_fn = _get_prepro_fn(mode='torch')
        elif cnn_name.startswith(('nasnet', 'mobilenet')):
            # preprocess_fn = tf.keras.applications.nasnet.preprocess_input
            preprocess_fn = _get_prepro_fn(mode='tf')
        else:
            raise ValueError('Invalid CNN choice: `{}`'.format(cnn_name))
        # Cast to float
        if images.dtype != tf.dtypes.float32:
            images = tf.cast(images, tf.dtypes.float32)
        return preprocess_fn(images)
    
    def get_cnn_layer(self):
        """
        :return: TF Hub CNN Keras layer.
        """
        assert self.cnn_name in KERAS_CNN_LIST, '`{}` not available as Keras layer.'.format(self.cnn_name)
        if self.cnn_name.startswith('mobilenet'):
            # mobilenet_v1_075_128, mobilenet_v2_130_224
            _net_type = self.cnn_name.split('_')
            assert len(_net_type) == 4
            if _net_type[1] == 'v1':
                # cnn_fn = tf.keras.applications.MobileNet
                cnn_fn = kapp.masked_mobilenet.MobileNet
            else:
                cnn_fn = tf.keras.applications.MobileNetV2
            self.cnn_kwargs.update(dict(alpha=int(_net_type[2]) / 100.))
            input_size = (int(_net_type[3]), int(_net_type[3]), 3)
            _input_shape = tuple(self.cnn_input_shape)
            if _input_shape != input_size and _input_shape != (None, None, 3):
                logging.warning('Overriding `input_shape` from `{}` to `{}`'.format(_input_shape, input_size))
                self.cnn_input_shape = input_size
        else:
            name_to_fn = dict(
                vgg_16=tf.keras.applications.VGG16,
                vgg_19=tf.keras.applications.VGG19,
                densenet_121=tf.keras.applications.DenseNet121,
                densenet_169=tf.keras.applications.DenseNet169,
                densenet_201=tf.keras.applications.DenseNet201,
                nasnet_large=tf.keras.applications.NASNetLarge,
                nasnet_mobile=tf.keras.applications.NASNetMobile,
                efficientnet_b0=kapp.efficientnet.EfficientNetB0,
                efficientnet_b1=kapp.efficientnet.EfficientNetB1,
                efficientnet_b2=kapp.efficientnet.EfficientNetB2,
                efficientnet_b3=kapp.efficientnet.EfficientNetB3,
                efficientnet_b4=kapp.efficientnet.EfficientNetB4,
                efficientnet_b5=kapp.efficientnet.EfficientNetB5,
                efficientnet_b6=kapp.efficientnet.EfficientNetB6,
                efficientnet_b7=kapp.efficientnet.EfficientNetB7,
            )
            cnn_fn = name_to_fn[self.cnn_name]
        layer = cnn_fn(
            input_shape=self.cnn_input_shape,
            include_top=self.include_top,
            pooling=self.pooling,
            weights='imagenet',
            **self.cnn_kwargs)
        layer.trainable = self.trainable
        return layer


class CNNModel(tf.keras.layers.Layer):
    def __init__(self,
                 cnn_name,
                 cnn_feat_map_name,
                 include_top,
                 trainable,
                 batch_norm_momentum=0.997,
                 input_shape=None,
                 pooling='avg',
                 cnn_kwargs=None,
                 layer_kwargs=None,
                 name='CNNModel'):
        super().__init__(name=name)
        _kwargs = dict(
            cnn_name=cnn_name,
            cnn_feat_map_name=cnn_feat_map_name,
            include_top=include_top,
            trainable=trainable,
            cnn_kwargs=cnn_kwargs,
            layer_kwargs=layer_kwargs)
        if cnn_name in TFHUB_CNN_LIST_TF2:
            self.cnn = TFHubCNN(batch_norm_momentum=batch_norm_momentum, **_kwargs)
            self.layer_type = 'tf_hub'
        else:
            self.cnn = KerasCNN(input_shape=input_shape, pooling=pooling, **_kwargs)
            self.layer_type = 'tf_keras'
        self.cnn_name = cnn_name
    
    # noinspection PyMethodOverriding
    def call(self, inputs, training):
        return self.cnn.__call__(inputs, training)
    
    def preprocess_inputs(self, images):
        return self.cnn.preprocess_inputs(images)


def get_cnn_default_input_size(cnn_name, is_training=True):
    assert isinstance(cnn_name, str)
    if cnn_name in ['lenet']:
        return 28, 28, 1
    elif cnn_name.startswith('mobilenet'):
        size = int(cnn_name.split('_')[-1])
    elif cnn_name.startswith('resnet_v1'):
        if is_training:
            size = 224
        else:
            size = 299
    elif cnn_name.startswith('resnet_v2'):
        # https://github.com/tensorflow/models/tree/v1.12.0/research/slim
        # https://github.com/keras-team/keras-applications/blob/976050c468ff949bcbd9b9cf64fe1d5c81db3f3a/README.md
        size = 299
    elif cnn_name.startswith(('vgg', 'densenet')):
        size = 224
    elif cnn_name in ['inception_v1',
                      'inception_v2',
                      'nasnet_mobile',
                      'pnasnet_mobile']:
        size = 224
    elif cnn_name in ['overfeat']:
        size = 231
    elif cnn_name in ['inception_resnet_v2', 'inception_v3', 'inception_v4']:
        size = 299
    elif cnn_name in ['nasnet_large', 'pnasnet_large', 'amoebanet_a_n18_f448']:
        size = 331
    elif cnn_name.startswith('efficientnet_b'):
        eff_type = int(cnn_name.split('_b')[1])
        all_sizes = [224, 240, 260, 300, 380, 456, 528, 600]
        size = all_sizes[eff_type]
    elif cnn_name in ['cifarnet']:
        size = 32
    else:
        raise ValueError('Invalid CNN choice: `{}`'.format(cnn_name))
    return size, size, 3
