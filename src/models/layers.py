#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope


def get_shape4D(in_val):
    """
    Return a 4D shape
    Args:
        in_val (int or list with length 2)
    Returns:
        list with length 4
    """
    # if isinstance(in_val, int):
    return [1] + get_shape2D(in_val) + [1]

def get_shape2D(in_val):
    """
    Return a 2D shape 
    Args:
        in_val (int or list with length 2) 
    Returns:
        list with length 2
    """
    # in_val = int(in_val)
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))

def deconv_size(input_height, input_width, stride=2):
    """
    Compute the feature size (height and width) after filtering with
    a specific stride. Mostly used for setting the shape for deconvolution.
    Args:
        input_height (int): height of input feature
        input_width (int): width of input feature
        stride (int): stride of the filter
    Return:
        (int, int): Height and width of feature after filtering.
    """
    return int(math.ceil(float(input_height) / float(stride))),\
           int(math.ceil(float(input_width) / float(stride)))

def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

def softplus(inputs, name='softplus'):
    return tf.log(1 + tf.exp(inputs), name=name)

def softmax(logits, axis=-1, name='softmax'):
    with tf.name_scope(name):
        max_in = tf.reduce_max(logits, axis=axis, keepdims=True)
        stable_in = logits - max_in
        normal_p = tf.reduce_sum(tf.exp(stable_in), axis=axis, keepdims=True)
 
        return tf.exp(stable_in) / normal_p

def leaky_relu(x, leak=0.3, name='LeakyRelu'):
    """ 
    leaky_relu 
        Allow a small non-zero gradient when the unit is not active
    Args:
        x (tf.tensor): a tensor 
        leak (float): Default to 0.3
    Returns:
        tf.tensor with name 'name'
    """
    return tf.maximum(x, leak*x, name=name)

def batch_norm(x, center=True, scale=True,
               init_beta=tf.zeros_initializer(),
               init_gamma=tf.ones_initializer(),
               init_move_mean=tf.zeros_initializer(),
               init_move_var=tf.ones_initializer(),
               is_training=False, trainable=True, name='bn'):
    """ 
    batch normal 
    Args:
        x (tf.tensor): a tensor 
        name (str): name scope
        train (bool): whether training or not
    Returns:
        tf.tensor with name 'name'
    """
    # return tf.contrib.layers.batch_norm(
    #     x, decay=0.9, updates_collections=None,
    #     epsilon=1e-5, scale=False,
    #     is_training=train, scope=name)

    return tf.layers.batch_normalization(
       inputs=x,
       axis=-1,
       momentum=0.99,
       epsilon=0.001,
       center=center,
       scale=scale,
       beta_initializer=init_beta,
       gamma_initializer=init_gamma,
       moving_mean_initializer=init_move_mean,
       moving_variance_initializer=init_move_var,
       beta_regularizer=None,
       gamma_regularizer=None,
       beta_constraint=None,
       gamma_constraint=None,
       training=is_training,
       trainable=trainable,
       name=name,
       reuse=None,
       renorm=False,
       renorm_clipping=None,
       renorm_momentum=0.99,
       fused=None,
       virtual_batch_size=None,
       adjustment=None
       )


@add_arg_scope
def linear(out_dim,
           layer_dict=None,
           inputs=None,
           init_w=None,
           init_b=tf.zeros_initializer(),
           wd=0,
           bn=False,
           is_training=True,
           name='Linear',
           nl=tf.identity,
           trainable=True,
           add_summary=False):

    with tf.variable_scope(name):
        if inputs is None:
            assert layer_dict is not None
            inputs = layer_dict['cur_input']
        inputs = batch_flatten(inputs)
        in_dim = inputs.get_shape().as_list()[1]
        if wd > 0:
            regularizer = tf.contrib.layers.l2_regularizer(scale=wd)
        else:
            regularizer=None
        weights = tf.get_variable('weights',
                                  shape=[in_dim, out_dim],
                                  # dtype=None,
                                  initializer=init_w,
                                  regularizer=regularizer,
                                  trainable=trainable)
        biases = tf.get_variable('biases',
                                  shape=[out_dim],
                                  # dtype=None,
                                  initializer=init_b,
                                  regularizer=None,
                                  trainable=trainable)

        if add_summary:
            tf.summary.histogram(
                'weights/{}'.format(name), weights, collections = ['train'])

        # print('init: {}'.format(weights))
        act = tf.nn.xw_plus_b(inputs, weights, biases)
        if bn is True:
            act = batch_norm(act, is_training=is_training, name='bn')
        result = nl(act, name='output')
        if layer_dict is not None:
            layer_dict['cur_input'] = result
            layer_dict['weights'] = weights
            
        return result

@add_arg_scope
def transpose_conv(
                   filter_size,
                   layer_dict,
                   inputs=None,
                   out_dim=None,
                   out_shape=None,
                   stride=2,
                   padding='SAME',
                   trainable=True,
                   nl=tf.identity,
                   init_w=None,
                   init_b=tf.zeros_initializer(),
                   wd=0,
                   bn=False,
                   is_training=True,
                   constant_init=False,
                   name='dconv',
                   add_summary=False):
    if inputs is None:
        inputs = layer_dict['cur_input']
    stride = get_shape4D(stride)
    in_dim = inputs.get_shape().as_list()[-1]

    # TODO other ways to determine the output shape 
    x_shape = tf.shape(inputs)
    # assume output shape is input_shape*stride
    if out_shape is None:
        out_shape = tf.stack([x_shape[0],
                              tf.multiply(x_shape[1], stride[1]), 
                              tf.multiply(x_shape[2], stride[2]),
                              out_dim])
    if out_dim is None:
        out_dim = out_shape[-1] 

    filter_shape = get_shape2D(filter_size) + [out_dim, in_dim]

    with tf.variable_scope(name) as scope:
        if wd > 0:
            regularizer = tf.contrib.layers.l2_regularizer(scale=wd)
        else:
            regularizer=None
        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable,
                                  regularizer=regularizer)
        biases = tf.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)
        
        output = tf.nn.conv2d_transpose(inputs,
                                        weights, 
                                        output_shape=out_shape, 
                                        strides=stride, 
                                        padding=padding, 
                                        name=scope.name)
        if add_summary:
            tf.summary.histogram(
                'weights/{}'.format(name), weights, collections = ['train'])

        output = tf.nn.bias_add(output, biases)
        output.set_shape([None, None, None, out_dim])
        if bn is True:
            output = batch_norm(output, is_training=is_training, name='bn')
        output = nl(output, name='output')
        layer_dict['cur_input'] = output
        layer_dict[name] = output
        return output

def max_pool(layer_dict,
             inputs=None,
             name='max_pool',
             filter_size=2,
             stride=None,
             padding='SAME',
             switch=False):
    """ 
    Max pooling layer 
    Args:
        x (tf.tensor): a tensor 
        name (str): name scope of the layer
        filter_size (int or list with length 2): size of filter
        stride (int or list with length 2): Default to be the same as shape
        padding (str): 'VALID' or 'SAME'. Use 'SAME' for FCN.
    Returns:
        tf.tensor with name 'name'
    """
    if inputs is not None:
        layer_dict['cur_input'] = inputs
    padding = padding.upper()
    filter_shape = get_shape4D(filter_size)
    if stride is None:
        stride = filter_shape
    else:
        stride = get_shape4D(stride)

    if switch == True:
        layer_dict['cur_input'], switch_s = tf.nn.max_pool_with_argmax(
            layer_dict['cur_input'],
            ksize=filter_shape, 
            strides=stride, 
            padding=padding,
            Targmax=tf.int64,
            name=name)
        return layer_dict['cur_input'], switch_s
    else:
        layer_dict['cur_input'] = tf.nn.max_pool(
            layer_dict['cur_input'],
            ksize=filter_shape, 
            strides=stride, 
            padding=padding,
            name=name)
        return layer_dict['cur_input'], None

@add_arg_scope
def conv(filter_size,
         out_dim,
         layer_dict,
         inputs=None,
         pretrained_dict=None,
         stride=1,
         dilations=[1, 1, 1, 1],
         bn=False,
         nl=tf.identity,
         init_w=None,
         init_b=tf.zeros_initializer(),
         use_bias=True,
         padding='SAME',
         pad_type='ZERO',
         trainable=True,
         is_training=None,
         wd=0,
         name='conv',
         add_summary=False):
    if inputs is None:
        inputs = layer_dict['cur_input']
    stride = get_shape4D(stride)
    in_dim = inputs.get_shape().as_list()[-1]
    filter_shape = get_shape2D(filter_size) + [in_dim, out_dim]

    if padding == 'SAME' and pad_type == 'REFLECT':
        pad_size_1 = int((filter_shape[0] - 1) / 2)
        pad_size_2 = int((filter_shape[1] - 1) / 2)
        inputs = tf.pad(
            inputs,
            [[0, 0], [pad_size_1, pad_size_1], [pad_size_2, pad_size_2], [0, 0]],
            "REFLECT")
        padding = 'VALID'

    with tf.variable_scope(name):
        if wd > 0:
            regularizer = tf.contrib.layers.l2_regularizer(scale=wd)
        else:
            regularizer=None

        if pretrained_dict is not None and name in pretrained_dict:
            try:
                load_w = pretrained_dict[name][0]
            except KeyError:
                load_w = pretrained_dict[name]['weights']
            print('Load {} weights!'.format(name))

            load_w = np.reshape(load_w, filter_shape)
            init_w = tf.constant_initializer(load_w)

        weights = tf.get_variable('weights',
                                  filter_shape,
                                  initializer=init_w,
                                  trainable=trainable,
                                  regularizer=regularizer)
        if add_summary:
            tf.summary.histogram(
                'weights/{}'.format(name), weights, collections = ['train'])
        outputs = tf.nn.conv2d(inputs,
                               filter=weights,
                               strides=stride,
                               padding=padding,
                               use_cudnn_on_gpu=True,
                               data_format="NHWC",
                               dilations=dilations,
                               name='conv2d')

        if use_bias:
            if pretrained_dict is not None and name in pretrained_dict:
                try:
                    load_b = pretrained_dict[name][1]
                except KeyError:
                    load_b = pretrained_dict[name]['biases']
                print('Load {} biases!'.format(name))

                load_b = np.reshape(load_b, [out_dim])
                init_b = tf.constant_initializer(load_b)

            biases = tf.get_variable('biases',
                                 [out_dim],
                                 initializer=init_b,
                                 trainable=trainable)
            outputs += biases

        if bn is True:
            outputs = batch_norm(outputs, is_training=is_training, name='bn')

        layer_dict['cur_input'] = nl(outputs)
        layer_dict[name] = layer_dict['cur_input']
        return layer_dict['cur_input']
# def conv(filter_size,
#          out_dim,
#          layer_dict,
#          inputs=None,
#          pretrained_dict=None,
#          stride=1,
#          dilations=[1, 1, 1, 1],
#          bn=False,
#          nl=tf.identity,
#          init_w=None,
#          init_b=tf.zeros_initializer(),
#          use_bias=True,
#          fix_pad=None,
#          custom_padding=None,
#          padding='SAME',
#          pad_type='ZERO',
#          trainable=True,
#          is_training=None,
#          wd=0,
#          name='conv',
#          add_summary=False):
#     if inputs is None:
#         inputs = layer_dict['cur_input']
#     stride = get_shape4D(stride)
#     in_dim = inputs.get_shape().as_list()[-1]
#     filter_shape = get_shape2D(filter_size) + [in_dim, out_dim]

#     if custom_padding is not None:
#         inputs = custom_padding(inputs)
#         padding = 'VALID'
#     elif fix_pad is not None:
#         inputs = tf.pad(
#             inputs,
#             [[0, 0], [fix_pad, fix_pad], [fix_pad, fix_pad], [0, 0]],
#             "CONSTANT")
#         padding = 'VALID'
#     elif padding == 'SAME' and pad_type == 'REFLECT':
#         pad_size_1 = int((filter_shape[0] - 1) / 2)
#         pad_size_2 = int((filter_shape[1] - 1) / 2)
#         inputs = tf.pad(
#             inputs,
#             [[0, 0], [pad_size_1, pad_size_1], [pad_size_2, pad_size_2], [0, 0]],
#             "REFLECT")
#         padding = 'VALID'

#     with tf.variable_scope(name):
#         if wd > 0:
#             regularizer = tf.contrib.layers.l2_regularizer(scale=wd)
#         else:
#             regularizer=None

#         if pretrained_dict is not None and name in pretrained_dict:
#             try:
#                 load_w = pretrained_dict[name][0]
#             except KeyError:
#                 load_w = pretrained_dict[name]['weights']
#             print('Load {} weights!'.format(name))

#             load_w = np.reshape(load_w, filter_shape)
#             init_w = tf.constant_initializer(load_w)
#         else:
#             print('Random init {} weights!'.format(name))

#         weights = tf.get_variable('weights',
#                                   filter_shape,
#                                   initializer=init_w,
#                                   trainable=trainable,
#                                   regularizer=regularizer)
#         if add_summary:
#             tf.summary.histogram(
#                 'weights/{}'.format(name), weights, collections = ['train'])
#         outputs = tf.nn.conv2d(inputs,
#                                filter=weights,
#                                strides=stride,
#                                padding=padding,
#                                use_cudnn_on_gpu=True,
#                                data_format="NHWC",
#                                dilations=dilations,
#                                name='conv2d')

#         if use_bias:
#             if pretrained_dict is not None and name in pretrained_dict:
#                 try:
#                     load_b = pretrained_dict[name][1]
#                 except KeyError:
#                     load_b = pretrained_dict[name]['biases']
#                 print('Load {} biases!'.format(name))

#                 load_b = np.reshape(load_b, [out_dim])
#                 init_b = tf.constant_initializer(load_b)
#             else:
#                 print('Random init {} biases!'.format(name))

#             biases = tf.get_variable('biases',
#                                  [out_dim],
#                                  initializer=init_b,
#                                  trainable=trainable)
#             outputs += biases

#         if bn is True:
#             # if bias is used for conv, then beta is useless
#             # as it can be absorbed in the bias
#             # center = False if use_bias else True
#             center = True

#             # init_beta=tf.zeros_initializer()
#             # init_gamma=tf.ones_initializer()
#             # init_move_mean=tf.zeros_initializer()
#             # init_move_var=tf.ones_initializer()

#             if pretrained_dict is not None and name in pretrained_dict:
#                 if 'bn' in pretrained_dict[name]:
#                     bn_weight_list = pretrained_dict[name]['bn']
#                     init_gamma = tf.constant_initializer(bn_weight_list[0])
#                     init_beta = tf.constant_initializer(bn_weight_list[1])
#                     init_move_mean = tf.constant_initializer(bn_weight_list[2])
#                     init_move_var = tf.constant_initializer(bn_weight_list[3])
#                     print('Load {} bn!'.format(name))

#                     outputs = batch_norm(
#                         outputs, center=center,
#                         init_beta=init_beta,
#                         init_gamma=init_gamma,
#                         init_move_mean=init_move_mean,
#                         init_move_var=init_move_var,
#                         is_training=is_training, name='bn')
                    
#             else:
#                 print('Random init {} bn!'.format(name))
#                 outputs = batch_norm(outputs, is_training=is_training, name='bn')

#         layer_dict['cur_input'] = nl(outputs)
#         layer_dict[name] = layer_dict['cur_input']
#         return layer_dict['cur_input']

def drop_out(layer_dict, is_training, inputs=None, keep_prob=0.5):
    if inputs is None:
        inputs = layer_dict['cur_input']
    if is_training:
        layer_dict['cur_input'] = tf.nn.dropout(inputs, keep_prob=keep_prob)
    else:
        layer_dict['cur_input'] = inputs
    return layer_dict['cur_input']

def NN_upsampling(layer_dict, inputs=None, factor=2, name='NN_upsampling'):
    """ upsampling using nearest neighborhood """
    if inputs is None:
        inputs = layer_dict['cur_input']
    layer_dict['cur_input'] = inputs

    in_shape = tf.shape(inputs)
    factor = int(factor)
    new_h = factor * in_shape[1]
    new_w = factor * in_shape[2]

    layer_dict['cur_input'] = tf.image.resize_nearest_neighbor(
                inputs, [new_h, new_w], align_corners=False, name=name)

    return layer_dict['cur_input']

@add_arg_scope
def residual_block_iwgan(
    filter_size,
    out_dim,
    layer_dict,
    resample=None,
    name='res_block',
    padding='SAME',
    pad_type='ZERO',
    is_training=True,
    init_w=None,
    init_b=tf.zeros_initializer(),
    trainable=True):
 
    with tf.variable_scope(name):
        inputs = layer_dict['cur_input']

        bn_input = batch_norm(inputs, is_training=is_training, name='input_bn')
        relu_bn_input = tf.nn.relu(bn_input, name='input_relu')
        layer_dict['cur_input'] = relu_bn_input

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([conv, transpose_conv],
                        filter_size=filter_size,
                        out_dim=out_dim,
                        layer_dict=layer_dict,
                        # bn=True,
                        init_w=init_w,
                        init_b=init_b,
                        padding=padding,
                        is_training=is_training,
                        trainable=trainable,                        
                        ):
            if resample == None:
                shortcut = inputs
                out1 = conv(nl=tf.nn.relu, name='conv1', bn=True, stride=1, pad_type=pad_type)
            elif resample == 'up':
                shortcut = transpose_conv(inputs=inputs, name='input_dconv', bn=False, stride=2)
                out1 = transpose_conv(inputs=inputs, nl=tf.nn.relu, name='dconv1', bn=True, stride=2)
            elif resample == 'down':
                shortcut = conv(inputs=inputs, name='input_conv', bn=False, stride=2, pad_type=pad_type)
                out1 = conv(inputs=inputs, nl=tf.nn.relu, name='conv1', bn=True, stride=2, pad_type=pad_type)
            else:
                raise Exception('invalid resample value')

            conv(inputs=out1, name='conv2', bn=False, stride=1, pad_type=pad_type)

        layer_dict['cur_input'] += shortcut
        return layer_dict['cur_input']

def global_avg_pool(layer_dict, inputs=None, name='global_avg_pool', data_format='NHWC', keepdims=None):
    if inputs is None:
        inputs = layer_dict['cur_input']
    assert inputs.shape.ndims == 4
    assert data_format in ['NHWC', 'NCHW']
    with tf.name_scope(name):
        axis = [1, 2] if data_format == 'NHWC' else [2, 3]
        layer_dict['cur_input'] = tf.reduce_mean(inputs, axis, keepdims=keepdims)
        return layer_dict['cur_input']

def make_LSTM_cell(hidden_size, forget_bias=0.0, is_training=True, keep_prob=1.0):

    cell = tf.contrib.rnn.LSTMCell(
        num_units=hidden_size,
        use_peepholes=False,
        cell_clip=None,
        initializer=None,
        num_proj=None,
        proj_clip=None,
        num_unit_shards=None,
        num_proj_shards=None,
        forget_bias=forget_bias,
        state_is_tuple=True,
        activation=None,
        reuse=None,
        name='lstm'
    )

    if is_training is True:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, 
            output_keep_prob=keep_prob,
            variational_recurrent=True,
            dtype=tf.float32)

    return cell

        
