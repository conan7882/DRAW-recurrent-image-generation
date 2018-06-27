#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

def softplus(inputs, name):
    return tf.log(1 + tf.exp(inputs), name=name)

def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))

def linear(inputs, out_dim, name='Linear', nl=tf.identity):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        inputs = batch_flatten(inputs)
        in_dim = inputs.get_shape().as_list()[1]
        weights = tf.get_variable('weights',
                                  shape=[in_dim, out_dim],
                                  # dtype=None,
                                  initializer=None,
                                  regularizer=None,
                                  trainable=True)
        biases = tf.get_variable('biases',
                                  shape=[out_dim],
                                  # dtype=None,
                                  initializer=None,
                                  regularizer=None,
                                  trainable=True)
        # print('init: {}'.format(weights))
        act = tf.nn.xw_plus_b(inputs, weights, biases)

        return nl(act, name='output')

def make_cell(hidden_size, forget_bias=0.0,
              is_training=True, keep_prob=1.0):

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

def get_shape2D(in_val):
    """
    Return a 2D shape 
    Args:
        in_val (int or list with length 2) 
    Returns:
        list with length 2
    """
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))
