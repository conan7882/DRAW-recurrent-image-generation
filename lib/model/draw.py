#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: draw.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
from tensorcv.models.base import BaseModel

import lib.model.layers as L
import lib.model.ops as ops


class DRAW(BaseModel):
    def __init__(self,
                 im_channel,
                 n_encoder_hidden,
                 n_decoder_hidden,
                 n_code,
                 im_size,
                 n_step,
                 ):

        # self._is_transform = is_transform
        # self._trans_size = transform_size
        self._n_channel = im_channel
        self._im_size = L.get_shape2D(im_size)
        self._n_pixel = self._im_size[0] * self._im_size[1] * self._n_channel

        if not isinstance(n_encoder_hidden, list):
            n_encoder_hidden = [n_encoder_hidden]
        self._n_encoder_hidden = n_encoder_hidden
        if not isinstance(n_decoder_hidden, list):
            n_decoder_hidden = [n_decoder_hidden]
        self._n_decoder_hidden = n_decoder_hidden
        self._n_code = n_code
        self._n_step = n_step

        self.layers = {}

        self.set_is_training(True)

    def create_generate_model(self, b_size):
        self.set_is_training(False)
        self.layers['generate'] = self._create_generate_model(b_size)
        self.setup_summary()

    def _create_generate_model(self, b_size):
        self.layers['gen_step'] = [0] * self._n_step
        def _make_cell(hidden_size):
            return L.make_cell(hidden_size,
                               forget_bias=1.0,
                               is_training=self.is_training,
                               keep_prob=1.0)

        gen_im = tf.zeros(shape=[b_size, 28 * 28],
                          dtype=tf.float32)
        with tf.name_scope('init'):
            decoder_cell = tf.contrib.rnn.MultiRNNCell(
                [_make_cell(hidden_size) for hidden_size in self._n_decoder_hidden])
            
            decoder_state = decoder_cell.zero_state(b_size, tf.float32)

            decoder_out = tf.zeros(shape=[b_size, self._n_decoder_hidden[-1]],
                                   dtype=tf.float32)
            c = tf.zeros_like(gen_im)
            self.layers['gen_step'][0] = tf.reshape(
                tf.sigmoid(c),
                [b_size, self._im_size[0], self._im_size[1], self._n_channel])

        for step_id in range(0, self._n_step):
            with tf.variable_scope('step', reuse=tf.AUTO_REUSE):
                z = ops.tf_sample_standard_diag_guassian(b_size, self._n_code)
                
                with tf.variable_scope('decoder'):
                    decoder_out, decoder_state = decoder_cell(z, decoder_state)

                with tf.name_scope('write'):
                    c = c + ops.write(decoder_out, n_pixel=self._n_pixel)
                    self.layers['gen_step'][step_id] = tf.reshape(
                        tf.sigmoid(c),
                        [b_size, self._im_size[0], self._im_size[1], self._n_channel])
        
        gen_im = tf.sigmoid(c)
        return tf.reshape(gen_im, [b_size, 28, 28, 1])

    def create_model(self):
        self._create_input()
        self.layers['cT'] = self._create_model()

    def _create_input(self):
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.raw_image = tf.placeholder(
            tf.float32,
            [None, self._im_size[0], self._im_size[1], self._n_channel],
            name='raw_image')

        self.image = tf.reshape(self.raw_image, (-1, self._n_pixel))

    def _latent_net(self, inputs):
        b_size = tf.shape(inputs)[0]
        z_mean = L.linear(inputs, out_dim=self._n_code, name='mean')
        z_std = L.linear(inputs, out_dim=self._n_code, name='std', nl=L.softplus)
        z_log_std = tf.log(z_std)

        z = ops.tf_sample_diag_guassian(z_mean, z_std, b_size, self._n_code)
        return z, z_mean, z_std, z_log_std

    def _create_model(self):
        def _make_cell(hidden_size):
            return L.make_cell(hidden_size,
                               forget_bias=1.0,
                               is_training=self.is_training,
                               keep_prob=1.0)

        with tf.name_scope('init'):
            encoder_cell = tf.contrib.rnn.MultiRNNCell(
                [_make_cell(hidden_size) for hidden_size in self._n_encoder_hidden])
            decoder_cell = tf.contrib.rnn.MultiRNNCell(
                [_make_cell(hidden_size) for hidden_size in self._n_decoder_hidden])

            b_size = tf.shape(self.image)[0]
            encoder_state = encoder_cell.zero_state(b_size, tf.float32)
            decoder_state = decoder_cell.zero_state(b_size, tf.float32)

            decoder_out = tf.zeros(shape=[b_size, self._n_decoder_hidden[-1]],
                                   dtype=tf.float32)
            c = tf.zeros_like(self.image)

        self.layers['z_mu'] = [0] * self._n_step
        self.layers['z_std'] = [0] * self._n_step
        self.layers['z_log_std'] = [0] * self._n_step
        for step_id in range(0, self._n_step):
            with tf.variable_scope('step', reuse=tf.AUTO_REUSE):
                with tf.name_scope('encoder_input'):
                    error_im = self.image - tf.sigmoid(c)
                    r = ops.read(self.image, error_im, decoder_out)
                    encoder_input = tf.concat((r, decoder_out), axis=-1)

                with tf.variable_scope('encoder'):
                    encoder_out, encoder_state = encoder_cell(encoder_input, encoder_state)

                with tf.variable_scope('latent'):
                    z, self.layers['z_mu'][step_id], self.layers['z_std'][step_id], self.layers['z_log_std'][step_id] =\
                        self._latent_net(encoder_out)
                # z = ops.tf_sample_diag_guassian(z_mean, z_std, b_size, self._n_code)

                with tf.variable_scope('decoder'):
                    decoder_out, decoder_state = decoder_cell(z, decoder_state)

                with tf.name_scope('write'):
                    c = c + ops.write(decoder_out, n_pixel=self._n_pixel)
        return c

    def _get_loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('reconstruction'):
                logits = self.layers['cT']
                labels = self.image

                recons_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.image,
                    logits=logits,
                    name='recons_loss')
                recons_loss = tf.reduce_mean(recons_loss) * self._n_pixel

            with tf.name_scope('KL'):
                self.layers['z_mu'] = tf.convert_to_tensor(self.layers['z_mu']) # [step, batch, code]
                self.layers['z_std'] = tf.convert_to_tensor(self.layers['z_std'])
                self.layers['z_log_std'] = tf.convert_to_tensor(self.layers['z_log_std'])

                kl_loss = tf.reduce_sum(
                    tf.square(self.layers['z_mu'])
                    + tf.square(self.layers['z_std'])
                    - 2 * self.layers['z_log_std'],
                    axis=[0, 2])

                kl_loss = 0.5 * kl_loss - self._n_step / 2 # [batch]
                kl_loss = tf.reduce_mean(kl_loss)

            return recons_loss + kl_loss

    def get_loss(self):
        try:
            return self.loss
        except AttributeError:
            self.loss = self._get_loss()
            return self.loss

    def train_op(self):
        loss = self.get_loss()
        var_list = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        # grads = opt.compute_gradients(loss)
        grads = tf.gradients(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, 5)
        train_op = opt.apply_gradients(zip(grads, var_list))
        return train_op

    def get_summary(self):
        # assert key in ['train', 'test']
        return tf.summary.merge_all(key='generate')

    def setup_summary(self):
        with tf.name_scope('generate'):
            tf.summary.image(
                'image',
                tf.cast(self.layers['generate'], tf.float32),
                collections=['generate'])
