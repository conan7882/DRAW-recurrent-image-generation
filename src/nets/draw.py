#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: draw.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

import src.utils.viz as viz
import src.models.layers as L
import src.models.modules as modules
from src.models.base import BaseModel
import src.models.distribution as distribution


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

    # def create_generate_model(self, b_size):
    #     self.set_is_training(False)
    #     self.layers['generate'] = self._create_generate_model(b_size)
    #     self.setup_summary()

    def _create_generate_input(self):
        """ receive and create training data
            Args:
                input_batch (tensor): Return of tf.data.Iterator.get_next() with length 1.
                    The order of data should be: image
        """
        self.keep_prob = 1.

    def create_generate_model(self, b_size):
        self.set_is_training(False)
        self._create_generate_input()
        self.layers['cT'] = self._create_generate_model(b_size)
        self.layers['generate'] = tf.reshape(
            tf.sigmoid(self.layers['cT']), [-1, self._im_size[0], self._im_size[1], self._n_channel])

        self.generate_summary_op = self.get_summary('generate')
        self.global_step = 0

    def _create_train_input(self, input_batch):
        """ receive and create training data
            Args:
                input_batch (tensor): Return of tf.data.Iterator.get_next() with length 1.
                    The order of data should be: image
        """
        self.raw_image = input_batch
        self.image = tf.reshape(self.raw_image, (-1, self._n_pixel))
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def create_train_model(self, input_batch):
        self.set_is_training(True)
        self._create_train_input(input_batch)
        self.layers['cT'] = self._create_train_model()
        self.layers['generate'] = tf.reshape(
            tf.sigmoid(self.layers['cT']), [-1, self._im_size[0], self._im_size[1], self._n_channel])

        self.train_op = self.get_train_op()
        self.loss_op = self.get_loss()
        self.train_summary_op = self.get_summary('train')

        self.global_step = 0

    def _create_train_model(self):
        def _make_cell(hidden_size):
            return L.make_LSTM_cell(hidden_size,
                                    forget_bias=1.0,
                                    is_training=self.is_training,
                                    keep_prob=self.keep_prob)

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
                    # error_im = tf.sigmoid(c)
                    r = modules.draw_read(self.image, error_im, decoder_out)

                    modules.draw_attention_read(
                        self.image, error_im, decoder_out,
                        im_size=self._im_size, filter_grid_size=12,
                        bn=True, init_w=None, wd=0,
                        is_training=self.is_training, trainable=True, 
                        name='draw_attention_read')

                    encoder_input = tf.concat((r, decoder_out), axis=-1)

                with tf.variable_scope('encoder'):
                    encoder_out, encoder_state = encoder_cell(encoder_input, encoder_state)

                with tf.variable_scope('latent'):
                    z, self.layers['z_mu'][step_id], self.layers['z_std'][step_id], self.layers['z_log_std'][step_id] =\
                        self._latent_net(encoder_out)

                with tf.variable_scope('decoder'):
                    decoder_out, decoder_state = decoder_cell(z, decoder_state)

                with tf.name_scope('write'):
                    c = c + modules.draw_write(decoder_out, n_pixel=self._n_pixel, is_training=self.is_training)
        return c

    def _latent_net(self, inputs):
        return modules.sample_gaussian_latent(
            encoder_out=inputs,
            embed_dim=self._n_code,
            layer_dict=self.layers,
            is_training=self.is_training,
            init_w=None,
            bn=False,
            wd=0,
            trainable=True,
            name='sample_gaussian_latent')

    def _create_generate_model(self, b_size):
        self.layers['gen_step'] = [0] * self._n_step
        def _make_cell(hidden_size):
            return L.make_LSTM_cell(
                hidden_size,
                forget_bias=1.0,
                is_training=self.is_training,
                keep_prob=1.0)

        gen_im = tf.zeros(shape=[b_size, self._n_pixel],
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
                z = distribution.tf_sample_standard_diag_guassian(b_size, self._n_code)
                
                with tf.variable_scope('decoder'):
                    decoder_out, decoder_state = decoder_cell(z, decoder_state)

                with tf.name_scope('write'):
                    c = c + modules.draw_write(decoder_out, n_pixel=self._n_pixel, is_training=self.is_training)
                    self.layers['gen_step'][step_id] = tf.reshape(
                        tf.sigmoid(c),
                        [b_size, self._im_size[0], self._im_size[1], self._n_channel])
        
        # gen_im = tf.sigmoid(c)
        # return tf.reshape(gen_im, [b_size, 28, 28, 1])
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

    def get_train_op(self):
        loss = self.get_loss()
        var_list = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        # grads = opt.compute_gradients(loss)
        grads = tf.gradients(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, 5)
        train_op = opt.apply_gradients(zip(grads, var_list))
        return train_op

    def get_summary(self, name):
        with tf.name_scope('generate'):
            tf.summary.image(
                'generate',
                tf.cast(self.layers['generate'], tf.float32),
                collections=[name])
        # assert key in ['train', 'test']
        return tf.summary.merge_all(key=name)

    # def setup_summary(self):
    #     with tf.name_scope('generate'):
    #         tf.summary.image(
    #             'image',
    #             tf.cast(self.layers['generate'], tf.float32),
    #             collections=['generate'])

    def train_epoch(self, sess, lr, summary_writer=None):

        display_name_list = ['loss']
        cur_summary = None

        loss_sum = 0
        step = 0

        while True:
            try:
                step += 1
                self.global_step += 1

                _, loss, cur_summary = sess.run(
                    [self.train_op, self.loss_op, self.train_summary_op],
                    feed_dict={self.lr: lr, self.keep_prob: 1.})
                loss_sum += loss
                
                if step % 100 == 0:
                    viz.display(
                        self.global_step,
                        step,
                        [loss_sum],
                        display_name_list,
                        'train',
                        summary_val=cur_summary,
                        summary_writer=summary_writer)                

            except tf.errors.OutOfRangeError:
                break

        viz.display(
            self.global_step,
            step,
            [loss_sum],
            display_name_list,
            'train',
            summary_val=cur_summary,
            summary_writer=summary_writer) 

    def generate_batch(self, sess, summary_writer=None):
        display_name_list = []
        self.global_step += 1
        cur_summary = sess.run(self.generate_summary_op)
        viz.display(
            self.global_step,
            1,
            [],
            display_name_list,
            'train',
            summary_val=cur_summary,
            summary_writer=summary_writer) 

    def viz_generate_step(self, sess, save_path):
        batch_step = sess.run(self.layers['gen_step'])
        step_gen_im = np.vstack(batch_step)

        viz.viz_batch_im(
            batch_im=step_gen_im * 255.,
            grid_size=[10, 10],
            save_path='{}/generate_step.png'.format(save_path),
            is_transpose=True)
        


