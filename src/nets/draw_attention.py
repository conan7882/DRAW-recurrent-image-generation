#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: draw_attention.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

import src.utils.viz as viz
import src.models.layers as L
import src.models.modules as modules
from src.models.base import BaseModel
import src.models.distribution as distribution


INIT_W = tf.keras.initializers.he_normal()
WD = 0
BN = False
EPS = 1e-6

class DRAW(BaseModel):
    def __init__(self,
                 im_channel,
                 n_encoder_hidden,
                 n_decoder_hidden,
                 n_code,
                 im_size,
                 n_step,
                 read_N=2,
                 write_N=5,
                 ):

        self._n_channel = im_channel
        self._im_size = L.get_shape2D(im_size)
        self._n_pixel = self._im_size[0] * self._im_size[1] * self._n_channel

        self.read_N = read_N
        self.write_N = write_N

        if not isinstance(n_encoder_hidden, list):
            n_encoder_hidden = [n_encoder_hidden]
        self._n_encoder_hidden = n_encoder_hidden
        if not isinstance(n_decoder_hidden, list):
            n_decoder_hidden = [n_decoder_hidden]
        self._n_decoder_hidden = n_decoder_hidden

        self._n_code = n_code
        self._n_step = n_step

        self.layers = {}

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
        self.image = tf.reshape(self.raw_image, (-1, self._im_size[0], self._im_size[1]))
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

        with tf.name_scope('Init_RNN_Cell'):
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

        for t in range(0, self._n_step):
            with tf.variable_scope('step', reuse=tf.AUTO_REUSE):
                im_hat = self.image - tf.sigmoid(c)
                r = self.read(self.image, im_hat, decoder_out)
                encoder_out, encoder_state = self.encoder(r, decoder_out, encoder_cell, encoder_state)
                z, self.layers['z_mu'][t], self.layers['z_std'][t], self.layers['z_log_std'][t] =\
                    self.latent_net(encoder_out)
                decoder_out, decoder_state = self.decoder(z, decoder_cell, decoder_state)
                c = c + self.write(decoder_out)
        return c

    def read(self, im, im_hat, decoder_out):
        r = modules.draw_attention_read(
            im, im_hat, decoder_out,
            im_size=self._im_size, filter_grid_size=self.read_N,
            bn=BN, init_w=INIT_W, wd=WD,
            is_training=self.is_training, trainable=True, 
            name='draw_attention_read')
        return r

    def write(self, decoder_out):
        ct = modules.draw_attention_write(
            decoder_out, im_size=self._im_size, filter_grid_size=self.write_N,
            bn=BN, init_w=INIT_W, wd=WD,
            is_training=self.is_training, trainable=True, 
            name='draw_attention_write')
        return ct

    def encoder(self, r, decoder_out, encoder_cell, encoder_state):
        with tf.variable_scope('encoder'):
            encoder_input = tf.concat((r, decoder_out), axis=-1)
            encoder_out, encoder_state = encoder_cell(encoder_input, encoder_state)

        return encoder_out, encoder_state

    def decoder(self, z, decoder_cell, decoder_state):
        with tf.variable_scope('decoder'):
            decoder_out, decoder_state = decoder_cell(z, decoder_state)
            return decoder_out, decoder_state

    def latent_net(self, inputs):
        return modules.sample_gaussian_latent(
            encoder_out=inputs,
            embed_dim=self._n_code,
            layer_dict=self.layers,
            is_training=self.is_training,
            init_w=INIT_W,
            bn=BN,
            wd=WD,
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

        gen_im = tf.zeros(shape=[b_size, self._im_size[0], self._im_size[1]],
                          dtype=tf.float32)
        with tf.name_scope('Init_RNN_Cell'):
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
                decoder_out, decoder_state = self.decoder(z, decoder_cell, decoder_state)
                c = c + self.write(decoder_out)
                self.layers['gen_step'][step_id] = tf.reshape(
                    tf.sigmoid(c),
                    [b_size, self._im_size[0], self._im_size[1], self._n_channel])
        return c

    def _get_loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('reconstruction'):
                output = tf.nn.sigmoid(self.layers['cT'])
                labels = self.image

                recons_loss = -tf.reduce_sum(labels * tf.log(output + EPS)\
                    + (1. - labels) * tf.log(EPS + 1. - output), axis=[1, 2])
                recons_loss = tf.reduce_mean(recons_loss)
                self.print_op = tf.print("Debug output:", recons_loss)

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
            try:
                tf.summary.image(
                    'input',
                    tf.cast(tf.expand_dims(self.image, axis=-1), tf.float32),
                    collections=[name])
            except AttributeError:
                pass
        # assert key in ['train', 'test']
        return tf.summary.merge_all(key=name)

    def test(self, sess):
        sess.run(
            self.print_op,
            feed_dict={self.lr: 0.1, self.keep_prob: 1.})
        # print(ct)

    def train_epoch(self, sess, lr, max_step=None, summary_writer=None):
        if max_step is None:
            max_step = 2 ** 30

        display_name_list = ['loss']
        cur_summary = None

        loss_sum = 0
        step = 0

        while True and step < max_step:
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

    def viz_generate_step(self, sess, save_path, is_animation=False, file_id=None):
        batch_step = sess.run(self.layers['gen_step'])

        if not is_animation:
            step_gen_im = np.vstack(batch_step)
            # print(step_gen_im.shape)

            if file_id is None:
                save_name = '{}/generate_step.png'.format(save_path)
            else:
                save_name = '{}/generate_step_{}.png'.format(save_path, file_id)

            viz.viz_batch_im(
                batch_im=np.clip(step_gen_im * 255., 0., 255.),
                grid_size=[10, self._n_step],
                save_path=save_name,
                is_transpose=True)
        
        else:
            import imageio

            image_list = []
            bsize = batch_step[0].shape[0]
            grid_size = int(bsize ** 0.5)
            for step_id, batch_im in enumerate(batch_step):
                if file_id is None:
                    save_name = '{}/generate_step_{}.png'.format(save_path, step_id)
                else:
                    save_name = '{}/generate_step_{}_{}.png'.format(save_path, file_id, step_id)

                merge_im = viz.viz_batch_im(
                    batch_im=np.clip(batch_im * 255., 0., 255.),
                    grid_size=[grid_size, grid_size],
                    save_path=None,
                    is_transpose=False)

                image_list.append(np.squeeze(merge_im))

            if file_id is None:
                save_name = '{}/draw_generation.gif'.format(save_path)
            else:
                save_name = '{}/draw_generation_{}.gif'.format(save_path, file_id)
            imageio.mimsave(save_name, image_list)

        


