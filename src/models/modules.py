#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: modules.py
# Author: Qian Ge <geqian1001@gmail.com>

# import sys
import tensorflow as tf
import src.models.layers as L
import src.models.distribution as distribution

EPS = 1e-8

def sample_gaussian_latent(encoder_out, embed_dim, layer_dict, is_training,
                           init_w=None, bn=True, wd=0, trainable=True, name='sample_gaussian_latent'):
    with tf.variable_scope(name):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.linear],
                        out_dim=embed_dim, layer_dict=layer_dict, bn=bn, init_w=init_w,
                        is_training=is_training, wd=wd, trainable=trainable):

            z_mean = L.linear(inputs=encoder_out, init_w=init_w, wd=wd, name='latent_mean')
            z_std = L.linear(nl=L.softplus, inputs=encoder_out, init_w=init_w, wd=wd, name='latent_std')
            z_log_std = tf.log(z_std + 1e-8)

        b_size = tf.shape(encoder_out)[0]
        z = distribution.tf_sample_diag_guassian(z_mean, z_std, b_size, embed_dim)
        return z, z_mean, z_std, z_log_std

def diagonal_gaussian_latent(inputs, embed_dim, layer_dict, is_training,
                             init_w=None, bn=True, wd=0, trainable=True,
                             name='diagonal_gaussian_latent'):
    with tf.variable_scope(name):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.linear],
                        out_dim=embed_dim, layer_dict=layer_dict, bn=bn, init_w=init_w,
                        is_training=is_training, wd=wd, trainable=trainable):

            z_mean = L.linear(inputs=inputs, init_w=init_w, wd=wd, name='latent_mean')
            z_std = L.linear(nl=L.softplus, inputs=inputs, init_w=init_w, wd=wd, name='latent_std')

            return z_mean, z_std

def draw_read(im, error_im, prev_h_decoder):
    with tf.name_scope('read'):
        return tf.concat((im, error_im), axis=-1)

def draw_write(cur_h_decoder, n_pixel, is_training):
    with tf.variable_scope('write'):
        # return L.linear(cur_h_decoder, out_dim=n_pixel, name='write')
        return L.linear(
            out_dim=n_pixel,
            # layer_dict=None,
            inputs=cur_h_decoder,
            init_w=None,
            init_b=tf.zeros_initializer(),
            wd=0,
            bn=False,
            is_training=is_training,
            name='write',
            nl=tf.identity,
            trainable=True,
            add_summary=False)

def draw_filterbank(h_decoder, im_size, filter_grid_size=12,
                    bn=True, init_w=None, wd=0, is_training=True, trainable=True, 
                    name='draw_filterbank'):
    with tf.variable_scope(name):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.linear],
                        bn=bn, init_w=init_w,
                        is_training=is_training, wd=wd, trainable=trainable):

            gxy_hat = L.linear(out_dim=2, inputs=h_decoder, name='grid_center')
            greater_zero = L.linear(out_dim=3, inputs=h_decoder, nl=L.softplus, name='greater_zero')

            A = im_size[1] # width
            B = im_size[0] # heigth
            N = filter_grid_size

            gx_hat = gxy_hat[..., 0]
            gy_hat = gxy_hat[..., 1]

            gx = (A + 1.) / 2. * (gx_hat + 1)
            gy = (B + 1.) / 2. * (gy_hat + 1)
            delta = (max(A, B) - 1.) / (N - 1) * greater_zero[..., 0]

            sigma_square = greater_zero[..., 1] # [batch]
            gamma = greater_zero[..., 2]

            mu_x = tf.stack([gx + (i - N / 2. - 0.5) * delta for i in range(N)], axis=0) # [N, batch]
            mu_y = tf.stack([gy + (i - N / 2. - 0.5) * delta for i in range(N)], axis=0) # [N, batch]

            mu_x = tf.expand_dims(tf.transpose(mu_x), axis=-1) # [batch, N, 1]
            mu_y = tf.expand_dims(tf.transpose(mu_y), axis=-1) # [batch, N, 1]

            a_list = tf.cast(tf.expand_dims([[a for a in range(A)]], axis=1), tf.float32) # [1, 1, A]
            b_list = tf.cast(tf.expand_dims([[b for b in range(B)]], axis=1), tf.float32) # [1, 1, B]

            sigma_square = tf.reshape(sigma_square, (-1, 1, 1))

            Fx = tf.math.exp(-tf.math.square((a_list - mu_x)) / (2 * sigma_square)) # [batch, N, A]
            Fy = tf.math.exp(-tf.math.square((b_list - mu_y)) / (2 * sigma_square)) # [batch, N, B]
            
            Zx = tf.maximum(tf.reduce_sum(Fx, axis=-1, keep_dims=True), EPS) # [batch, N, 1]
            Zy = tf.maximum(tf.reduce_sum(Fy, axis=-1, keep_dims=True), EPS) # [batch, N, 1]

            Fx = Fx / Zx # [batch, N, A]
            Fy = Fy / Zy # [batch, N, B]

            # print_op = tf.print("Debug output:", Zx, Zy)
            return Fx, Fy, gamma

def draw_attention_read(im, error_im, prev_h_decoder, im_size, filter_grid_size=12,
                        bn=True, init_w=None, wd=0, is_training=True, trainable=True, 
                        name='draw_attention_read'):
    with tf.variable_scope(name):
        Fx, Fy, gamma = draw_filterbank(
            prev_h_decoder, im_size, filter_grid_size=filter_grid_size,
            bn=bn, init_w=init_w, wd=wd, is_training=is_training, trainable=trainable)
        FxT = tf.transpose(Fx, perm=[0, 2, 1])

        # im = tf.reshape(im, (-1, 28, 28))
        # error_im = tf.reshape(error_im, (-1, 28, 28))

        att_im = tf.linalg.matmul(tf.linalg.matmul(Fy, im), FxT) # [batch, N, N]
        att_error = tf.linalg.matmul(tf.linalg.matmul(Fy, error_im), FxT) # [batch, N, N]

        gamma = tf.reshape(gamma, (-1, 1, 1))
        att_im = gamma * att_im # [batch, N, N]
        att_error = gamma * att_error # [batch, N, N]

        bsize = tf.shape(im)[0]
        n_pixel = tf.cast(filter_grid_size * filter_grid_size, tf.int32)

        return tf.concat((tf.reshape(att_im, (bsize, n_pixel)), tf.reshape(att_error, (bsize, n_pixel))), axis=-1)

def draw_attention_write(h_decoder, im_size, filter_grid_size=12,
                         bn=True, init_w=None, wd=0, is_training=True, trainable=True, 
                         name='draw_attention_write'):
    with tf.variable_scope(name):
        N = filter_grid_size
        w = L.linear(
            out_dim=N * N,
            inputs=h_decoder,
            init_w=init_w,
            wd=wd,
            bn=bn,
            is_training=is_training,
            name='write',
            nl=tf.identity,
            trainable=trainable)
        w = tf.reshape(w, (-1, N, N))

        Fx, Fy, gamma = draw_filterbank(
            h_decoder, im_size, filter_grid_size=N,
            bn=bn, init_w=init_w, wd=wd, is_training=is_training, trainable=trainable)
        FyT = tf.transpose(Fy, perm=[0, 2, 1])
        gamma = tf.reshape(gamma, (-1, 1, 1))

        # print_op = tf.print("Debug output:", Fx, Fy)

        return 1. / gamma * tf.linalg.matmul(tf.linalg.matmul(FyT, w), Fx) # [batch, B, A]
