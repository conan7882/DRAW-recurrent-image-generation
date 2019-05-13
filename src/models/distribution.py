#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distribution.py
# Author: Qian Ge <geqian1001@gmail.com>


import tensorflow as tf
import tensorflow_probability as tfp

EPS = 1e-8


def tf_sample_standard_diag_guassian(b_size, n_code):
    mean_list = [0.0 for i in range(0, n_code)]
    std_list = [1.0 for i in range(0, n_code)]
    mvn = tfp.distributions.MultivariateNormalDiag(
        loc=mean_list,
        scale_diag=std_list)
    samples = mvn.sample(sample_shape=(b_size,), seed=None, name='sample')
    return samples

def tf_sample_diag_guassian(mean, std, b_size, n_code):
    mean_list = [0.0 for i in range(0, n_code)]
    std_list = [1.0 for i in range(0, n_code)]
    mvn = tfp.distributions.MultivariateNormalDiag(
        loc=mean_list,
        scale_diag=std_list)
    samples = mvn.sample(sample_shape=(b_size,), seed=None, name='sample')
    samples = mean + tf.multiply(std, samples)

    return samples

def kl_diag_gaussian(mu_1, mu_2, std_1, std_2):
    log_std_1 = tf.log(std_1 + EPS)
    log_std_2 = tf.log(std_2 + EPS)

    std_square_1 = tf.square(std_1)
    std_square_2 = tf.square(std_2)

    kl_loss = 2 * log_std_2 - 2 * log_std_1 - 1 + std_square_1 / (std_square_2 + EPS)\
        + tf.square(mu_1 - mu_2) / (std_square_2 + EPS)

    kl_loss = 0.5 * kl_loss
    kl_loss_mean = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    return kl_loss_mean, kl_loss


