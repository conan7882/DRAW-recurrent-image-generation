#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: draw.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import argparse
import tensorflow as tf
import numpy as np
import scipy.io

import sys
sys.path.append('../')
import loader
# from src.dataflow.synthetic import Circle
# from src.dataflow.generator import Generator
from src.nets.draw_attention import DRAW
# from src.models.interpolate import linear_interpolate
# from src.helper.visualizer import Visualizer
import config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--viz', action='store_true',
                        help='')

    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--step', type=int, default=10)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--embed', type=int, default=100)
    parser.add_argument('--load', type=int, default=49)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser.parse_args()

def read_mnist():
    def binary_im(im):
        thr = 0.7
        im = im / 255.
        im[np.where(im < thr)] = 0
        im[np.where(im > 0)] = 1
        return im

    train_generator = loader.loadMNIST(pf=binary_im)
    return train_generator

def train():
    FLAGS = get_args()
    n_code = FLAGS.embed
    n_encoder_hidden=256
    n_decoder_hidden=256
    n_step = FLAGS.step
    max_epoch = FLAGS.epoch

    save_draw_path = config.save_draw_path

    data_name = 'mnist'
    im_size = 28
    im_channel = 1
    max_step = None
    read_N = 2
    write_N = 5

    train_generator = read_mnist()
    save_path = os.path.join(save_draw_path, data_name)

    train_net = DRAW(
        im_channel=im_channel,
        n_encoder_hidden=n_encoder_hidden,
        n_decoder_hidden=n_decoder_hidden,
        n_code=n_code,
        im_size=im_size,
        n_step=n_step,
        read_N=read_N,
        write_N=write_N)
    train_net.create_train_model(train_generator.batch_data)

    generate_net = DRAW(
        im_channel=im_channel,
        n_encoder_hidden=n_encoder_hidden,
        n_decoder_hidden=n_decoder_hidden,
        n_code=n_code,
        im_size=im_size,
        n_step=n_step,
        read_N=read_N,
        write_N=write_N)
    generate_net.create_generate_model(b_size=10)

    writer = tf.summary.FileWriter(save_path)
    saver = tf.train.Saver()
    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        
        for i in range(max_epoch):
            train_generator.init_iterator(sess)

            # train_net.test(sess)
            train_net.train_epoch(sess, max_step=max_step, lr=FLAGS.lr, summary_writer=writer)
            generate_net.generate_batch(sess, summary_writer=writer)
            generate_net.viz_generate_step(sess, save_path, file_id=i)
            if i % 10 == 0:
                saver.save(sess, '{}/draw_step_{}'.format(save_path, i))
        saver.save(sess, '{}/draw_step_{}'.format(save_path, i))

def viz():
    FLAGS = get_args()
    n_code = FLAGS.embed
    n_encoder_hidden=256
    n_decoder_hidden=256
    n_step = FLAGS.step
    max_epoch = FLAGS.epoch

    save_draw_path = config.save_draw_path

    data_name = 'mnist'
    im_size = 28
    im_channel = 1
    max_step = None
    read_N = 2
    write_N = 5

    train_generator = read_mnist()
    save_path = os.path.join(save_draw_path, data_name)

    generate_net = DRAW(
        im_channel=im_channel,
        n_encoder_hidden=n_encoder_hidden,
        n_decoder_hidden=n_decoder_hidden,
        n_code=n_code,
        im_size=im_size,
        n_step=n_step,
        read_N=read_N,
        write_N=write_N)
    generate_net.create_generate_model(b_size=400)

    saver = tf.train.Saver()
    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}/draw_step_{}'.format(save_path, FLAGS.load))

        generate_net.viz_generate_step(sess, save_path, is_animation=True)


if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.train:
        train()
    elif FLAGS.viz:
        viz()


    