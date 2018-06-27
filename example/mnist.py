#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

import sys
import argparse
import scipy.misc
import numpy as np
import tensorflow as tf

sys.path.append('../')
from lib.dataflow.mnist import MNISTData 
from lib.model.draw import DRAW

DATA_PATH = '/home/qge2/workspace/data/MNIST_data/'
SAVE_PATH = '/home/qge2/workspace/data/out/draw/'
# DATA_PATH = 'E://Dataset//MNIST//'
# SAVE_PATH = 'E:/tmp/draw/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test')
    # parser.add_argument('--trans', action='store_true',
    #                     help='Transform image')
    # parser.add_argument('--center', action='store_true',
    #                     help='Center')

    # parser.add_argument('--step', type=int, default=1,
    #                     help='Number of glimpse')
    # parser.add_argument('--batch', type=int, default=128,
    #                     help='Batch size')
    # parser.add_argument('--epoch', type=int, default=1000,
    #                     help='Max number of epoch')
    # parser.add_argument('--load', type=int, default=100,
    #                     help='Load pretrained parameters with id')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Init learning rate')
    
    return parser.parse_args()

def preprocess_im(im):
    thr = 0.7
    im[np.where(im < thr)] = 0
    im[np.where(im > 0)] = 1
    return im

if __name__ == '__main__':
    FLAGS = get_args()

    model = DRAW(im_channel=1,
                 n_encoder_hidden=256,
                 n_decoder_hidden=256,
                 n_code=10,
                 im_size=28,
                 n_step=10)

    if FLAGS.train:
        train_data = MNISTData('train',
                               data_dir=DATA_PATH,
                               shuffle=True,
                               pf=preprocess_im,
                               batch_dict_name=['im'])
        train_data.setup(epoch_val=0, batch_size=128)

        model.create_model()
        train_op = model.train_op()
        loss_op = model.get_loss()

        model.create_generate_model(b_size=10)
        generate_op = model.layers['generate']
        summary_op = model.get_summary()

        writer = tf.summary.FileWriter(SAVE_PATH)
        decoder_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='step/decoder')
        write_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='step/write')
        saver = tf.train.Saver(
                var_list=decoder_var + write_var)
        sessconfig = tf.ConfigProto()
        sessconfig.gpu_options.allow_growth = True
        with tf.Session(config=sessconfig) as sess:
            sess.run(tf.global_variables_initializer())
            writer.add_graph(sess.graph)
            loss_sum = 0
            for i in range(0, 10000):
                batch_data = train_data.next_batch_dict()

                _, loss = sess.run([train_op, loss_op],
                                   feed_dict={model.raw_image: batch_data['im'],
                                              model.lr: FLAGS.lr})

                loss_sum += loss

                if i % 100 == 0:
                    print(loss_sum / 100.)
                    loss_sum = 0
                    cur_summary = sess.run(summary_op)
                    writer.add_summary(cur_summary, i)
                    saver.save(sess, '{}draw_step_{}'.format(SAVE_PATH, i))

    if FLAGS.test:
        model.create_generate_model(b_size=10)
        generate_op = model.layers['generate']
        step_op = model.layers['gen_step'] 

        decoder_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='step/decoder')
        write_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='step/write')
        saver = tf.train.Saver(
                var_list=decoder_var + write_var)
        sessconfig = tf.ConfigProto()
        sessconfig.gpu_options.allow_growth = True
        with tf.Session(config=sessconfig) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, '{}draw_step_{}'.format(SAVE_PATH, 9900))
            gen_im, gen_step = sess.run([generate_op, step_op])

            for step_id, step_batch in enumerate(gen_step):
                for idx, im in enumerate(step_batch):
                    scipy.misc.imsave('{}im_{}_step_{}.png'.format(SAVE_PATH, idx, step_id), np.squeeze(im))
