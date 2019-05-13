#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: generator.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import tensorflow as tf

class Generator(object):
    def __init__(self, data_generator, output_types, output_shapes, batch_size,
                 buffer_size, num_parallel_preprocess, shuffle=True):

        with tf.name_scope('data_generator'):
            dataset = tf.data.Dataset().from_generator(
                data_generator,
                output_types=output_types,
                output_shapes=output_shapes,
                )

            self._n_preprocess = num_parallel_preprocess
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=buffer_size)
            if shuffle:
                dataset = dataset.shuffle(buffer_size=buffer_size * 10)

            self.iter = dataset.make_initializable_iterator()
            self.batch_data = self.iter.get_next()
            self.dataset = dataset

    def init_iterator(self, sess, reset_scale=False):
        sess.run(self.iter.initializer)

class PairGenerator(object):
    def __init__(self, dataflow, n_channle, image_size,
                 batch_size, buffer_size, num_parallel_preprocess):

        dataflow.set_batch_size(1)
        self._n_preprocess = num_parallel_preprocess

        def generator():
            while True:
                batch_data = dataflow.next_sample()
                yield batch_data[0], batch_data[1], batch_data[2]

        with tf.name_scope('data_generator'):
            dataset = tf.data.Dataset().from_generator(
                generator,
                output_types=(tf.float32, tf.float32, tf.int64),
                output_shapes=(tf.TensorShape([image_size[0], image_size[1], n_channle]),
                               tf.TensorShape([image_size[0], image_size[1], n_channle]),
                               tf.TensorShape([1])),
                )

            # dataset = dataset.map(lambda x, y: self.preprocessor.tf_process_batch(x, y, 2),
            #                       num_parallel_calls=self._n_preprocess)
            # dataset = dataset.map(map_func=self.preprocessor.tf_process_batch,
            #                       num_parallel_calls=num_parallel_preprocess)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=buffer_size)

            self.iter = dataset.make_initializable_iterator()
            self.batch_data = self.iter.get_next()

            self.dataset = dataset

    def init_iterator(self, sess, reset_scale=False):
        sess.run(self.iter.initializer)

class TripletGenerator(object):
    def __init__(self, dataflow, n_channle, image_size,
                 batch_size, num_sample_per_class,
                 buffer_size, num_parallel_preprocess):

        dataflow.set_batch_size(1)
        self._n_preprocess = num_parallel_preprocess

        def generator():
            return dataflow.triplet_generator(num_sample_per_class)

        with tf.name_scope('data_generator'):
            dataset = tf.data.Dataset().from_generator(
                generator,
                output_types= (tf.float32, tf.int64),
                output_shapes=(tf.TensorShape([image_size[0], image_size[1], n_channle]),
                               tf.TensorShape([1])),
                )

            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=buffer_size)

            self.iter = dataset.make_initializable_iterator()
            self.batch_data = self.iter.get_next()

            self.dataset = dataset

    def init_iterator(self, sess, reset_scale=False):
        sess.run(self.iter.initializer)


