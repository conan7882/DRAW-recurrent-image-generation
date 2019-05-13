#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import sys
import numpy as np
import tensorflow as tf
# import skimage.transform
sys.path.append('../')
import config
from src.dataflow.mnist import MNISTData
from src.dataflow.generator import Generator


def loadMNIST(pf, batch_size=128):
    data_path = config.mnist_path
    train_data = MNISTData(name='train', data_dir=data_path, pf=pf)

    output_types = (tf.float32)
    output_shapes = (tf.TensorShape([28, 28, 1]))

    train_generator = Generator(
        data_generator=train_data.next_im_generator,
        output_types=output_types,
        output_shapes=output_shapes,
        batch_size=batch_size,
        buffer_size=4,
        num_parallel_preprocess=2)

    return train_generator
    