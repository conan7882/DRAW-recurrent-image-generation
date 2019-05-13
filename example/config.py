#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: config.py
# Author: Qian Ge <geqian1001@gmail.com>


import platform


if platform.node() == 'arostitan':
    save_draw_path = '/home/qge2/workspace/data/out/draw/'
    mnist_path = '/home/qge2/workspace/data/MNIST_data/'
else:
    raise ValueError('No data dir setup on this platform!')