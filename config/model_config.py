# Copyright 2018 Jaewook Kang (jwkang10@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# import tensorflow.contrib.slim as slim
import json


class ModelConfig(object):

    def __init__(self):
        self.input_size = 192
        self.output_size = 24  # it will be changed later

        self.input_chnum = 3
        self.output_chnum = 14  # number of keypoints
        self.channel_num = 96
        self.reception = RecepConfig()
        self.hourglass = HourglassConfig(channel_num=self.channel_num)
        self.output = OutputConfig()
        self.separable_conv = SeparableConfig()

        self.dtype = tf.float32

        self.decoder_layers = [
            {"filters": 64, "kernel_size": 4, "strides": (4, 4)},
            {"filters": 64, "kernel_size": 4, "strides": (3, 3)},
            {"filters": 64, "kernel_size": 4, "strides": (2, 2)},
        ]


class RecepConfig(object):

    def __init__(self):
        # batch norm config
        self.batch_norm_decay = 0.999
        self.batch_norm_fused = True

        # self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        # self.biases_initializer     = slim.init_ops.zeros_initializer()
        # self.weights_regularizer    = None
        #
        # self.activation_fn          = tf.nn.relu
        # self.normalizer_fn          = slim.batch_norm
        self.is_trainable = True

        self.kernel_shape = { \
            'r1': [7, 7],
            'r4': [3, 3]
        }

        self.strides = { \
            'r1': 2,
            'r4': 2
        }


class HourglassConfig(object):

    def __init__(self, channel_num):
        self.updown_rate = 2
        self.maxpool_kernel_size = [3, 3]
        self.num_stage = 4
        self.center_conv_num = 1
        self.skip_invbottle_num = 3
        self.center_ch_num = channel_num  # output channel num
        # self.center_ch_num          = 14 #output channel num


class OutputConfig(object):

    def __init__(self):
        # batch norm config
        self.batch_norm_decay = 0.999
        self.batch_norm_fused = True

        self.dropout_keeprate = 1.0
        # self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        # self.weights_regularizer    = tf.contrib.layers.l2_regularizer(4E-5)
        # self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.weights_regularizer = None
        self.activation_fn = None
        self.is_trainable = True
        # self.normalizer_fn          = slim.batch_norm

        self.kernel_shape = [1, 1]
        self.stride = 1


class SeparableConfig(object):

    def __init__(self):
        # batch norm config
        self.batch_norm_decay = 0.999
        self.batch_norm_fused = True

        # self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        # self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.weights_regularizer = None
        self.invbottle_expansion_rate = 7.0

        # self.normalizer_fn          = slim.batch_norm
        self.is_trainable = True

        self.activation_fn_dwise = None
        # self.activation_fn_pwise = tf.nn.relu

        self.kernel_shape_dwise = [3, 3]
        self.stride_dwise = 1
