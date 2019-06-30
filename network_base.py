# -*- coding: utf-8 -*-
# @Time    : 18-4-24 5:48 PM
# @Author  : edvard_hua@live.com
# @FileName: network_base.py
# @Software: PyCharm

import tensorflow as tf
# from keras import layers, regularizers, activations
from tensorflow.keras import layers, regularizers, activations

# import tensorflow.contrib.slim as slim

# _init_xavier = tf.contrib.layers.xavier_initializer()
# _init_norm = tf.truncated_normal_initializer(stddev=0.01)
# _init_zero = slim.init_ops.zeros_initializer()
# _l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)
_trainable = True


def is_trainable(trainable=True):
    global _trainable
    _trainable = trainable


def max_pool(inputs, k_h, k_w, s_h, s_w, name, padding="same"):
    return layers.MaxPool2D(pool_size=(k_h, k_w),
                            strides=(s_h, s_w),
                            padding=padding)(inputs)


def upsample(inputs, factor, name):
    return layers.UpSampling2D(size=(factor, factor))(inputs)


def separable_conv(input, c_o, k_s, stride, scope):
    tower = layers.SeparableConv2D(filters=1,
                                   kernel_size=[k_s, k_s],
                                   strides=stride,
                                   padding='same',
                                   activation=None,
                                   depthwise_initializer='glorot_normal',
                                   pointwise_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   depthwise_regularizer=regularizers.l2(0.00004),
                                   pointwise_regularizer=regularizers.l2(0.00004))(input)
    tower = layers.ReLU(max_value=6)(tower)

    tower = layers.Conv2D(c_o,
                          kernel_size=[1, 1],
                          activation=None,
                          kernel_initializer='glorot_normal')(tower)
    tower = layers.BatchNormalization()(tower)
    output = layers.ReLU(max_value=6)(tower)

    return output


def inverted_bottleneck(inputs, up_channel_rate, channels, subsample, k_s=3, scope=""):
    # with tf.variable_scope("inverted_bottleneck_%s" % scope):

    stride = 2 if subsample else 1

    tower = layers.Conv2D(up_channel_rate * inputs.get_shape().as_list()[-1],
                          kernel_size=[1, 1],
                          kernel_initializer='glorot_normal',
                          activation=None)(inputs)
    tower = layers.BatchNormalization()(tower)
    tower = layers.ReLU(max_value=6)(tower)

    tower = layers.SeparableConv2D(filters=1,
                                   kernel_size=k_s,
                                   strides=stride,
                                   padding='same',
                                   depthwise_initializer='glorot_normal',
                                   pointwise_initializer='glorot_normal',
                                   bias_initializer='zeros',
                                   depthwise_regularizer=regularizers.l2(0.00004),
                                   pointwise_regularizer=regularizers.l2(0.00004))(tower)

    tower = layers.Conv2D(channels,
                          kernel_size=[1, 1],
                          kernel_initializer='glorot_normal',
                          activation='relu')(tower)
    tower = layers.BatchNormalization()(tower)
    output = layers.ReLU(max_value=6)(tower)

    if inputs.get_shape().as_list()[-1] == channels:
        # output = tf.add(inputs, output)
        output = layers.Add()([inputs, output])

    return output


def convb(input, k_h, k_w, c_o, stride, name, relu=True):
    tower = layers.Conv2D(c_o,
                          kernel_size=[k_h, k_w],
                          strides=(stride, stride),
                          activation=None,
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=regularizers.l2(0.00004))(input)
    if relu:
        tower = layers.ReLU()(tower)
    tower = layers.BatchNormalization()(tower)
    output = layers.ReLU(max_value=6)(tower)

    return output
