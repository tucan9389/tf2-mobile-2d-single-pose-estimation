# Copyright 2019 Doyoung Gwak (tucan.dev@gmail.com)
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
#-*- coding: utf-8 -*-

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf

l2_regularizer_00004 = regularizers.l2(0.00004)

def _inverted_bottleneck(input, up_channel_rate, channels, is_subsample, kernel_size):
    if is_subsample:
        strides = (2, 2)
    else:
        strides = (1, 1)

    kernel_size = (kernel_size, kernel_size)

    # 1x1 conv2d
    x = layers.Conv2D(filters=up_channel_rate * input.shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='SAME')(input)
    x = layers.BatchNormalization(momentum=0.999)(x)
    x = layers.ReLU(max_value=6)(x)

    # activation
    x = layers.ReLU()(x)

    # 3x3 separable_conv2d
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding="SAME",
                               kernel_regularizer=l2_regularizer_00004)(x)
    # activation
    x = layers.ReLU()(x)

    # 1x1 conv2d
    x = layers.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), padding='SAME')(x)
    x = layers.BatchNormalization(momentum=0.999)(x)
    x = layers.ReLU(max_value=6)(x)

    if input.shape[-1] == channels:
        x = input + x

    return x

def _separable_conv(input, channels, kernel_size, strides):
    # 3x3 separable_conv2d
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding="SAME",
                               kernel_regularizer=l2_regularizer_00004)(input)
    # activation
    x = layers.ReLU()(x)

    # 1x1 conv2d
    x = layers.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), padding='SAME')(x)
    x = layers.BatchNormalization(momentum=0.999)(x)
    x = layers.ReLU(max_value=6)(x)

    return x

def _mobilenetV2(input):
    x = _inverted_bottleneck(input, up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)
    x = _inverted_bottleneck(x, up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)
    mv2_branch_0 = x
    # print("mv2_branch_0.shape:", mv2_branch_0.shape)

    # x = _inverted_bottleneck(x, up_channel_rate=6, channels=18, is_subsample=True, kernel_size=3)
    # x = _inverted_bottleneck(x, up_channel_rate=6, channels=18, is_subsample=False, kernel_size=3)
    # x = _inverted_bottleneck(x, up_channel_rate=6, channels=18, is_subsample=False, kernel_size=3)
    # x = _inverted_bottleneck(x, up_channel_rate=6, channels=18, is_subsample=False, kernel_size=3)
    # x = _inverted_bottleneck(x, up_channel_rate=6, channels=18, is_subsample=False, kernel_size=3)
    # mv2_branch_1 = x

    x = _inverted_bottleneck(x, up_channel_rate=6, channels=24, is_subsample=True, kernel_size=3)
    x = _inverted_bottleneck(x, up_channel_rate=6, channels=24, is_subsample=False, kernel_size=3)
    x = _inverted_bottleneck(x, up_channel_rate=6, channels=24, is_subsample=False, kernel_size=3)
    x = _inverted_bottleneck(x, up_channel_rate=6, channels=24, is_subsample=False, kernel_size=3)
    x = _inverted_bottleneck(x, up_channel_rate=6, channels=24, is_subsample=False, kernel_size=3)
    mv2_branch_2 = x
    # print("mv2_branch_2.shape:", mv2_branch_2.shape)

    # x = _inverted_bottleneck(x, up_channel_rate=6, channels=48, is_subsample=True, kernel_size=3)
    # x = _inverted_bottleneck(x, up_channel_rate=6, channels=48, is_subsample=False, kernel_size=3)
    # x = _inverted_bottleneck(x, up_channel_rate=6, channels=48, is_subsample=False, kernel_size=3)
    # x = _inverted_bottleneck(x, up_channel_rate=6, channels=48, is_subsample=False, kernel_size=3)
    # x = _inverted_bottleneck(x, up_channel_rate=6, channels=48, is_subsample=False, kernel_size=3)
    # mv2_branch_3 = x

    x = _inverted_bottleneck(x, up_channel_rate=6, channels=72, is_subsample=True, kernel_size=3)
    x = _inverted_bottleneck(x, up_channel_rate=6, channels=72, is_subsample=False, kernel_size=3)
    x = _inverted_bottleneck(x, up_channel_rate=6, channels=72, is_subsample=False, kernel_size=3)
    x = _inverted_bottleneck(x, up_channel_rate=6, channels=72, is_subsample=False, kernel_size=3)
    x = _inverted_bottleneck(x, up_channel_rate=6, channels=72, is_subsample=False, kernel_size=3)
    mv2_branch_4 = x
    # print("mv2_branch_4.shape:", mv2_branch_4.shape)

    x = layers.Concatenate(axis=3)([
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(mv2_branch_0),
        # layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')(mv2_branch_1),
        mv2_branch_2,
        # layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(mv2_branch_3),
        layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(mv2_branch_4),
    ])

    return x

def build_mv2_cpm_model(number_of_keypoints):
    input_shape = (192, 192, 3)  # h, w, c
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                      kernel_regularizer=l2_regularizer_00004)(input)
    # batch norm
    x = layers.BatchNormalization(momentum=0.999)(x)
    # activation
    x = layers.ReLU(max_value=6)(x)

    # ===============================================
    # ================= MobileNetV2 =================
    mobilenetv2 = _mobilenetV2(x)

    # ===============================================
    # ===================== CPM =====================
    cpm_stage_num = 4
    previous_x = None
    middle_output_layers = []
    for stage in range(cpm_stage_num):
        if previous_x is None:
            x = mobilenetv2
        else:
            x = layers.Concatenate(axis=3)([mobilenetv2, previous_x])

        if stage == 0:
            kernel_size = 3
            lastest_channel_size = 512
        else:
            kernel_size = 7
            lastest_channel_size = 128

        x = _inverted_bottleneck(x, up_channel_rate=2, channels=24, is_subsample=False, kernel_size=kernel_size)
        print("ib1:", x.shape)
        x = _inverted_bottleneck(x, up_channel_rate=4, channels=24, is_subsample=False, kernel_size=kernel_size)
        print("ib2:", x.shape)
        x = _inverted_bottleneck(x, up_channel_rate=4, channels=24, is_subsample=False, kernel_size=kernel_size)
        print("ib3:", x.shape)

        x = _separable_conv(x, channels=lastest_channel_size, kernel_size=1, strides=1)
        print("sc1:", x.shape)
        x = _separable_conv(x, channels=number_of_keypoints, kernel_size=1, strides=1)
        print("sc2:", x.shape)

        middle_output_layers.append(x)
        previous_x = x

    model = models.Model(input, outputs=middle_output_layers)
    return model

model = build_mv2_cpm_model(number_of_keypoints=14)
# model.summary()
print()
