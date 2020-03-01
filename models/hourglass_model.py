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
# -*- coding: utf-8 -*-

# ref: https://github.com/edvardHua/PoseEstimationForMobile/blob/master/training/src/network_mv2_hourglass.py

import tensorflow as tf
from tensorflow.keras import models, layers
# from keras import models, layers
# import keras
from network_base import max_pool, upsample, inverted_bottleneck, separable_conv, convb, is_trainable

N_KPOINTS = 14
STAGE_NUM = 3

out_channel_ratio = lambda d: int(d * 1.0)
up_channel_ratio = lambda d: int(d * 1.0)


class HourglassModelBuilder():

    def __init__(self):
        # self.build_model()
        print("new HourglassModelBuilder")

    def build_model(self, inputs=None, trainable=True):
        if inputs != None:
            inputs = tf.keras.Input(tensor=inputs)  # input must be (256, 256, 3)
        else:
            inputs = tf.keras.Input(shape=(128, 128, 3))  # Returns a placeholder tensor

        predictions, l2s = self.build_network(inputs, trainable=trainable)

        self.model = tf.keras.Model(inputs=inputs, outputs=predictions)

    def hourglass_module(self, inp, stage_nums, intermediate_heatmap_layers):
        if stage_nums > 0:
            down_sample = max_pool(inp, 2, 2, 2, 2, name="hourglass_downsample_%d" % stage_nums)

            tower = inverted_bottleneck(down_sample, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
            tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
            tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
            tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
            block_front = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)

            stage_nums -= 1
            block_mid = self.hourglass_module(block_front, stage_nums, intermediate_heatmap_layers)
            block_back = inverted_bottleneck(
                block_mid, up_channel_ratio(6), N_KPOINTS,
                0, 3, scope="hourglass_back_%d" % stage_nums)

            up_sample = upsample(block_back, 2, "hourglass_upsample_%d" % stage_nums)

            # jump layer
            tower = inverted_bottleneck(inp, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
            tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
            tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
            tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
            branch_jump = inverted_bottleneck(tower, up_channel_ratio(6), N_KPOINTS, 0, 3)

            curr_hg_out = layers.Add()([up_sample, branch_jump])

            # mid supervise
            intermediate_heatmap_layers.append(curr_hg_out)

            return curr_hg_out
        else:
            return inverted_bottleneck(
                inp, up_channel_ratio(6), out_channel_ratio(24),
                0, 3, scope="hourglass_mid_%d" % stage_nums
            )

    def build_network(self, input, trainable):
        is_trainable(trainable)

        intermediate_heatmap_layers = []

        tower = convb(input, 3, 3, out_channel_ratio(16), 2, name="Conv2d_0")

        # 128, 112
        tower = inverted_bottleneck(tower, 1, out_channel_ratio(16), 0, 3)
        tower = inverted_bottleneck(tower, 1, out_channel_ratio(16), 0, 3)

        # 64, 56
        tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 1, 3)
        tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
        tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
        tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)
        tower = inverted_bottleneck(tower, up_channel_ratio(6), out_channel_ratio(24), 0, 3)

        net_h_w = int(tower.shape[1])
        # build network recursively
        hg_out = self.hourglass_module(tower, STAGE_NUM, intermediate_heatmap_layers)

        for index, l2 in enumerate(intermediate_heatmap_layers):
            l2_w_h = int(l2.shape[1])
            if l2_w_h == net_h_w:
                continue
            scale = net_h_w // l2_w_h
            intermediate_heatmap_layers[index] = upsample(l2, scale, name="upsample_for_loss_%d" % index)
        merged_layer = tf.keras.layers.Average()(intermediate_heatmap_layers)
        return hg_out, merged_layer
