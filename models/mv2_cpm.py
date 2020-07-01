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
tf.random.set_seed(3)

class InvertedBottleneck(tf.keras.layers.Layer):
    def __init__(self, up_channel_rate, channels, is_subsample, kernel_size):
        super(InvertedBottleneck, self).__init__()

        self.up_channel_rate = up_channel_rate
        self.l2_regularizer_00004 = regularizers.l2(0.00004)
        strides = (2, 2) if is_subsample else (1, 1)
        kernel_size = (kernel_size, kernel_size)
        self.dw_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding="SAME",
                               kernel_regularizer=self.l2_regularizer_00004)
        self.conv1 = layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.conv2 = layers.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), padding='SAME')

        self.bn1 = layers.BatchNormalization(momentum=0.999)
        self.bn2 = layers.BatchNormalization(momentum=0.999)
        self.relu = layers.ReLU()
        self.relu6 = layers.ReLU(max_value=6)

    def call(self, inputs, training=True):
        # 1x1 conv2d
        self.conv1.filters = self.up_channel_rate * inputs.shape[-1]
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu6(x)

        # activation
        x = self.relu(x)

        # 3x3 separable_conv2d
        x = self.dw_conv(x)
        # activation
        x = self.relu(x)

        # 1x1 conv2d
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu6(x)

        if inputs.shape[-1] == x.shape[-1] and inputs.shape[-2] == x.shape[-2] and inputs.shape[-3] == x.shape[-3]:
            x = inputs + x

        return x

class SeparableConv2D(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size, strides):
        super(SeparableConv2D, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.strides = strides

        self.l2_regularizer_00004 = regularizers.l2(0.00004)
        self.dw_conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding="SAME",
                                              kernel_regularizer=self.l2_regularizer_00004)
        self.conv = layers.Conv2D(filters=channels, kernel_size=(1, 1), strides=(1, 1), padding='SAME')
        self.relu = layers.ReLU()
        self.relu6 = layers.ReLU(max_value=6)
        self.bn = layers.BatchNormalization(momentum=0.999)

    def call(self, inputs, training=True):
        # 3x3 separable_conv2d
        x = self.dw_conv(inputs)
        # activation
        x = self.relu(x)

        # 1x1 conv2d
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu6(x)

        return x


class BranchBlock(tf.keras.layers.Layer):
    def __init__(self, ibs_is_subsamples, up_channel_rate, channels, kernel_size):
        super(BranchBlock, self).__init__()

        self.number_of_inverted_bottlenecks = len(ibs_is_subsamples)
        self.ibs_is_subsamples = ibs_is_subsamples
        self.up_channel_rate = up_channel_rate
        self.channels = channels
        self.kernel_size = kernel_size

        self.ibs = []
        for is_subsample in self.ibs_is_subsamples:
            ib = InvertedBottleneck(up_channel_rate=self.up_channel_rate,
                                    channels=self.channels,
                                    is_subsample=is_subsample,
                                    kernel_size=self.kernel_size)
            self.ibs.append(ib)

    def call(self, inputs):
        x = inputs
        for ib in self.ibs:
            x = ib(x)

        return x


class MobileNetV2(tf.keras.layers.Layer):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        self.front_ib1 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)
        self.front_ib2 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)

        self.branch1 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=18, kernel_size=3)
        self.branch2 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False, False],
                                   up_channel_rate=6, channels=24, kernel_size=3)
        self.branch3 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False, False],
                                   up_channel_rate=6, channels=48, kernel_size=3)
        self.branch4 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False, False],
                                   up_channel_rate=6, channels=72, kernel_size=3)

        self.max_pool4x4 = layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='SAME')
        self.max_pool2x2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
        self.upsampleing2x2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsampleing4x4 = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')
        self.concat = layers.Concatenate(axis=3)

    def call(self, inputs):
        x = self.front_ib1(inputs)
        x = self.front_ib2(x)
        mv2_branch_0 = x

        x = self.branch1(x)
        mv2_branch_1 = x

        x = self.branch2(x)
        mv2_branch_2 = x

        x = self.branch3(x)
        mv2_branch_3 = x

        x = self.branch4(x)
        mv2_branch_4 = x

        x = self.concat([
            self.max_pool4x4(mv2_branch_0),
            self.max_pool2x2(mv2_branch_1),
            mv2_branch_2,
            self.upsampleing2x2(mv2_branch_3),
            self.upsampleing4x4(mv2_branch_4)
        ])

        return x

class Backbone_1(tf.keras.layers.Layer):
    def __init__(self):
        super(Backbone_1, self).__init__()

        self.front_ib1 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)
        self.front_ib2 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)

        self.branch1 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=18, kernel_size=3)
        self.branch2 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=24, kernel_size=3)
        self.branch3 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=48, kernel_size=3)
        self.branch4 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=72, kernel_size=3)

        # self.max_pool4x4 = layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='SAME')
        self.max_pool2x2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
        self.upsampleing2x2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsampleing4x4 = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')
        self.upsampleing8x8 = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')
        self.concat = layers.Concatenate(axis=3)

    def call(self, inputs):
        x = self.front_ib1(inputs)
        x = self.front_ib2(x)
        mv2_branch_0 = x

        x = self.branch1(x)
        mv2_branch_1 = x

        x = self.branch2(x)
        mv2_branch_2 = x

        x = self.branch3(x)
        mv2_branch_3 = x

        x = self.branch4(x)
        mv2_branch_4 = x

        x = self.concat([
            self.max_pool2x2(mv2_branch_0),
            mv2_branch_1,
            self.upsampleing2x2(mv2_branch_2),
            self.upsampleing4x4(mv2_branch_3),
            self.upsampleing8x8(mv2_branch_4)
        ])

        return x

class Backbone_2(tf.keras.layers.Layer):
    def __init__(self):
        super(Backbone_2, self).__init__()

        self.front_ib1 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)
        self.front_ib2 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)

        self.branch1 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=18, kernel_size=3)
        self.branch2 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=24, kernel_size=3)
        self.branch3 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=48, kernel_size=3)
        self.branch4 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=72, kernel_size=3)

        # self.max_pool4x4 = layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='SAME')
        # self.max_pool2x2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
        self.upsampleing2x2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsampleing4x4 = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')
        self.upsampleing8x8 = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')
        self.upsampleing16x16 = layers.UpSampling2D(size=(16, 16), interpolation='bilinear')
        self.concat = layers.Concatenate(axis=3)

    def call(self, inputs):
        x = self.front_ib1(inputs)
        x = self.front_ib2(x)
        mv2_branch_0 = x

        x = self.branch1(x)
        mv2_branch_1 = x

        x = self.branch2(x)
        mv2_branch_2 = x

        x = self.branch3(x)
        mv2_branch_3 = x

        x = self.branch4(x)
        mv2_branch_4 = x

        x = self.concat([
            mv2_branch_0,
            self.upsampleing2x2(mv2_branch_1),
            self.upsampleing4x4(mv2_branch_2),
            self.upsampleing8x8(mv2_branch_3),
            self.upsampleing16x16(mv2_branch_4)
        ])

        return x


class CPMStageBlock(tf.keras.layers.Layer):
    def __init__(self, kernel_size, lastest_channel_size, number_of_keypoints):
        super(CPMStageBlock, self).__init__()

        self.ib1 = InvertedBottleneck(up_channel_rate=2, channels=24, is_subsample=False, kernel_size=kernel_size)
        self.ib2 = InvertedBottleneck(up_channel_rate=4, channels=24, is_subsample=False, kernel_size=kernel_size)
        self.ib3 = InvertedBottleneck(up_channel_rate=4, channels=24, is_subsample=False, kernel_size=kernel_size)

        self.sconv1 = SeparableConv2D(channels=lastest_channel_size, kernel_size=1, strides=1)
        self.sconv2 = SeparableConv2D(channels=number_of_keypoints, kernel_size=1, strides=1)

    def call(self, inputs):
        x = self.ib1(inputs)
        x = self.ib2(x)
        x = self.ib3(x)

        x = self.sconv1(x)
        x = self.sconv2(x)

        return x

class ConvolutionalPoseMachine(tf.keras.models.Model):
    def __init__(self, backbone, number_of_stages, number_of_keypoints):
        super(ConvolutionalPoseMachine, self).__init__()

        self.number_of_stages = number_of_stages
        self.number_of_keypoints = number_of_keypoints

        self.l2_regularizer_00004 = regularizers.l2(0.00004)
        self.conv = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                      kernel_regularizer=self.l2_regularizer_00004)
        self.bn = layers.BatchNormalization(momentum=0.999)

        # self.mobilenetv2 = MobileNetV2_1()  # MobileNetV2()
        self.backbone = backbone
        self.concat = layers.Concatenate(axis=3)

        self.cpm_stage_blocks = []
        for stage in range(self.number_of_stages):
            if stage == 0:
                cpm_stage_block = CPMStageBlock(kernel_size=3, lastest_channel_size=512,
                                                number_of_keypoints=self.number_of_keypoints)
            else:
                cpm_stage_block = CPMStageBlock(kernel_size=7, lastest_channel_size=128,
                                                number_of_keypoints=self.number_of_keypoints)
            self.cpm_stage_blocks.append(cpm_stage_block)

        self.relu6 = layers.ReLU(max_value=6)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu6(x)

        # x = self.mobilenetv2(x)
        x = self.backbone(x)

        decoder_input = x
        middle_output_layers = []
        for stage, cpm_stage_block in enumerate(self.cpm_stage_blocks):
            if stage != 0:  # if not first stage
                x = self.concat([decoder_input, x])

            x = cpm_stage_block(x)
            middle_output_layers.append(x)

        return middle_output_layers

# Backbone_1과 같은 구조, 필터 갯수
class Backbone_3_1(models.Model):
    def __init__(self):
        super(Backbone_3_1, self).__init__()

        self.front_ib1 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)
        self.front_ib2 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)

        self.branch1 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=18, kernel_size=3)

        self.branch2 = BranchBlock(ibs_is_subsamples=[True, True, False, False, False],
                                   up_channel_rate=6, channels=24, kernel_size=3)

        self.branch3 = BranchBlock(ibs_is_subsamples=[True, True, True, False, False],
                                   up_channel_rate=6, channels=48, kernel_size=3)

        self.branch4 = BranchBlock(ibs_is_subsamples=[True, True, True, True, False],
                                   up_channel_rate=6, channels=72, kernel_size=3)

        self.upsample2 = layers.UpSampling2D(size=(2, 2))
        self.upsample4 = layers.UpSampling2D(size=(4, 4))
        self.upsample8 = layers.UpSampling2D(size=(8, 8))

        self.concat = layers.Concatenate(axis=3)

    def call(self, inputs):
        x = self.front_ib1(inputs)
        x = self.front_ib2(x)

        # print("front_ib2:", x.shape)

        b1 = x
        b1 = self.branch1(b1)
        # print("b1:", b1.shape)

        b2 = self.branch2(x)
        b2 = self.upsample2(b2)
        # print("b2:", b2.shape)

        b3 = self.branch3(x)
        b3 = self.upsample4(b3)
        # print("b3:", b3.shape)

        b4 = self.branch4(x)
        b4 = self.upsample8(b4)
        # print("b4:", b4.shape)

        x = self.concat([b1, b2, b3, b4])

        return x


# maxpool and upsample
class Backbone_3_2(models.Model):
    def __init__(self):
        super(Backbone_3_2, self).__init__()

        self.front_ib1 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)
        self.front_ib2 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)

        self.branch1 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=18, kernel_size=3)

        self.branch2 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=24, kernel_size=3)

        self.branch3 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=48, kernel_size=3)

        self.branch4 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=72, kernel_size=3)

        self.maxpool2 = layers.MaxPool2D(pool_size=(2, 2))
        self.maxpool4 = layers.MaxPool2D(pool_size=(4, 4))
        self.maxpool8 = layers.MaxPool2D(pool_size=(8, 8))
        self.upsample2 = layers.UpSampling2D(size=(2, 2))
        self.upsample4 = layers.UpSampling2D(size=(4, 4))
        self.upsample8 = layers.UpSampling2D(size=(8, 8))

        self.concat = layers.Concatenate(axis=3)

    def call(self, inputs):
        x = self.front_ib1(inputs)
        x = self.front_ib2(x)

        # print("front_ib2:", x.shape)

        b1 = x
        b1 = self.branch1(b1)
        # print("b1:", b1.shape)

        b2 = self.maxpool2(x)
        b2 = self.branch2(b2)
        b2 = self.upsample2(b2)
        # print("b2:", b2.shape)

        b3 = self.maxpool4(x)
        b3 = self.branch3(b3)
        b3 = self.upsample4(b3)
        # print("b3:", b3.shape)

        b4 = self.maxpool8(x)
        b4 = self.branch4(b4)
        b4 = self.upsample8(b4)
        # print("b4:", b4.shape)

        x = self.concat([b1, b2, b3, b4])

        return x


# 채널 갯수를 32로 통일
class Backbone_3_3(models.Model):
    def __init__(self):
        super(Backbone_3_3, self).__init__()

        self.front_ib1 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)
        self.front_ib2 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)

        self.branch1 = BranchBlock(ibs_is_subsamples=[True, False, False, False, False],
                                   up_channel_rate=6, channels=32, kernel_size=3)

        self.branch2 = BranchBlock(ibs_is_subsamples=[True, True, False, False, False],
                                   up_channel_rate=6, channels=32, kernel_size=3)

        self.branch3 = BranchBlock(ibs_is_subsamples=[True, True, True, False, False],
                                   up_channel_rate=6, channels=32, kernel_size=3)

        self.branch4 = BranchBlock(ibs_is_subsamples=[True, True, True, True, False],
                                   up_channel_rate=6, channels=32, kernel_size=3)

        self.upsample2 = layers.UpSampling2D(size=(2, 2))
        self.upsample4 = layers.UpSampling2D(size=(4, 4))
        self.upsample8 = layers.UpSampling2D(size=(8, 8))

        self.concat = layers.Concatenate(axis=3)

    def call(self, inputs):
        x = self.front_ib1(inputs)
        x = self.front_ib2(x)

        # print("front_ib2:", x.shape)

        b1 = x
        b1 = self.branch1(b1)
        # print("b1:", b1.shape)

        b2 = self.branch2(x)
        b2 = self.upsample2(b2)
        # print("b2:", b2.shape)

        b3 = self.branch3(x)
        b3 = self.upsample4(b3)
        # print("b3:", b3.shape)

        b4 = self.branch4(x)
        b4 = self.upsample8(b4)
        # print("b4:", b4.shape)

        x = self.concat([b1, b2, b3, b4])

        return x

class Backbone_3_4(models.Model):
    def __init__(self):
        super(Backbone_3_4, self).__init__()

        self.front_ib1 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)
        self.front_ib2 = InvertedBottleneck(up_channel_rate=1, channels=12, is_subsample=False, kernel_size=3)

        self.branch1 = BranchBlock(ibs_is_subsamples=[True, False, False, False],
                                   up_channel_rate=6, channels=18, kernel_size=3)

        self.branch2 = BranchBlock(ibs_is_subsamples=[True, True, False, False],
                                   up_channel_rate=6, channels=24, kernel_size=3)

        self.branch3 = BranchBlock(ibs_is_subsamples=[True, True, True, False],
                                   up_channel_rate=6, channels=48, kernel_size=3)

        self.branch4 = BranchBlock(ibs_is_subsamples=[True, True, True, True],
                                   up_channel_rate=6, channels=72, kernel_size=3)

        self.upsample2 = layers.UpSampling2D(size=(2, 2))
        self.upsample4 = layers.UpSampling2D(size=(4, 4))
        self.upsample8 = layers.UpSampling2D(size=(8, 8))

        self.concat = layers.Concatenate(axis=3)

    def call(self, inputs):
        x = self.front_ib1(inputs)
        x = self.front_ib2(x)

        # print("front_ib2:", x.shape)

        b1 = x
        b1 = self.branch1(b1)
        # print("b1:", b1.shape)

        b2 = self.branch2(x)
        b2 = self.upsample2(b2)
        # print("b2:", b2.shape)

        b3 = self.branch3(x)
        b3 = self.upsample4(b3)
        # print("b3:", b3.shape)

        b4 = self.branch4(x)
        b4 = self.upsample8(b4)
        # print("b4:", b4.shape)

        x = self.concat([b1, b2, b3, b4])

        return x

if __name__ == '__main__':
    backbone = Backbone_3_1()
    model = ConvolutionalPoseMachine(backbone=backbone,
                                     number_of_keypoints=14,
                                     number_of_stages=4)
    model.build(input_shape=(None, 192, 192, 3))
    model.summary()

    backbone = Backbone_3_2()
    model = ConvolutionalPoseMachine(backbone=backbone,
                                     number_of_keypoints=14,
                                     number_of_stages=4)
    model.build(input_shape=(None, 192, 192, 3))
    model.summary()

    backbone = Backbone_3_3()
    model = ConvolutionalPoseMachine(backbone=backbone,
                                     number_of_keypoints=14,
                                     number_of_stages=4)
    model.build(input_shape=(None, 192, 192, 3))
    model.summary()

    backbone = Backbone_3_4()
    model = ConvolutionalPoseMachine(backbone=backbone,
                                     number_of_keypoints=14,
                                     number_of_stages=4)
    model.build(input_shape=(None, 192, 192, 3))
    model.summary()

