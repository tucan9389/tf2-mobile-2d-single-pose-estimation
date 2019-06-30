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
#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
# import tensorflow.contrib.slim as slim
import json




class ModelConfig(object):

    def __init__(self, setuplog_dir):

        self.input_size   = 128
        self.output_size  = 32 # it will be changed later

        self.input_chnum   = 3
        self.output_chnum  = 14 # number of keypoints
        self.channel_num   = 96

        """
        # pre-trained model list: https://keras.io/applications
        # MobileNetV2, ResNet50,...
        
        'mnv2'      MobileNetV2     14 MB	0.713	0.901	3,538,984	88
        'mn'        MobileNet       16 MB	0.704	0.895	4,253,864	88
        'nnm'       NASNetMobile    23 MB	0.744	0.919	5,326,716	-
        'dn121'     DenseNet121     33 MB	0.750	0.923	8,062,504	121
        'dn169'     DenseNet169     57 MB	0.762	0.932	14,307,880	169
        
        'dn201'     DenseNet201     80 MB	0.773	0.936	20,242,984	201
        'xp'        Xception        88 MB
        'rnxt50'    ResNeXt50       96 MB	0.777	0.938	25,097,128	-
        'ipv3'      InceptionV3     92 MB	0.779	0.937	23,851,784	159
        'rn50'      ResNet50        98 MB	0.749	0.921	25,636,712	-
        'rn50v2'    ResNet50V2      98 MB	0.760	0.930	25,613,800	-
                
        'rnxt101'   ResNeXt101      170 MB	0.787	0.943	44,315,560	-
        'rn101'     ResNet101	    171 MB	0.764	0.928	44,707,176	-
        'rn101v2'   ResNet101V2	    171 MB	0.772	0.938	44,675,560	-
        """
        self.base_model_name  = 'mnv2'
        self.reception      = RecepConfig()
        self.hourglass      = HourglassConfig(channel_num=self.channel_num)
        self.output         = OutputConfig()
        self.separable_conv = SeparableConfig()

        self.dtype = tf.float32

        self.decoder_layers = [
            {"filters": 64, "kernel_size": 4, "strides": (4, 4)},
            {"filters": 64, "kernel_size": 4, "strides": (3, 3)},
            {"filters": 64, "kernel_size": 4, "strides": (2, 2)},
        ]
        # self.output_size = 5
        # for layer in self.decoder_layers:
        #     w_stride, h_stride = layer["strides"]
        #     self.output_size = self.output_size * w_stride
        # print(self.output_size)

        # for i in range(len(self.decoder_layers)):
        #     if i == 0:
        #         self.filter_name = str(self.decoder_layers[i]["filters"])
        #     else:
        #         self.filter_name = self.filter_name + "x" + str(self.decoder_layers[i]["filters"])


        # model config logging
        if setuplog_dir is not None:
            self.model_config_dict      = self.__dict__
            self.reception_config_dict  = self.reception.__dict__
            self.hourglass_config_dict  = self.hourglass.__dict__
            self.output_config_dict     = self.output.__dict__

            model_config_filename       = setuplog_dir + 'model_config.json'
            reception_config_filename   = setuplog_dir + 'recept_config.json'
            hourglass_config_filename   = setuplog_dir + 'hourglass_config.json'
            output_config_filename      = setuplog_dir + 'output_config.json'

            # with open(model_config_filename,'w') as fp:
            #     json.dump(str(self.model_config_dict), fp)
            #
            # with open(reception_config_filename,'w') as fp:
            #     json.dump(str(self.reception_config_dict),fp)
            #
            # with open(hourglass_config_filename,'w') as fp:
            #     json.dump(str(self.hourglass_config_dict),fp)
            #
            # with open(output_config_filename,'w') as fp:
            #     json.dump(str(self.output_config_dict),fp)




class RecepConfig(object):

    def __init__(self):
        # batch norm config
        self.batch_norm_decay   =  0.999
        self.batch_norm_fused   =  True

        # self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        # self.biases_initializer     = slim.init_ops.zeros_initializer()
        # self.weights_regularizer    = None
        #
        # self.activation_fn          = tf.nn.relu
        # self.normalizer_fn          = slim.batch_norm
        self.is_trainable           = True

        self.kernel_shape ={\
            'r1': [7,7],
            'r4': [3,3]
            }

        self.strides = {\
            'r1': 2,
            'r4': 2
            }





class HourglassConfig(object):

    def __init__(self,channel_num):
        self.updown_rate            = 2
        self.maxpool_kernel_size    =[3,3]
        self.num_stage              = 4
        self.center_conv_num        = 1
        self.skip_invbottle_num          = 3
        self.center_ch_num          = channel_num #output channel num
        # self.center_ch_num          = 14 #output channel num





class OutputConfig(object):

    def __init__(self):

        # batch norm config
        self.batch_norm_decay   =  0.999
        self.batch_norm_fused   =  True

        self.dropout_keeprate       = 1.0
        # self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        # self.weights_regularizer    = tf.contrib.layers.l2_regularizer(4E-5)
        # self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.weights_regularizer    = None
        self.activation_fn          = None
        self.is_trainable           = True
        # self.normalizer_fn          = slim.batch_norm

        self.kernel_shape   = [1,1]
        self.stride         = 1





class SeparableConfig(object):

    def __init__(self):
        # batch norm config
        self.batch_norm_decay   =  0.999
        self.batch_norm_fused   =  True

        # self.weights_initializer    = tf.contrib.layers.xavier_initializer()
        # self.biases_initializer     = slim.init_ops.zeros_initializer()
        self.weights_regularizer    = None
        self.invbottle_expansion_rate   = 7.0

        # self.normalizer_fn          = slim.batch_norm
        self.is_trainable           = True

        self.activation_fn_dwise = None
        # self.activation_fn_pwise = tf.nn.relu

        self.kernel_shape_dwise =[3,3]
        self.stride_dwise       = 1