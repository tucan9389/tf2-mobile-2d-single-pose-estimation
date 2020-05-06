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
# import keras
from datetime import datetime
# from config.path_manager import EXPORT_DIR
# from config.path_manager import LOCAL_LOG_DIR
from subprocess import check_output


class TrainConfig(object):

    def __init__(self):
        self.epochs = 100
        self.epochs_finetuning = 1000
        self.steps_per_epoch = 100
        self.steps_per_epoch_finetuning = 100

        self.learning_rate = 1e-4
        self.learning_rate_decay_step = 2000
        self.learning_rate_decay_rate = 0.95
        self.learning_rate_finetuning = 1e-5
        self.opt_fn = tf.keras.optimizers.Adam
        self.loss_fn = tf.nn.l2_loss
        self.batch_size = 32  # 8
        self.shuffle_size = 1024
        # self.prefetch_size              = 1024
        self.metric_fn = tf.keras.metrics.mae

        # the number of step between evaluation
        self.display_step = 100
        self.ckpt_step = 100

        self.train_data_size = 22000
        self.valid_data_size = 1500
        self.eval_data_size = 1500

        self.total_steps = int(float(self.train_data_size / self.batch_size) * 20.0)
        self.multiprocessing_num = 4
        self.random_seed = 66478

        # tensorboard config
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        # self.root_logdir = EXPORT_DIR
        # self.local_logdir = LOCAL_LOG_DIR
        self.is_summary_heatmap = True

        # self.tflogdir = "{}/run-{}/".format(self.root_logdir + '/tf_logs', now)
        # self.ckpt_dir = self.tflogdir + 'pb_and_ckpt/'

        # self.setuplog_dir = "{}/run-{}/".format(self.local_logdir + '/train_setup_log', now)

        # print('[train_config] tflog    dir = %s' % self.tflogdir)
        # print('[train_config] setuplog dir = %s' % self.setuplog_dir)
        self.train_config_dict = self.__dict__

        # if not tf.gfile.Exists(self.setuplog_dir):
        #     tf.gfile.MakeDirs(self.setuplog_dir)
        # train_config_filename = self.setuplog_dir + 'train_config.json'

        # with open(train_config_filename,'w') as fp:
        #     json.dump(str(self.train_config_dict),fp)

    def send_setuplog_to_gcp_bucket(self):

        try:
            cmd = "sudo gsutil cp -r {} {}".format(self.setuplog_dir + '* ', self.tflogdir)
            print('[main] cmd=%s' % cmd)
            check_output(cmd, shell=True)
            # tf.logging.info('[main] success logging config in bucket')
        except:
            # tf.logging.info('[main] failure logging config in bucket')
            print('[main] failure logging config in bucket')


class PreprocessingConfig(object):

    def __init__(self):
        # image pre-processing
        self.is_crop = False
        self.is_rotate = True
        self.is_flipping = False
        self.is_scale = False
        self.is_resize_shortest_edge = True

        # this is when classification task
        # which has an input as pose coordinate
        # self.is_label_coordinate_norm   = False

        # for ground true heatmap generation
        self.heatmap_std = 5.0

        self.MIN_AUGMENT_ROTATE_ANGLE_DEG = -1.0
        self.MAX_AUGMENT_ROTATE_ANGLE_DEG = 1.0

        # For normalize the image to zero mean and unit variance.
        self.MEAN_RGB = [0.485, 0.456, 0.406]
        self.STDDEV_RGB = [0.229, 0.224, 0.225]

        # if setuplog_dir is not None:
        #     preproc_config_dict = self.__dict__
        #     preproc_config_filename = setuplog_dir + 'preproc_config.json'
        #
        #     # with open(preproc_config_filename,'w') as fp:
        #     #     json.dump(str(preproc_config_dict),fp)

    def show_info(self):
        # tf.logging.info('------------------------')
        # tf.logging.info('[train_config] Use is_crop: %s'        % str(self.is_crop))
        # tf.logging.info('[train_config] Use is_rotate  : %s'    % str(self.is_rotate))
        # tf.logging.info('[train_config] Use is_flipping: %s'    % str(self.is_flipping))
        # tf.logging.info('[train_config] Use is_scale: %s'       % str(self.is_scale))
        # tf.logging.info('[train_config] Use is_resize_shortest_edge: %s' % str(self.is_resize_shortest_edge))

        # if self.is_rotate:
        #     tf.logging.info('[train_config] MIN_ROTATE_ANGLE_DEG: %s' % str(self.MIN_AUGMENT_ROTATE_ANGLE_DEG))
        #     tf.logging.info('[train_config] MAX_ROTATE_ANGLE_DEG: %s' % str(self.MAX_AUGMENT_ROTATE_ANGLE_DEG))
        # tf.logging.info('[train_config] Use heatmap_std: %s'    % str(self.heatmap_std))
        # tf.logging.info('------------------------')
        print("---- show info ----")
