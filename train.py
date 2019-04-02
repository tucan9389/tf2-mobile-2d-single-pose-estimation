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
from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import sys
from os import getcwd
from datetime import datetime

from path_manager import PROJ_HOME

import configparser

from path_manager import TF_MODULE_DIR
from path_manager import EXPORT_DIR
from path_manager import COCO_DATALOAD_DIR
from path_manager import DATASET_DIR


from model_config import ModelConfig
from train_config import PreprocessingConfig
from train_config import TrainConfig

from data_loader   import DataLoader

from hourglass_model import HourglassModelBuilder


print("tensorflow version   :", tf.__version__)
print("keras version        :", tf.keras.__version__)



def main():

    sys.path.insert(0, TF_MODULE_DIR)
    sys.path.insert(0, EXPORT_DIR)
    sys.path.insert(0, COCO_DATALOAD_DIR)

    # # configuration file
    # config = configparser.ConfigParser()
    #
    # config_file = "mv2_cpm.cfg"
    # if os.path.exists(config_file):
    #     config.read(config_file)

    # params = {}
    # for _ in config.options("Train"):
    #     params[_] = eval(config.get("Train", _))
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = params['visible_devices']



    train_config    = TrainConfig()
    model_config    = ModelConfig(setuplog_dir = train_config.setuplog_dir)
    preproc_config  = PreprocessingConfig(setuplog_dir = train_config.setuplog_dir)



    # ================================================
    # =============== dataset pipeline ===============
    # ================================================

    # dataloader instance gen
    dataloader_train, dataloader_valid = \
    [DataLoader(
    is_training     =is_training,
    data_dir        =DATASET_DIR,
    transpose_input =False,
    train_config    =train_config,
    model_config    =model_config,
    preproc_config  =preproc_config,
    use_bfloat16    =False) for is_training in [True, False]]


    dataset_train   = dataloader_train.input_fn()
    # dataset_valid   = dataloader_valid.input_fn()

    data = dataset_train.repeat()
    # data = dataset_train


    # ================================================
    # ============== configure model =================
    # ================================================

    model_builder = HourglassModelBuilder()
    model_builder.build_model()

    model = model_builder.model
    model.summary()

    model.compile(optimizer=tf.optimizers.Adam(0.001, epsilon=1e-8),#'adam',
                  loss=tf.losses.MeanSquaredError(),
                  metrics=['accuracy'])#tf.metrics.Accuracy

    # ================================================
    # =============== setup output ===================
    # ================================================
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = os.path.join(PROJ_HOME, "outputs")

    # output model file(.hdf5)
    model_path = os.path.join(output_path, "models")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    checkpoint_path = os.path.join(model_path, "hg_" + current_time + ".hdf5") #".ckpt"
    check_pointer = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                       save_weights_only=False,
                                                       verbose=1)
    # output tensorboard log
    log_path = os.path.join(output_path, "logs")
    log_path = os.path.join(log_path, "hg_" + current_time)
    tensorboard = tf.keras.callbacks.TensorBoard(log_path)


    # ================================================
    # ==================== train! ====================
    # ================================================

    model.fit(data,
              epochs=300,
              steps_per_epoch=100,
              callbacks=[check_pointer, tensorboard]) # steps_per_epoch=100,

    # ================================================
    # =================== evaluate ===================
    # ================================================





if __name__ =='__main__':
    main()