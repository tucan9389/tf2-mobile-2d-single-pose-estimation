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
from datetime import datetime

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





sys.path.insert(0, TF_MODULE_DIR)
sys.path.insert(0, EXPORT_DIR)
sys.path.insert(0, COCO_DATALOAD_DIR)

# configuration file
config = configparser.ConfigParser()

config_file = "mv2_cpm.cfg"
if os.path.exists(config_file):
    config.read(config_file)

# params = {}
# for _ in config.options("Train"):
#     params[_] = eval(config.get("Train", _))
#
# os.environ['CUDA_VISIBLE_DEVICES'] = params['visible_devices']



train_config    = TrainConfig()
model_config    = ModelConfig(setuplog_dir = train_config.setuplog_dir)
preproc_config  = PreprocessingConfig(setuplog_dir = train_config.setuplog_dir)


#
# # path
# if not os.path.exists(params['modelpath']):
#     os.makedirs(params['modelpath'])
# if not os.path.exists(params['logpath']):
#     os.makedirs(params['logpath'])





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
dataset_valid   = dataloader_valid.input_fn()

#data = dataset_train.repeat()
data = dataset_train

# dataloader_train._parse_function(3)

# dataset_train_iterator  = dataset_train.make_one_shot_iterator()
# dataset_valid_iterator  = dataset_valid.make_one_shot_iterator()
#
# dataset_iterator    = tf.data.Iterator.from_string_handle(dataset_handle,
#                                                        dataset_train.output_types,
#                                                        dataset_train.output_shapes)
# inputs, true_heatmap =  dataset_iterator.get_next()

# print(data.next())

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

current_time = datetime.now().strftime("%Y%m%d%H%M%S")
# model_path = 'hourglass' + ".h5"
# print(model_path)
# model.save(model_path)
# model.save_weights(model_path)

# checkpoint_path = "./outputs/hourglass-" + current_time + ".hdf5"#"".ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

output_path = "outputs/"
save_path = output_path + "models/" + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'
check_pointer = tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, verbose=1)

log_path = output_path + "logs/" + "hourglass-" + current_time
tensorboard = tf.keras.callbacks.TensorBoard(log_path)



# ================================================
# ==================== train! ====================
# ================================================

model.fit(data, epochs=5, callbacks=[check_pointer, tensorboard])

# ================================================
# =================== evaluate ===================
# ================================================


# if __name__ =='__main__':
#     main()