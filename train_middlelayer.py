# Copyright 2020 Doyoung Gwak (tucan.dev@gmail.com)
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

from __future__ import absolute_import, division, print_function

import os
import datetime

import tensorflow as tf
import numpy as np

from config.model_config import ModelConfig
from config.train_config import PreprocessingConfig
from config.train_config import TrainConfig

from common import get_time_and_step_interval

print("tensorflow version   :", tf.__version__) # 2.1.0
print("keras version        :", tf.keras.__version__) # 2.2.4-tf

train_config = TrainConfig()
model_config = ModelConfig()
preproc_config = PreprocessingConfig()

train_config.input_size = 192
train_config.output_size = 24
train_config.batch_size = 32

import sys
import getopt
from configparser import ConfigParser

"""
python train_middlelayer.py --dataset_config=config/dataset/coco2017-gpu.cfg --experiment_config=config/training/experiment01.cfg
python train_middlelayer.py --dataset_config=config/dataset/ai_challenger-gpu.cfg --experiment_config=config/training/experiment01.cfg
"""

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "d:e:", ["dataset_config=", "experiment_config="])
except getopt.GetoptError:
    print('train_hourglass.py --dataset_config <inputfile> --experiment_config <outputfile>')
    sys.exit(2)

dataset_config_file_path = "config/dataset/coco2017-gpu.cfg"
experiment_config_file_path = "config/training/experiment01.cfg"
for opt, arg in opts:
    if opt == '-h':
        print('train_middlelayer.py --dataset_config <inputfile> --experiment_config <outputfile>')
        sys.exit()
    elif opt in ("-d", "--dataset_config"):
        dataset_config_file_path = arg
    elif opt in ("-e", "--experiment_config"):
        experiment_config_file_path = arg

parser = ConfigParser()

# get dataset config
print(dataset_config_file_path)
parser.read(dataset_config_file_path)
config_dataset = {}
for key in parser["dataset"]:
    config_dataset[key] = eval(parser["dataset"][key])

# get training config
print(experiment_config_file_path)
parser.read(experiment_config_file_path)
config_model = {}
for key in parser["model"]:
    config_model[key] = eval(parser["model"][key])
config_training = {}
for key in parser["training"]:
    config_model[key] = eval(parser["training"][key])
config_output = {}
for key in parser["output"]:
    config_output[key] = eval(parser["output"][key])

dataset_root_path = config_dataset["dataset_root_path"]  # "/Volumes/tucan-SSD/datasets"
dataset_directory_name = config_dataset["dataset_directory_name"]  # "coco_dataset"
dataset_path = os.path.join(dataset_root_path, dataset_directory_name)

output_root_path = config_output["output_root_path"]  # "/home/outputs"  # "/Volumes/tucan-SSD/ml-project/outputs"
output_experiment_name = config_output["experiment_name"]  # "experiment01"
sub_experiment_name = config_output["sub_experiment_name"]  # "basic"
current_time = datetime.datetime.now().strftime("%m%d%H%M")
model_name = config_model["model_name"]  # "simplepose"
output_name = f"{current_time}_{model_name}_{sub_experiment_name}"
output_path = os.path.join(output_root_path, output_experiment_name, dataset_directory_name)
output_log_path = os.path.join(output_path, "logs", output_name)

# ================================================
# ================= load dataset =================
# ================================================

from data_loader.data_loader import DataLoader

# dataloader instance gen
train_images = config_dataset["train_images"]
train_annotation = config_dataset["train_annotation"]
train_images_dir_path = os.path.join(dataset_path, train_images)
train_annotation_json_filepath = os.path.join(dataset_path, train_annotation)
print(">> LOAD TRAIN DATASET FORM:", train_annotation_json_filepath)
dataloader_train = DataLoader(
    images_dir_path=train_images_dir_path,
    annotation_json_path=train_annotation_json_filepath,
    train_config=train_config,
    model_config=model_config,
    preproc_config=preproc_config)

valid_images = config_dataset["valid_images"]
valid_annotation = config_dataset["valid_annotation"]
valid_images_dir_path = os.path.join(dataset_path, valid_images)
valid_annotation_json_filepath = os.path.join(dataset_path, valid_annotation)
print(">> LOAD VALID DATASET FORM:", valid_annotation_json_filepath)
dataloader_valid = DataLoader(
    images_dir_path=valid_images_dir_path,
    annotation_json_path=valid_annotation_json_filepath,
    train_config=train_config,
    model_config=model_config,
    preproc_config=preproc_config)

number_of_keypoints = dataloader_train.number_of_keypoints  # 17

# train dataset
dataset_train = dataloader_train.input_fn()
dataset_valid = dataloader_valid.input_fn()

# validation images
val_images, val_heatmaps = dataloader_valid.get_images(0, batch_size=25) # from 22 index 6 images and 6 labels

# ================================================
# ============== configure model =================
# ================================================

# from models.mv2_hourglass import build_mv2_hourglass_model
# model = build_mv2_hourglass_model(number_of_keypoints=number_of_keypoints)

from models.mv2_cpm import build_mv2_cpm_model
model = build_mv2_cpm_model(number_of_keypoints=number_of_keypoints)

# model configuration
# model.return_heatmap = True

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(0.001, epsilon=1e-8)
train_loss = tf.keras.metrics.Mean(name="train_loss")
valid_loss = tf.keras.metrics.Mean(name="valid_loss")
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")

# =================================================
# ============== prepare training =================
# =================================================

train_summary_writer = tf.summary.create_file_writer(output_log_path)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        model_output = model(images, training=False)
        # if type(model_output) is list:    
        predictions_layers = model_output

        losses = [loss_object(labels, predictions) for predictions in predictions_layers]
        total_loss = tf.math.add_n(losses)

    max_val = tf.math.reduce_max(predictions_layers[-1])
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(total_loss)
    return total_loss, losses[-1], max_val

from save_result_as_image import save_result_image

def val_step(step, images, heamaps):
    predictions = model(images, training=False)
    predictions = np.array(predictions)
    save_image_results(step, images, heamaps, predictions)

from evaluate import calculate_pckh

def calculate_pckh_on_valid_dataset():
    total_scores = []
    for images, gt_heatmaps in dataset_valid:
        pred_heatmaps_layers = model(images, training=False)

        if type(pred_heatmaps_layers) is list:
            pred_heatmaps = pred_heatmaps_layers[-1]
        else:
            pred_heatmaps = pred_heatmaps_layers

        gt_heatmaps = gt_heatmaps.numpy()
        pred_heatmaps = pred_heatmaps.numpy()

        score = calculate_pckh(gt_heatmaps, pred_heatmaps,
                               batch_size=train_config.batch_size,
                               kp_size=number_of_keypoints,
                               head_index=0, neck_index=1)
        total_scores.append(score)

    total_score = np.mean(total_scores)
    return total_score

@tf.function
def valid_step(images, labels):
    predictions = model(images, training=False)
    v_loss = loss_object(labels, predictions)
    valid_loss(v_loss)
    # valid_accuracy(labels, predictions)
    return v_loss

def save_image_results(step, images, true_heatmaps, predicted_heatmaps):
    val_image_results_directory = "val_image_results"

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, output_name)):
        os.mkdir(os.path.join(output_path, output_name))
    if not os.path.exists(os.path.join(output_path, output_name, val_image_results_directory)):
        os.mkdir(os.path.join(output_path, output_name, val_image_results_directory))

    for i in range(images.shape[0]):
        image = images[i, :, :, :]
        heamap = true_heatmaps[i, :, :, :]
        prediction = predicted_heatmaps[-1][i, :, :, :]

        # result_image = display(i, image, heamap, prediction)
        result_image_path = os.path.join(output_path, output_name, val_image_results_directory, f"result{i}-{step:0>6d}.jpg")
        save_result_image(result_image_path, image, heamap, prediction, title=f"step:{int(step/1000)}k")
        # print("val_step: save result image on \"" + result_image_path + "\"")

def save_model(step=None, label=None):
    saved_model_directory = "saved_model"
    if step is not None:
        saved_model_directory = saved_model_directory + f"-{step}"
    if label is not None:
        saved_model_directory = saved_model_directory + "-" + label

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, output_name)):
        os.mkdir(os.path.join(output_path, output_name))
    if not os.path.exists(os.path.join(output_path, output_name, saved_model_directory)):
        os.mkdir(os.path.join(output_path, output_name, saved_model_directory))

    saved_model_path = os.path.join(output_path, output_name, saved_model_directory)

    print("-"*20 + " MODEL SAVE!! " + "-"*20)
    print("saved model path: " + saved_model_path)
    model.save(saved_model_path)
    print("-"*18 + " MODEL SAVE DONE!! " + "-"*18)

if __name__ == '__main__':
    # ================================================
    # ============= load hyperparams =================
    # ================================================
    # config_dataset = ...
    # config_model = ...
    # config_output = ...

    # ================================================
    # =============== load dataset ===================
    # ================================================
    # dataset_train = ...
    # dataset_valid = ...
    # dataset_test = ...

    # ================================================
    # =============== build model ====================
    # ================================================
    # model = ...
    # model.summary()

    # ================================================
    # ============== train the model =================
    # ================================================
    num_epochs = 1000
    step = 1
    number_of_echo_period = 100
    number_of_validimage_period = 100000  # 1000
    number_of_modelsave_period = 1000
    tensorbaord_period = 10
    validation_period = 10000  # 1000
    valid_check = False

    # TRAIN!!
    get_time_and_step_interval(step, is_init=True)

    for epoch in range(num_epochs):
        print("-" * 10 + " " + str(epoch + 1) + " EPOCH " + "-" * 10)
        for images, heatmaps in dataset_train:

            # print(images.shape)  # (32, 128, 128, 3)
            # print(heatmaps.shape)  # (32, 32, 32, 17)
            total_loss, last_layer_loss, max_val = train_step(images, heatmaps)

            step += 1

            if number_of_echo_period is not None and step % number_of_echo_period == 0:
                total_interval, per_step_interval = get_time_and_step_interval(step)
                echo_textes = []
                if step is not None:
                    echo_textes.append(f"step: {step}")
                if total_interval is not None:
                    echo_textes.append(f"total: {total_interval}")
                if per_step_interval is not None:
                    echo_textes.append(f"per_step: {per_step_interval}")
                if total_loss is not None:
                    echo_textes.append(f"total loss: {total_loss:.6f}")
                if last_layer_loss is not None:
                    echo_textes.append(f"last loss: {last_layer_loss:.6f}")
                print(">> " + ", ".join(echo_textes))

            # validation phase
            if number_of_validimage_period is not None and step % number_of_validimage_period == 0:
                val_step(step, val_images, val_heatmaps)

            if number_of_modelsave_period is not None and step % number_of_modelsave_period == 0:
                save_model(step=step)

            if tensorbaord_period is not None and step % tensorbaord_period == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("total_loss", total_loss.numpy(), step=step)
                    tf.summary.scalar("max_value - last_layer_loss", max_val.numpy(), step=step)
                    if last_layer_loss is not None:
                        tf.summary.scalar("last_layer_loss", last_layer_loss.numpy(), step=step)

            if validation_period is not None and step % validation_period == 0:
                # print("calcuate pckh")
                pckh_score = calculate_pckh_on_valid_dataset()
                with train_summary_writer.as_default():
                    tf.summary.scalar('pckh_score', pckh_score, step=step)
                print(f"calcuate pckh done: {pckh_score}")

        # if not valid_check:
        #     continue

        # for v_images, v_heatmaps in dataloader_valid:
        #     v_loss = valid_step(v_images, v_heatmaps)



    # last model save
    save_model(step=step, label="final")

    # last pckh
    pckh_score = calculate_pckh_on_valid_dataset()
    with train_summary_writer.as_default():
        tf.summary.scalar('pckh_score', pckh_score, step=step)
    print(f"calcuate pckh done: {pckh_score}")