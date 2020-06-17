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

import tensorflow as tf

saved_model_path = "/Volumes/tucan-SSD/ml-project/simplepose/outputs/05312106_sp-ai_challenger/saved_model-5"
tflite_model_dir = "/Volumes/tucan-SSD/ml-project/simplepose/outputs/05312106_sp-ai_challenger/tflite"
tflite_model_filename = "mv2_hourglass-5.tflite"

# from models.mv2_hourglass import build_mv2_hourglass_model
# model = build_mv2_hourglass_model(number_of_keypoints=17)
# model.summary()

model = tf.keras.models.load_model(saved_model_path)
model.summary()

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

import os

if not os.path.exists(tflite_model_dir):
    os.mkdir(tflite_model_dir)

open(os.path.join(tflite_model_dir, tflite_model_filename), "wb").write(tflite_model)


"""
tflite_convert \
  --saved_model_dir=/Volumes/tucan-SSD/ml-project/simplepose/outputs/05312106_sp-ai_challenger/saved_model-5 \
  --output_file=/Volumes/tucan-SSD/ml-project/simplepose/outputs/05312106_sp-ai_challenger/tflite/mv2_hourglass-5.tflite
"""