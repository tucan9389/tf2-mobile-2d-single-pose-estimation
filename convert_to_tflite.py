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

import os

def save_tflite(saved_model_path, tflite_model_path=None):
    if tflite_model_path is None:
        # Make tflite dir
        tflite_model_dir_path = os.path.join(os.path.dirname(saved_model_path), 'tflite')
        if not os.path.exists(tflite_model_dir_path):
            os.mkdir(tflite_model_dir_path)
        # tflite file
        filename = saved_model_path.split('/')[-1]
        filename = filename.split('.')[0]
        step = filename.split('-')[-1]
        model_name = saved_model_path.split('/')[-2]
        tflite_filename = f'{model_name}-{step}.tflite'
        tflite_model_path = os.path.join(tflite_model_dir_path, tflite_filename)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    # Save the TF Lite model.
    with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f'Saved TFLite on: {tflite_model_path}')
    return tflite_model_path