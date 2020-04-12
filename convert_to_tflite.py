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

import os
import tensorflow as tf




# ------------------------------------------------------
# ----------------- YOU MUST CHANGE --------------------
trained_model_file_name = "hg_1e9_20190403204228.hdf5"
# ------------------------------------------------------
# ------------------------------------------------------

saved_model_path = "/Users/doyounggwak/Google Drive/ml-project/outputs/03060816_sp-receipt_3307/saved_model"
tflite_model_directory_path = "/Users/doyounggwak/Google Drive/ml-project/outputs/03060816_sp-receipt_3307/tflite/"
if not os.path.exists(tflite_model_directory_path):
    os.mkdir(tflite_model_directory_path)
tflite_model_path = os.path.join(tflite_model_directory_path, "03060816_sp-receipt_3307.tflite")

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()
open(tflite_model_path, "wb").write(tflite_model)


interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()

# Print input shape and type
print(interpreter.get_input_details()[0]['shape'])  # Example: [1 224 224 3]
print(interpreter.get_input_details()[0]['dtype'])  # Example: <class 'numpy.float32'>

# Print output shape and type
print(interpreter.get_output_details()[0]['shape'])  # Example: [1 1000]
print(interpreter.get_output_details()[0]['dtype'])  # Example: <class 'numpy.float32'>


# def convert_model(model, model_file_path):
#     print('converting...')
#
#     # file path
#     file_name = os.path.splitext(os.path.basename(model_file_path))[0]
#     tflite_model_path = os.path.join(model_path, "tflite")
#     if not os.path.exists(tflite_model_path):
#         os.mkdir(tflite_model_path)
#         print("Create TFLite model directory:", tflite_model_path)
#     tflite_model_file_path = os.path.join(tflite_model_path, file_name + '.tflite')
#     print("TFLite model path:", tflite_model_file_path)
#
#     # Get the concrete function from the Keras model.
#     run_model = tf.function(lambda x: model(x))
#
#     # Save the concrete function.
#     concrete_func = run_model.get_concrete_function(
#         tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
#
#     # Create converter with concrete function.
#     converter = tf.lite.TFLiteConverter.from_concrete_function(concrete_func)
#     #converter.post_training_quantize = True
#
#     # Convert!
#     tflite_model = converter.convert()
#
#     # Save tflite file
#     file = open(tflite_model_file_path, 'wb')
#     file.write(tflite_model)
#
#     print('end of converting')
#
#
# # output_path = os.path.join(PROJ_HOME, "outputs")
# output_path = ""
# model_path = os.path.join(output_path, "models")
# model_file_path = os.path.join(model_path, trained_model_file_name)
#
# print("Model path:", model_path)
#
# if os.path.isfile(model_file_path):
#     print(model_file_path)
#     # model = load_model(model_file_path)
#
#     # model_builder = HourglassModelBuilder()
#     # model_builder.build_model()
#
#     model = model_builder.model
#     model.load_weights(model_file_path)
#     convert_model(model, model_file_path)
#
# else:
#     print('no model found')