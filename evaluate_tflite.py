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


import numpy as np
import tensorflow as tf
tf.random.set_seed(3)

# tflite_model_path = "/Volumes/tucan-SSD/ml-project/experime
# nt001/ai_challenger/06022331_mv2_hourglass_basic/tflite/mv2_cpm-249000.tflite"
# input_index = 0
# output_index = 3

class TFLiteModel:
    def __init__(self, tflite_model_path, input_index=0, output_index=-1):
        self.tflite_model_path = tflite_model_path
        self.input_index = input_index
        self.output_index = output_index

        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Test the model on random input data.
        self.input_shape = self.input_details[input_index]['shape']

        self.output_shape = self.output_details[output_index]['shape']

        # for output_detail in self.output_details:
        #     print(output_detail)

        print("model loaded")
        print(self.interpreter.get_input_details())
        print(self.output_details[output_index])
        print("output_details.name:", self.output_details[output_index]['name'])
        print("output_details.shape:", self.output_details[output_index]['shape'])

    def inference(self, input_data):
        # print("input_shape == input_data.shape:", self.input_shape == input_data.shape)

        self.interpreter.set_tensor(self.input_details[self.input_index]['index'], input_data)

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(self.output_details[self.output_index]['index'])

        return output_data


