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

# import os
# import tensorflow as tf
# import tensorflow.keras.losses
# tensorflow.keras.losses.custom_loss = tf.losses.MeanSquaredError()
# import coremltools

# from path_manager import PROJ_HOME

# output_path = os.path.join(PROJ_HOME, "outputs")
# model_path = os.path.join(output_path, "models", "hg_20190402082910.hdf5")

# print(model_path)
# model = tf.keras.models.load_model(model_path)

import coremltools as ct
import os

model_path = "/home/centos/projects/doyoung/tf2-mobile-pose-estimation/outputs/receipt-experiment004/receipt_top_bottom_kp4_5982/03190230_cpm_sg5/saved_model-030000"
output_mlmodel_path = "/home/centos/projects/doyoung/tf2-mobile-pose-estimation/outputs/receipt-experiment004/receipt_top_bottom_kp4_5982/03190230_cpm_sg5/coreml"

# ============================================================
# ============================================================
# ============================================================

image_input = ct.ImageType(shape=(1, 192, 192, 3,),
                           scale=1)
heatmap_output = ct.ImageType
model = ct.convert(
    model_path,
    inputs=[image_input],
    # output_feature_names=['Identity']
    # minimum_deployment_target=ct.target.iOS13,
)

if not os.path.exists(output_mlmodel_path):
    os.mkdir(output_mlmodel_path)
model.save(os.path.join(output_mlmodel_path, "03190230_cpm_sg5_030000_noscale.mlmodel"))


# ============================================================
# ============================================================
# ============================================================

image_input = ct.ImageType(shape=(1, 192, 192, 3,),
                           scale=1/255)
heatmap_output = ct.ImageType
model = ct.convert(
    model_path,
    inputs=[image_input],
    # output_feature_names=['Identity']
    # minimum_deployment_target=ct.target.iOS13,
)

if not os.path.exists(output_mlmodel_path):
    os.mkdir(output_mlmodel_path)
model.save(os.path.join(output_mlmodel_path, "03190230_cpm_sg5_030000_scale_div_255.mlmodel"))

# ============================================================
# ============================================================
# ============================================================
