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
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython.display import Image

def get_bgimg(inp, target_size=None):
    inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if target_size:
        inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
    return inp


# filepath: GOOGLEDRIVE_PATH/output_directory/{modeling_name}/result{index}-{step}.jpg
# filepath: GOOGLEDRIVE_PATH/output_directory/{modeling_name}/result{20}-{14000}.jpg
def save_result_image(filepath, inp_image, true_heat=None, pred_heat=None):
    fig = plt.figure()

    if true_heat is not None:
        a = fig.add_subplot(1, 2, 1)
        a.set_title('True Heatmap')
        plt.imshow(get_bgimg(inp_image, target_size=(true_heat.shape[1], true_heat.shape[0])), alpha=0.5)
        tmp = np.amax(true_heat, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.7)
        plt.colorbar()

    if pred_heat is not None:
        a = fig.add_subplot(1, 2, 2)
        a.set_title('Pred Heatmap')
        plt.imshow(get_bgimg(inp_image, target_size=(pred_heat.shape[1], pred_heat.shape[0])), alpha=0.5)
        tmp = np.amax(pred_heat, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=1, vmin=0.0, vmax=1.0)

        plt.colorbar()

    fig.savefig(filepath)