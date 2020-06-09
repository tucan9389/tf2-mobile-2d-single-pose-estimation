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

"""
- "simplepose"
- "mv2_cpm"
- "mv2_hourglass"
"""

def get_model(model_name, model_subname=None, number_of_keypoints=14):
    if model_name == "simplepose":
        return _get_simplepose_model(model_subname=model_subname, number_of_keypoints=number_of_keypoints)
    elif model_name == "cpm":
        return _get_cpm_model(model_subname=model_subname, number_of_keypoints=number_of_keypoints)
    elif model_name == "hourglass":
        return _get_hourglass_model(model_subname=model_subname, number_of_keypoints=number_of_keypoints)
    assert False, f"model name is weird: {model_name}"

def _get_simplepose_model(model_subname="", number_of_keypoints=14):
    from models import simplepose_coco
    if model_subname == "resnet18":
        model = simplepose_coco.simplepose_resnet18_coco(keypoints=number_of_keypoints)
    elif model_subname == "resnet50b":
        model = simplepose_coco.simplepose_resnet50b_coco(keypoints=number_of_keypoints)
    elif model_subname == "resnet101b":
        model = simplepose_coco.simplepose_resnet101b_coco(keypoints=number_of_keypoints)
    elif model_subname == "resnet152b":
        model = simplepose_coco.simplepose_resnet152b_coco(keypoints=number_of_keypoints)
    elif model_subname == "resneta50b":
        model = simplepose_coco.simplepose_resneta50b_coco(keypoints=number_of_keypoints)
    elif model_subname == "resnet101b":
        model = simplepose_coco.simplepose_resneta101b_coco(keypoints=number_of_keypoints)
    elif model_subname == "resneta152b":
        model = simplepose_coco.simplepose_resneta152b_coco(keypoints=number_of_keypoints)
    else:
        model = simplepose_coco.simplepose_resnet18_coco(keypoints=number_of_keypoints)
    model.return_heatmap = True
    return model

def _get_cpm_model(model_subname="", number_of_keypoints=14):
    from models import mv2_cpm
    return mv2_cpm.build_mv2_cpm_model(number_of_keypoints=number_of_keypoints)

def _get_hourglass_model(model_subname="", number_of_keypoints=14):
    from models import mv2_hourglass
    return mv2_hourglass.build_mv2_hourglass_model(number_of_keypoints=number_of_keypoints)