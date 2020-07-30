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

def get_model(model_name, model_subname=None, number_of_keypoints=14, config_extra={}):
    if model_name == "simplepose":
        return _get_simplepose_model(model_subname=model_subname, number_of_keypoints=number_of_keypoints, config_extra=config_extra)
    elif model_name == "cpm":
        return _get_cpm_model(model_subname=model_subname, number_of_keypoints=number_of_keypoints, config_extra=config_extra)
    elif model_name == "hourglass":
        return _get_hourglass_model(model_subname=model_subname, number_of_keypoints=number_of_keypoints, config_extra=config_extra)
    assert False, f"model name is weird: {model_name}"

def _get_simplepose_model(model_subname="", number_of_keypoints=14, config_extra={}):
    from models import simplepose_coco
    if model_subname == "mobilenetv2":
        # mv2_alpha: `0.35`,`0.50`,`0.75`,`1.0`,`1.3`,`1.4`
        model = simplepose_coco.simplepose_mobilenetv2_coco(keypoints=number_of_keypoints, mv2_alpha=1.0)
    elif model_subname == "mv2":
        model = simplepose_coco.simplepose_mv2_coco(keypoints=number_of_keypoints)
    elif model_subname == "resnet18":
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

def _get_cpm_model(model_subname="", number_of_keypoints=14, config_extra={}):
    from models import mv2_cpm
    number_of_stages = config_extra["number_of_stages"]
    return mv2_cpm.ConvolutionalPoseMachine(number_of_keypoints=number_of_keypoints, number_of_stages=number_of_stages)

def _get_hourglass_model(model_subname="", number_of_keypoints=14, config_extra={}):
    from models import mv2_hourglass
    return mv2_hourglass.build_mv2_hourglass_model(number_of_keypoints=number_of_keypoints)

if __name__ == '__main__':
    my_model = get_model(model_name="simplepose", model_subname="mobilenetv2")
    my_model.build(input_shape=(32, 192, 192, 3))
    my_model.heatmap_max_det.build((32, 192, 192, 3))
    my_model.summary()
