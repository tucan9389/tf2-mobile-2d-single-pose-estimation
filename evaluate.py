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
import math

# import tensorflow as tf
import numpy as np

# evaludate
# v 고려 안 함
def calculate_pckh(gt_heatmaps, pred_heatmaps, batch_size=32, kp_size=14, head_index=0, neck_index=1):
    heatmap_w_h = pred_heatmaps.shape[1]

    batch_scores = []
    for i in range(batch_size):
        b_gt_heatmaps = gt_heatmaps[i]
        b_pred_heatmaps = pred_heatmaps[i]

        # threshold
        kp0_heatmap = b_gt_heatmaps[:, :, head_index]
        kp1_heatmap = b_gt_heatmaps[:, :, neck_index]
        kp0 = [(kp0_heatmap[0] + 0.5) / heatmap_w_h, (kp0_heatmap[1] + 0.5) / heatmap_w_h]
        kp1 = [(kp1_heatmap[0] + 0.5) / heatmap_w_h, (kp1_heatmap[1] + 0.5) / heatmap_w_h]
        threshold_dist = np.linalg.norm(np.array(kp0) - np.array(kp1))

        scores = []
        for kp_index in range(kp_size):
            b_gt_heatmap = b_gt_heatmaps[:, :, kp_index]
            b_pred_heatmap = b_pred_heatmaps[:, :, kp_index]

            gt_kp = np.unravel_index(np.argmax(b_gt_heatmap), b_gt_heatmap.shape)
            pred_kp = np.unravel_index(np.argmax(b_pred_heatmap), b_pred_heatmap.shape)

            # print(kp_index, "gt_kp  :", gt_kp)
            # print(kp_index, "pred_kp:", pred_kp)

            gt_coord = [(gt_kp[0] + 0.5) / heatmap_w_h, (gt_kp[1] + 0.5) / heatmap_w_h]
            pred_coord = [(pred_kp[0] + 0.5) / heatmap_w_h, (pred_kp[1] + 0.5) / heatmap_w_h]

            dist = np.linalg.norm(np.array(gt_coord) - np.array(pred_coord))

            if dist <= threshold_dist:
                scores.append(1.0)
            else:
                scores.append(0.0)
        score = np.mean(scores)
        batch_scores.append(score)
    batch_score = np.mean(batch_scores)
    return batch_score

if __name__ == '__main__':
    from config.model_config import ModelConfig
    from config.train_config import PreprocessingConfig
    from config.train_config import TrainConfig

    # ================================================
    # ================= load dataset =================
    # ================================================

    from data_loader.data_loader import DataLoader

    train_config = TrainConfig()
    model_config = ModelConfig()
    preproc_config = PreprocessingConfig()

    train_config.input_size = 192
    train_config.output_size = 48
    train_config.batch_size = 32

    heatmap_w_h = train_config.output_size

    dataset_path = "/Volumes/tucan-SSD/datasets/ai_challenger"  # "/home/datasets/ai_challenger"

    # dataloader instance gen
    valid_images_dir_path = os.path.join(dataset_path, "valid/images")
    valid_annotation_json_filepath = os.path.join(dataset_path, "valid/annotation.json")
    print(">> LOAD VALID DATASET FORM:", valid_annotation_json_filepath)
    dataloader_valid = DataLoader(
        images_dir_path=valid_images_dir_path,
        annotation_json_path=valid_annotation_json_filepath,
        train_config=train_config,
        model_config=model_config,
        preproc_config=preproc_config)

    number_of_keypoints = dataloader_valid.number_of_keypoints # 17

    # train dataset
    dataset_valid = dataloader_valid.input_fn()

    # model
    from models.mv2_hourglass import build_mv2_hourglass_model
    model = build_mv2_hourglass_model(number_of_keypoints=number_of_keypoints)
    # model.load(...)

    head_index = 0
    neck_index = 1

    print("start iterate")

    total_scores = []
    for images, gt_heatmaps in dataset_valid:
        pred_heatmaps_layers = model(images)
        pred_heatmaps = pred_heatmaps_layers[-1]

        gt_heatmaps = gt_heatmaps.numpy()
        pred_heatmaps = pred_heatmaps.numpy()

        score = calculate_pckh(gt_heatmaps, pred_heatmaps,
                               batch_size=train_config.batch_size,
                               kp_size=number_of_keypoints,
                               head_index=head_index, neck_index=neck_index)
        total_scores.append(score)

    total_score = np.mean(total_scores)

    print(f"total_score: {total_score}")