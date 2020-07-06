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
# from scipy.ndimage.filters import gaussian_filter

import tensorflow as tf
tf.random.set_seed(3)
import numpy as np

def convert_heatmap_to_keypoint(heatmap, image_size):
    # heatmap = gaussian_filter(heatmap, sigma=5)
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x_idx = idx[1] / heatmap.shape[1]
    y_idx = idx[0] / heatmap.shape[0]
    return int(x_idx * image_size[1]), int(y_idx * image_size[0])  # exchange y, x sequence

def convert_heatmaps_to_keypoints(heatmaps, image_size):
    kp_num = heatmaps.shape[-1]
    return [convert_heatmap_to_keypoint(heatmaps[:, :, kp_index], image_size) for kp_index in range(kp_num)]

head_index = 0
neck_index = 1
kp_sequences = [1, 2, 4, 6, 8, 3, 5, 7, 10, 12, 14, 9, 11, 13]
# kp_sequences = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
def calculate_pckh(original_image_shape, keypoint_info, pred_heatmaps, distance_ratio=0.5):
    number_of_keypoints = pred_heatmaps.shape[-1]

    # pred heatmap -> coordinate
    pred_coords = convert_heatmaps_to_keypoints(pred_heatmaps, original_image_shape)  # (x, y)s

    # gt coordinate
    gt_coords = np.array(keypoint_info["keypoints"])
    gt_coords = np.reshape(gt_coords, (number_of_keypoints, 3))

    # head coordinate
    kp0 = gt_coords[head_index, 0:2]
    # neck coordinate
    kp1 = gt_coords[neck_index, 0:2]

    threshold_dist = math.sqrt((kp0[0] - kp1[0]) ** 2 + (kp0[1] - kp1[1]) ** 2)
    threshold_dist *= distance_ratio

    scores = []
    for kp_index in range(gt_coords.shape[-1]):
        pred_x, pred_y = pred_coords[kp_index]
        gt_x, gt_y, _ = gt_coords[kp_sequences[kp_index] - 1, :]

        d = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
        if d < threshold_dist:
            scores.append(1.0)
            # each_scores[kp_index].append(1.0)
        else:
            scores.append(0.0)
            # each_scores[kp_index].append(0.0)
    score = np.mean(scores)
    # print(f'img_id = {keypoint_info["image_id"]}, threshold = {threshold_dist:.2f}, score = {score:.3f}')
    return score

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

import json
import cv2
import datetime

from evaluate_tflite import TFLiteModel
from common import get_time_to_str

def calculate_total_pckh(saved_model_path=None,
                         tflite_model_path=None,
                         annotation_path=None,
                         images_path=None,
                         distance_ratio=0.5):

    # timestamp
    _start_time = datetime.datetime.now()

    # Convert to tflite
    if tflite_model_path is None:
        tflite_model_path = save_tflite(saved_model_path=saved_model_path)
    
    # Load tflite model
    output_index = -1  # 3
    model = TFLiteModel(tflite_model_path=tflite_model_path,
                        output_index=output_index)

    # Load annotation json
    annotaiton_dict = json.load(open(annotation_path))
    image_infos = {}
    for img_info in annotaiton_dict["images"]:
        image_infos[img_info["id"]] = img_info
    keypoint_infos = annotaiton_dict["annotations"]
    # category_infos = annotaiton_dict["categories"]

    # Evaluate
    # each_scores = [[] for _ in range(14)]
    total_scores = []
    for keypoint_info in keypoint_infos:
        # print(keypoint_info.keys()) # ['num_keypoints', 'area', 'keypoints', 'bbox', 'image_id', 'category_id', 'id']
        image_info = image_infos[keypoint_info["image_id"]]
        image_path = os.path.join(images_path, image_info["file_name"])
        # Load original image
        original_image = cv2.imread(image_path)
        # Resize image
        resized_image = cv2.resize(original_image, (model.input_shape[1], model.input_shape[2]))
        resized_image = np.array(resized_image, dtype=np.float32)
        input_data = np.expand_dims(resized_image, axis=0)

        pred_heatmaps = model.inference(input_data)
        pred_heatmaps = np.squeeze(pred_heatmaps)

        score = calculate_pckh(original_image_shape=original_image.shape,
                               keypoint_info=keypoint_info,
                               pred_heatmaps=pred_heatmaps,
                               distance_ratio=distance_ratio)

        # print(f'img_id = {keypoint_info["image_id"]}, score = {score:.3f}')
        total_scores.append(score)

        # print(f"{np.mean(total_scores):.2f}")
        # batch_scores.append(score)

    total_score = np.mean(total_scores)

    # timestamp
    _end_time = datetime.datetime.now()
    _process_time = _end_time - _start_time

    print(f' ------> PCKh@{distance_ratio:.1f}: {total_score * 100.0:.2f}%, duration: {get_time_to_str(_process_time.total_seconds())} <------')
    
    return total_score

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def calculate_total_pckh_tf2(tf2_model,
                            input_size=(224, 224),
                            batch_size=32,
                            annotation_path=None,
                            images_path=None,
                            distance_ratio=0.5):
    # timestamp
    _start_time = datetime.datetime.now()

    # Load annotation json
    annotaiton_dict = json.load(open(annotation_path))
    image_infos = {}
    for img_info in annotaiton_dict["images"]:
        image_infos[img_info["id"]] = img_info
    keypoint_infos = annotaiton_dict["annotations"]
    # category_infos = annotaiton_dict["categories"]

    # Evaluate
    # each_scores = [[] for _ in range(14)]
    total_scores = []
    input_batch = None
    keypoint_info_index = 0
    number_of_keypoint_infos = len(keypoint_infos)
    gt_keypoint_infos_batch = []
    original_images_batch = []
    for keypoint_info in keypoint_infos:
        # print(keypoint_info.keys()) # ['num_keypoints', 'area', 'keypoints', 'bbox', 'image_id', 'category_id', 'id']
        image_info = image_infos[keypoint_info["image_id"]]
        image_path = os.path.join(images_path, image_info["file_name"])
        # Load original image
        original_image = cv2.imread(image_path)
        # Resize image
        resized_image = cv2.resize(original_image, input_size)
        resized_image = np.array(resized_image, dtype=np.float32)

        if input_batch is None:
            input_batch = np.expand_dims(resized_image, axis=0)
            gt_keypoint_infos_batch.append(keypoint_info)
            original_images_batch.append(original_image)
        else:
            resized_image = np.expand_dims(resized_image, axis=0)
            input_batch = np.concatenate((input_batch, resized_image))
            gt_keypoint_infos_batch.append(keypoint_info)
            original_images_batch.append(original_image)

            if input_batch.shape[0] == batch_size or keypoint_info_index == number_of_keypoint_infos-1:
                output_batch = tf2_model.predict(input_batch)
                # for heatmap_tensor in output_batch:
                #     print(heatmap_tensor)
                output_batch = output_batch[-1]

                for (i, gt_keypoint_info, original_image) in zip(range(input_batch.shape[0]), gt_keypoint_infos_batch, original_images_batch):
                    pred_heatmaps = output_batch[i]
                    pred_heatmaps = np.squeeze(pred_heatmaps)

                    score = calculate_pckh(original_image_shape=original_image.shape,
                                           keypoint_info=gt_keypoint_info,
                                           pred_heatmaps=pred_heatmaps,
                                           distance_ratio=distance_ratio)

                    # print(f'img_id = {keypoint_info["image_id"]}, score = {score:.3f}')
                    total_scores.append(score)
                input_batch = None
                gt_keypoint_infos_batch = []
                original_images_batch = []

        keypoint_info_index += 1

        # print(f"{np.mean(total_scores):.2f}")
        # batch_scores.append(score)

    total_score = np.mean(total_scores)

    # timestamp
    _end_time = datetime.datetime.now()
    _process_time = _end_time - _start_time

    print(
        f' ------> PCKh@{distance_ratio:.1f}: {total_score * 100.0:.2f}%, duration: {get_time_to_str(_process_time.total_seconds())} <------')

    return total_score

if __name__ == '__main__':

    saved_model_path = "/Volumes/tucan-SSD/ml-project/experiment007/ai_challenger/07061234_cpm_backbone_4_1/saved_model-015000"
    # tflite_model_path = "/Users/doyounggwak/projects/machine-learning/github/PoseEstimationForMobile/release/cpm_model/model.tflite"
    dataset_path = "/Volumes/tucan-SSD/datasets/ai_challenger/valid"
    annotation_path = os.path.join(dataset_path, "annotation.json")
    images_path = os.path.join(dataset_path, "images")
    distance_ratio = 0.5

    # calculate_total_pckh(tflite_model_path=tflite_model_path,
    #                      annotation_path=annotation_path,
    #                      images_path=images_path,
    #                      distance_ratio=distance_ratio)

    model = tf.keras.models.load_model(saved_model_path)
    calculate_total_pckh_tf2(model,
                             annotation_path=annotation_path,
                             images_path=images_path,
                             distance_ratio=distance_ratio)
    # print(model.input_shape)
    # print(model.output_shape)
    # print(model)

    # tflite_model_path = "/Volumes/tucan-SSD/ml-project/outputs/experiment007-mac/ai_challenger/07061221_cpm_backbone_4_1/tflite/07061221_cpm_backbone_4_1-000010.tflite"
    # calculate_total_pckh(tflite_model_path=tflite_model_path,
    #                      annotation_path=annotation_path,
    #                      images_path=images_path,
    #                      distance_ratio=distance_ratio)
