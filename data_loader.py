# Copyright 2018 Jaewook Kang (jwkang10@gmail.com) All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# -*- coding: utf-8 -*-

"""Efficient tf-tiny-pose-estimation using tf.data.Dataset.
    code ref: https://github.com/edvardHua/PoseEstimationForMobile
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
from os.path import join

from path_manager import DATASET_DIR
from pycocotools.coco import COCO


# for coco dataset
import dataset_augment
from dataset_prepare import CocoMetadata


sys.path.insert(0, DATASET_DIR)


class DataLoader(object):
    """Generates DataSet input_fn for training or evaluation
        Args:
            is_training: `bool` for whether the input is for training
            data_dir:   `str` for the directory of the training and validation data;
                            if 'null' (the literal string 'null', not None), then construct a null
                            pipeline, consisting of empty images.
            use_bfloat16: If True, use bfloat16 precision; else use float32.
            transpose_input: 'bool' for whether to use the double transpose trick
    """

    def __init__(self, is_training,
                 data_dir,
                 use_bfloat16,
                 train_config,
                 model_config,
                 preproc_config,
                 transpose_input=True):

        self.image_preprocessing_fn = dataset_augment.preprocess_image
        self.is_training            = is_training
        self.use_bfloat16           = use_bfloat16
        self.data_dir               = data_dir
        self.TRAIN_ANNO             = None
        self.train_config           = train_config
        self.model_config           = model_config
        self.preproc_config         = preproc_config

        if self.data_dir == 'null' or self.data_dir == '':
            self.data_dir = None
        self.transpose_input = transpose_input



    def _set_shapes(self,img, heatmap):

        batch_size = self.train_config.batch_size

        img.set_shape([batch_size,
                       self.model_config._input_size,
                       self.model_config._input_size,
                       self.model_config.input_chnum])

        heatmap.set_shape([batch_size,
                           self.model_config._output_size,
                           self.model_config._output_size,
                           self.model_config.output_chnum])
        return img, heatmap






    def _parse_function(self, imgId, ann=None):
        """
        :param imgId:
        :return:
        """
        imgId = imgId.numpy()

        # print("ann",ann)
        # print("imgId", imgId)

        if ann is not None:
            self.TRAIN_ANNO = ann

        # for img in self.TRAIN_ANNO.imgs:
        #     print(img)

        # print('imgId = %s' % imgId)


        img_meta = self.TRAIN_ANNO.loadImgs([imgId])[0]
        anno_ids = self.TRAIN_ANNO.getAnnIds(imgIds=imgId)
        img_anno = self.TRAIN_ANNO.loadAnns(anno_ids)
        idx = img_meta['id']

        filename_item_list = img_meta['file_name'].split('/')
        filename = filename_item_list[1] +'/' + filename_item_list[2]

        img_path = join(DATASET_DIR, filename)

        img_meta_data   = CocoMetadata(idx=idx,
                                       img_path=img_path,
                                       img_meta=img_meta,
                                       annotations=img_anno,
                                       sigma=self.preproc_config.heatmap_std)

        # print('joint_list = %s' % img_meta_data.joint_list)
        images, labels = self.image_preprocessing_fn(img_meta_data=img_meta_data,
                                                     preproc_config=self.preproc_config)
        return images, labels






    def input_fn(self, params=None):
        """Input function which provides a single batch for train or eval.
            Args:
                params: `dict` of parameters passed from the `TPUEstimator`.
                  `params['batch_size']` is always provided and should be used as the
                  effective batch size.
            Returns:
                A `tf.data.Dataset` object.
            doc reference: https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
        """
        # tf.logging.info('[Input_fn]------------------------------------')
        # tf.logging.info('[Input_fn] is_training = %s' % self.is_training)

        json_filename_split = DATASET_DIR.split('/')
        if self.is_training:
            json_filename       = json_filename_split[-1] + '_train.json'
        else:
            json_filename       = json_filename_split[-1] + '_valid.json'

        # tf.logging.info('json loading from %s' % json_filename)
        dataset_path = join(DATASET_DIR, json_filename)
        self.TRAIN_ANNO      = COCO(dataset_path)

        imgIds          = self.TRAIN_ANNO.getImgIds()
        dataset         = tf.data.Dataset.from_tensor_slices(imgIds)



        # if self.is_training:
        #     # tf.logging.info('[Input_fn] dataset shuffled and repeated.')
        #     dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=self.train_config.shuffle_size,
        #                                                                 count=None))
        # else:
        #     # tf.logging.info('[Input_fn] dataset repeated only.')
        #     dataset = dataset.repeat(count=None)

        # # Read the data from disk in parallel
        # where cycle_length is the Number of training files to read in parallel.
        # multiprocessing_num === < the number of CPU cores >

        dataset = dataset.apply(tf.data.experimental.map_and_batch(
                                    map_func=lambda imgId: tuple(
                                    tf.py_function(
                                        func=self._parse_function,
                                        inp=[imgId],
                                        Tout=[tf.float32, tf.float32])),
                                    batch_size=self.train_config.batch_size,
                                    num_parallel_calls=self.train_config.multiprocessing_num,
                                    drop_remainder=True))

        # cache entire dataset in memory after preprocessing
        # dataset = dataset.cache() # do not use this code for OOM problem

        dataset = dataset.map(self._set_shapes,
                              num_parallel_calls=self.train_config.multiprocessing_num)

        # Prefetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        # tf.logging.info('[Input_fn] dataset pipeline building complete')

        return dataset