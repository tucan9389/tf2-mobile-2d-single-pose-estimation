# Copyright 2018 Jaewook Kang (jwkang10@gmail.com)
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
# ===================================================================================
# -*- coding: utf-8 -*-
# ! /usr/bin/env python
'''
    filename: path_manager.py
    description: this module include all path information on this proj
    - Author : jaewook Kang @ 20180613
'''

from os import getcwd
from os import chdir
import os

# move to project home directory
chdir('..')

PROJ_HOME               = os.path.join(getcwd(), "tf2-mobile-pose-estimation")#"tf2-mobile-pose-estimation")
TF_MODULE_DIR           = PROJ_HOME              #+ '/tfmodules'

print("[pathmanager] PROJ HOME = %s" % PROJ_HOME)


# tf module related directory
EXPORT_DIR                = os.path.join(PROJ_HOME, 'export')
# EXPORT_DIR                = 'gs://tf-tiny-pose-est'
COCO_DATALOAD_DIR         = TF_MODULE_DIR          #+ '/coco_dataload_modules'


# data path
DATASET_DIR                 = PROJ_HOME     + '/datasets/ai_challenger'
#DATASET_DIR                  = '/home/jwkangmacpro2/dataset/ai_challenger'

COCO_TRAINSET_DIR            = DATASET_DIR     + '/train/'
COCO_VALIDSET_DIR            = DATASET_DIR     + '/valid/'
LOCAL_LOG_DIR                = PROJ_HOME       + '/export'
print("[pathmanager] DATASET_DIR = %s" % DATASET_DIR)
print("[pathmanager] COCO_DATALOAD_DIR = %s" % COCO_DATALOAD_DIR)
print("[pathmanager] EXPORT_DIR = %s" % EXPORT_DIR)