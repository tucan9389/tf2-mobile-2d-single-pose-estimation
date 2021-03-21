# -*- coding: utf-8 -*-
# @Time    : 18-3-7 下午2:36
# @Author  : edvard_hua@live.com
# @FileName: dataset_augument.py
# @Software: PyCharm
# @updated by Jaewook Kang 20181010 for tf-tiny-pose-estimation
# @updated by Doyoung Gwak 20200607 for tf2-mobile-pose-estimation

import math
import random
random.seed(3)

import cv2
import numpy as np
from tensorpack.dataflow.imgaug.geometry import RotationAndCropValid
from enum import Enum

# class CocoPart(Enum):
#     TOPLEFT = 0
#     TOPRIGHT = 1
#     BOTTOMLEFT = 2
#     BOTTOMRIGHT = 3

# class CocoPart(Enum):
#     Top = 0
#     Neck = 1
#     RShoulder = 2
#     RElbow = 3
#     RWrist = 4
#     LShoulder = 5
#     LElbow = 6
#     LWrist = 7
#     RHip = 8
#     RKnee = 9
#     RAnkle = 10
#     LHip = 11
#     LKnee = 12
#     LAnkle = 13
#     Background = 14  # Background is not used

class CocoPart(Enum):
    Nose = 0
    LeftEye = 1
    RightEye = 2
    LeftEar = 3
    RightEar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16
    # Background = 14  # Background is not used

def pose_random_scale(meta):
    scalew = random.uniform(0.8, 1.2)
    scaleh = random.uniform(0.8, 1.2)
    neww = int(meta.width * scalew)
    newh = int(meta.height * scaleh)

    dst = cv2.resize(meta.img, (neww, newh), interpolation=cv2.INTER_AREA)

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            adjust_joint.append((int(point[0] * scalew + 0.5), int(point[1] * scaleh + 0.5)))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww, newh
    meta.img = dst
    return meta


def pose_rotation(meta, config_preproc):
    deg = random.uniform(config_preproc["rotate_min_degree"], \
                         config_preproc["rotate_max_degree"])
    img = meta.img

    center = (img.shape[1] * 0.5, img.shape[0] * 0.5)  # x, y
    rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
    ret = cv2.warpAffine(img, rot_m, img.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    if img.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
    neww = min(neww, ret.shape[1])
    newh = min(newh, ret.shape[0])
    newx = int(center[0] - neww * 0.5)
    newy = int(center[1] - newh * 0.5)
    # print(ret.shape, deg, newx, newy, neww, newh)
    img = ret[newy:newy + newh, newx:newx + neww]

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1, -1))
            #     continue
            x, y = _rotate_coord((meta.width, meta.height), (newx, newy), point, deg)
            adjust_joint.append((x, y))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww, newh
    meta.img = img

    return meta


def pose_flip(meta):
    return meta

    r = random.uniform(0, 1.0)
    if r > 0.5:
        return meta

    img = meta.img
    img = cv2.flip(img, 1)

    # flip meta
    flip_list = [CocoPart.Top, CocoPart.Neck, CocoPart.LShoulder, CocoPart.LElbow, CocoPart.LWrist, CocoPart.RShoulder,
                 CocoPart.RElbow, CocoPart.RWrist,
                 CocoPart.LHip, CocoPart.LKnee, CocoPart.LAnkle, CocoPart.RHip, CocoPart.RKnee, CocoPart.RAnkle]

    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for cocopart in flip_list:
            point = joint[cocopart.value]
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((meta.width - point[0], point[1]))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list

    meta.img = img
    return meta


def pose_resize_shortestedge_random(meta, config_model):
    ratio_w = float(config_model["input_width"]) / float(meta.width)
    ratio_h = float(config_model["input_height"]) / float(meta.height)
    ratio = min(ratio_w, ratio_h)

    target_size = int(min(meta.width * ratio + 0.5, meta.height * ratio + 0.5))
    target_size = int(target_size * random.uniform(0.95, 1.2))

    # target_size = int(min(_network_w, _network_h) * random.uniform(0.7, 1.5))
    return pose_resize_shortestedge(meta=meta, target_size=target_size, config_model=config_model)


def _rotate_coord(shape, newxy, point, angle):
    angle = -1 * angle / 180.0 * math.pi

    ox, oy = shape
    px, py = point

    ox /= 2
    oy /= 2

    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    new_x, new_y = newxy

    qx += ox - new_x
    qy += oy - new_y

    return int(qx + 0.5), int(qy + 0.5)


def pose_resize_shortestedge(meta, target_size, config_model):
    img = meta.img

    # adjust image
    scale = float(target_size) / float(min(meta.height, meta.width))

    if meta.height < meta.width:
        newh, neww = target_size, int(scale * meta.width + 0.5)
    else:
        newh, neww = int(scale * meta.height + 0.5), target_size

    dst = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

    pw = ph = 0
    if neww < config_model["input_width"] or newh < config_model["input_height"]:
        pw = max(0, (config_model["input_width"] - neww) // 2)
        ph = max(0, (config_model["input_height"] - newh) // 2)
        mw = (config_model["input_width"] - neww) % 2
        mh = (config_model["input_height"] - newh) % 2
        color1 = random.randint(0, 255)
        color2 = random.randint(0, 255)
        color3 = random.randint(0, 255)
        dst = cv2.copyMakeBorder(dst, ph, ph + mh, pw, pw + mw, cv2.BORDER_CONSTANT, value=(color1, color2, color3))

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0 or int(point[0]*scale+0.5) > neww or int(point[1]*scale+0.5) > newh:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((int(point[0] * scale + 0.5) + pw, int(point[1] * scale + 0.5) + ph))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = neww + pw * 2, newh + ph * 2
    meta.img = dst
    return meta


def pose_crop(meta, x, y, w, h):
    # adjust image
    target_size = (w, h)

    img = meta.img
    resized = img[y:y + target_size[1], x:x + target_size[0], :]

    # adjust meta data
    adjust_joint_list = []
    for joint in meta.joint_list:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1000, -1000))
            #     continue
            new_x, new_y = point[0] - x, point[1] - y
            # if new_x <= 0 or new_y <= 0 or new_x > target_size[0] or new_y > target_size[1]:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((new_x, new_y))
        adjust_joint_list.append(adjust_joint)

    meta.joint_list = adjust_joint_list
    meta.width, meta.height = target_size
    meta.img = resized
    return meta


def pose_crop_random(meta, config_model):
    target_size = (config_model["input_width"], config_model["input_height"])
    for _ in range(50):
        x = random.randrange(0, meta.width - target_size[0]) if meta.width > target_size[0] else 0
        y = random.randrange(0, meta.height - target_size[1]) if meta.height > target_size[1] else 0

        # check whether any face is inside the box to generate a reasonably-balanced datasets

        # --------------------------------------------------------------------------
        # for joint in meta.joint_list:
        #     if x <= joint[CocoPart.RKnee.value][0] < x + target_size[0] and \
        #             y <= joint[CocoPart.RKnee.value][1] < y + target_size[1] and \
        #             x <= joint[CocoPart.RAnkle.value][0] < x + target_size[0] and \
        #             y <= joint[CocoPart.RAnkle.value][1] < y + target_size[1] and \
        #             x <= joint[CocoPart.LKnee.value][0] < x + target_size[0] and \
        #             y <= joint[CocoPart.LKnee.value][1] < y + target_size[1] and \
        #             x <= joint[CocoPart.LAnkle.value][0] < x + target_size[0] and \
        #             y <= joint[CocoPart.LAnkle.value][1] < y + target_size[1]:
        #         break
        # --------------------------------------------------------------------------

    return pose_crop(meta, x, y, target_size[0], target_size[1])


def pose_to_img(meta_l, config_model):
    # global _network_w, _network_h, _scale
    # return meta_l.img.astype(np.float32), \
    #        meta_l.get_heatmap(target_size=(_network_w // _scale, _network_h // _scale)).astype(np.float32)
    return meta_l.img.astype(np.float32), \
           meta_l.get_heatmap(target_size=(config_model["output_height"], config_model["output_width"])).astype(np.float32)
    # return meta_l.img.astype(np.float32), \
    #        meta_l.get_heatmap(target_size=(model_config.output_size, model_config.output_size)).astype(np.float32)


def preprocess_image(img_meta_data, config_model, config_preproc):
    if config_preproc["is_scale"]:
        img_meta_data = pose_random_scale(img_meta_data)

    if config_preproc["is_rotate"]:
        img_meta_data = pose_rotation(img_meta_data, config_preproc)

    if config_preproc["is_flipping"]:
        img_meta_data = pose_flip(img_meta_data)

    if config_preproc["is_resize_shortest_edge"]:
        img_meta_data = pose_resize_shortestedge_random(img_meta_data, config_model=config_model)

    if config_preproc["is_crop"]:
        img_meta_data = pose_crop_random(meta=img_meta_data, config_model=config_model)
    else:
        target_size = (config_model["input_width"], config_model["input_height"])
        pose_crop(img_meta_data, 0, 0, target_size[0], target_size[1])

    images, labels = pose_to_img(img_meta_data, config_model=config_model)

    return images, labels
