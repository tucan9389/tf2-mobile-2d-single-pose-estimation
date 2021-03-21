import os, json, shutil

coco_annotation_json_path = "/home/centos/datasets/coco/annotations/person_keypoints_train2017.json"
image_source_dir_path = "/home/centos/datasets/coco/images/train2017"
dst_coco_annotation_json_path = "/home/centos/datasets/coco_single_person_only/train/annotation.json"
dst_image_dir_path =            "/home/centos/datasets/coco_single_person_only/train/images"

# image_source_dir_path = "/home/centos/datasets/coco/images/val2017"
# coco_annotation_json_path = "/home/centos/datasets/coco/annotations/person_keypoints_val2017.json"
# dst_coco_annotation_json_path = "/home/centos/datasets/coco_single_person_only/valid/annotation.json"
# dst_image_dir_path =            "/home/centos/datasets/coco_single_person_only/valid/images"


# ================================================================================================
# ================================================================================================
# ================================================================================================

with open(coco_annotation_json_path, 'r') as f:
    annotation_json_info = json.loads(f.read())

# dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
print(annotation_json_info.keys())
print()
print()
# [{'supercategory': 'person', 'id': 1, 'name': 'person', 'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'], 'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}]
print(annotation_json_info['categories'])
print()
print()
# {'description': 'COCO 2017 Dataset', 'url': 'http://cocodataset.org', 'version': '1.0', 'year': 2017, 'contributor': 'COCO Consortium', 'date_created': '2017/09/01'}
print(annotation_json_info['info'])
print()
print()
# [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}, {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}, {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}, {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'}, {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}, {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}]
print("annotation_info['licenses']:\n", annotation_json_info['licenses'])

image_infos = annotation_json_info['images']
annotation_infos = annotation_json_info['annotations']

print()
print("="*80)
print(annotation_infos[0])
# dict_keys(['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 'image_id', 'bbox', 'category_id', 'id'])
print(annotation_infos[0].keys())

annotation_infos_by_image_id = {}
for annotation_info in annotation_infos:
    image_id = annotation_info['image_id']
    if image_id in annotation_infos_by_image_id:
        annotation_infos_by_image_id[image_id].append(annotation_info)
    else:
        annotation_infos_by_image_id[image_id] = [annotation_info]
    
image_ids = list(annotation_infos_by_image_id.keys())
maximum_anntated_num = max(list(map(lambda image_id: len(annotation_infos_by_image_id[image_id]), image_ids)))
minimum_anntated_num = min(list(map(lambda image_id: len(annotation_infos_by_image_id[image_id]), image_ids)))


print("max:", maximum_anntated_num, "min:", minimum_anntated_num)

print()
pnum_and_count = list(map(lambda num: (num, len(list(filter(lambda image_id: len(annotation_infos_by_image_id[image_id]) == num, image_ids)))), range(minimum_anntated_num, maximum_anntated_num+1)))
for person_num, image_num in pnum_and_count:
    print("", person_num, "->", image_num)


"""train
max: 20 min: 1

 1 -> 24832
 2 -> 10730
 3 -> 5889
 4 -> 3889
 5 -> 2726
 6 -> 2104
 7 -> 1691
 8 -> 1411
 9 -> 1238
 10 -> 1198
 11 -> 1226
 12 -> 1137
 13 -> 1323
 14 -> 4705
 15 -> 12
 16 -> 2
 17 -> 0
 18 -> 1
 19 -> 0
 20 -> 1
"""

"""valid
max: 14 min: 1

 1 -> 1045
 2 -> 436
 3 -> 268
 4 -> 148
 5 -> 119
 6 -> 110
 7 -> 67
 8 -> 37
 9 -> 60
 10 -> 64
 11 -> 44
 12 -> 38
 13 -> 47
 14 -> 210
"""

print("=" * 80)

image_id_to_image_info = {}
for image_info in image_infos:
    image_id_to_image_info[image_info['id']] = image_info

print("=" * 80)

single_person_image_ids = list(filter(lambda image_id: len(annotation_infos_by_image_id[image_id]) == 1, image_ids))
print(len(single_person_image_ids))

print()
sample_annotaiton_json_path = "/home/centos/datasets/ai_challenger_tucan9389/valid/annotation.json"
with open(sample_annotaiton_json_path, 'r') as f:
    s_annotation_json_info = json.loads(f.read())
print("images num of ai_challenger_tucan9389/valid/annotation.json:", len(s_annotation_json_info['images']))
print("annots num of ai_challenger_tucan9389/valid/annotation.json:", len(s_annotation_json_info['annotations']))
print()

sample_annotaiton_json_path = "/home/centos/datasets/ai_challenger_tucan9389/train/annotation.json"
with open(sample_annotaiton_json_path, 'r') as f:
    s_annotation_json_info = json.loads(f.read())
print("images num of ai_challenger_tucan9389/train/annotation.json:", len(s_annotation_json_info['images']))
print("annots num of ai_challenger_tucan9389/train/annotation.json:", len(s_annotation_json_info['annotations']))
print()

"""
images num of ai_challenger_tucan9389/valid/annotation.json: 1500
annots num of ai_challenger_tucan9389/valid/annotation.json: 1500

images num of ai_challenger_tucan9389/train/annotation.json: 22446
annots num of ai_challenger_tucan9389/train/annotation.json: 22446
"""
print("target:", len(annotation_json_info['categories']))
print("origin:", len(s_annotation_json_info['categories']))

# ['supercategory', 'id', 'name', 'keypoints', 'skeleton']
print("target:", annotation_json_info['categories'][0].keys())
print("origin:", s_annotation_json_info['categories'][0].keys())


print("target-supercategory:", annotation_json_info['categories'][0]['supercategory'])
print("origin-supercategory:", s_annotation_json_info['categories'][0]['supercategory'])

print("target-name:", annotation_json_info['categories'][0]['name'])
print("origin-name:", s_annotation_json_info['categories'][0]['name'])

print("target-keypoints:", annotation_json_info['categories'][0]['keypoints'])
print("origin-keypoints:", s_annotation_json_info['categories'][0]['keypoints'])

print("target-skeleton:", annotation_json_info['categories'][0]['skeleton'])
print("origin-skeleton:", s_annotation_json_info['categories'][0]['skeleton'])

exit(0)

# dict_keys(['images', 'annotations', 'categories'])
print(s_annotation_json_info.keys())
# {'license': 4, 'file_name': '000000397133.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'height': 427, 'width': 640, 'date_captured': '2013-11-14 17:02:52', 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
print(image_infos[0])
# {'file_name': '89faeae39d8dd03468085095452789e632bc9096.jpg', 'height': 681, 'width': 490, 'id': 0}
print(s_annotation_json_info['images'][0])

filtered_json_annotation_info = {}
filtered_json_annotation_info['categories'] = annotation_json_info['categories']
# image_infos
filtered_image_infos = list(map(lambda image_id: image_id_to_image_info[image_id], single_person_image_ids))
filtered_json_annotation_info['images'] = filtered_image_infos
print(len(filtered_image_infos))
# annotation_infos
filterted_annotation_infos = list(map(lambda image_id: annotation_infos_by_image_id[image_id][0], single_person_image_ids))
filtered_json_annotation_info['annotations'] = filterted_annotation_infos
print(len(filterted_annotation_infos))

print()
print("images num of new:", len(filtered_json_annotation_info['images']))
print("annots num of new:", len(filtered_json_annotation_info['annotations']))

"""valid
images num of new: 1045
annots num of new: 1045
"""

"""train
images num of new: 24832
annots num of new: 24832
"""

# ================================================================================================
# ================================================================================================
# ================================================================================================

for image_info in filtered_json_annotation_info['images']:
    if not os.path.exists(os.path.join(image_source_dir_path, image_info['file_name'])):
        print(f"ERR: no image file in {os.path.join(image_source_dir_path, image_info['file_name'])}")
        exit(0)
print("============ NO error for file existing check ============")
print()

if not os.path.exists("/home/centos/datasets"):
    os.mkdir("/home/centos/datasets")
if not os.path.exists("/home/centos/datasets/coco_single_person_only"):
    os.mkdir("/home/centos/datasets/coco_single_person_only")
if not os.path.exists("/home/centos/datasets/coco_single_person_only/train"):
    os.mkdir("/home/centos/datasets/coco_single_person_only/train")
if not os.path.exists("/home/centos/datasets/coco_single_person_only/train/images"):
    os.mkdir("/home/centos/datasets/coco_single_person_only/train/images")
if not os.path.exists("/home/centos/datasets/coco_single_person_only/valid"):
    os.mkdir("/home/centos/datasets/coco_single_person_only/valid")
if not os.path.exists("/home/centos/datasets/coco_single_person_only/valid/images"):
    os.mkdir("/home/centos/datasets/coco_single_person_only/valid/images")

# write annotation.json
print("=" * 80)
print("=" * 80)
print(f"WRITE START AT {dst_coco_annotation_json_path}")
with open(dst_coco_annotation_json_path, 'w') as fp:
    json.dump(filtered_json_annotation_info, fp)
print(f"WRITE END AT {dst_coco_annotation_json_path}")
print("=" * 80)
print("=" * 80)

print()

# copy image files
echo_num = 100
pass_num = 0
copy_num = 0
total_num = len(filtered_json_annotation_info['images'])
print(f"START COPYING {total_num} FILES")
for idx, image_info in enumerate(filtered_json_annotation_info['images']):
    src_image_path = os.path.join(image_source_dir_path, image_info['file_name'])
    dst_image_path = os.path.join(dst_image_dir_path, image_info['file_name'])
    if not os.path.exists(dst_image_path):
        shutil.copyfile(src_image_path, dst_image_path)
        copy_num += 1
    else:
        pass_num += 1
    
    if (idx+1) % echo_num == 0:
        print(f"  >> {idx+1} / {total_num}, copy:{copy_num}, pass:{pass_num}")
print(f"END COPYING {total_num} FILES, copy:{copy_num}, pass:{pass_num}")
print("=" * 80)
print("=" * 80)
