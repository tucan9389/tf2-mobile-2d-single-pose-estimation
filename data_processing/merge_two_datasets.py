import os, json, shutil

split_symbol = "train"
src1_coco_annotation_json_path = f"/home/centos/datasets/ai_challenger_tucan9389/{split_symbol}/annotation.json"
src1_image_dir_path =            f"/home/centos/datasets/ai_challenger_tucan9389/{split_symbol}/images"
src2_coco_annotation_json_path = f"/home/centos/datasets/coco_single_person_only/{split_symbol}/annotation.json"
src2_image_dir_path =            f"/home/centos/datasets/coco_single_person_only/{split_symbol}/images"
dst_coco_annotation_json_path =  f"/home/centos/datasets/aic_coco_tucan9389_001/{split_symbol}/annotation.json"
dst_image_dir_path =             f"/home/centos/datasets/aic_coco_tucan9389_001/{split_symbol}/images"

def mkdir_reculsively(full_path):
    path = ""
    for dir in full_path.split("/"):
        if dir == "":
            path += "/"
        else:
            path = os.path.join(path, dir)
        if path != "/" and not os.path.exists(path):
            os.mkdir(path)

mkdir_reculsively(dst_image_dir_path)

# merge annotation.json
print("=" * 80)
print("=" * 80)
with open(src1_coco_annotation_json_path, 'r') as f:
    s1_annotation_json_info = json.loads(f.read())
with open(src2_coco_annotation_json_path, 'r') as f:
    s2_annotation_json_info = json.loads(f.read())

d_annotation_json_info = {}

for key in s1_annotation_json_info.keys():
    if key == "images" or key == "annotations":
        d_annotation_json_info[key] = s1_annotation_json_info[key] + s2_annotation_json_info[key]
    else:
        d_annotation_json_info[key] = s1_annotation_json_info[key]

print(f"WRITE START AT {dst_coco_annotation_json_path}")
with open(dst_coco_annotation_json_path, 'w') as fp:
    json.dump(d_annotation_json_info, fp)
print(f"WRITE END AT {dst_coco_annotation_json_path}")
print("=" * 80)
print("=" * 80)

# copy two source image files
src1_filenames = os.listdir(src1_image_dir_path)
src1_filenames = list(filter(lambda filename: not filename.startswith("."), src1_filenames))
src2_filenames = os.listdir(src2_image_dir_path)
src2_filenames = list(filter(lambda filename: not filename.startswith("."), src2_filenames))

echo_num = 100

pass_num = 0
copy_num = 0
total_num = len(src1_filenames)
print(f"START COPYING {total_num} FILES from src1")
for idx, filename in enumerate(src1_filenames):
    src_image_path = os.path.join(src1_image_dir_path, filename)
    dst_image_path = os.path.join(dst_image_dir_path, filename)
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

pass_num = 0
copy_num = 0
total_num = len(src2_filenames)
print(f"START COPYING {total_num} FILES from src2")
for idx, filename in enumerate(src2_filenames):
    src_image_path = os.path.join(src2_image_dir_path, filename)
    dst_image_path = os.path.join(dst_image_dir_path, filename)
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