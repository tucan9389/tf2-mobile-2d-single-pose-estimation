import os
import datetime

import tensorflow as tf
import numpy as np

from config.model_config import ModelConfig
from config.train_config import PreprocessingConfig
from config.train_config import TrainConfig
from save_result_as_image import save_result_image

print("tensorflow version   :", tf.__version__)  # 2.1.0
print("keras version        :", tf.keras.__version__)  # 2.2.4-tf

train_config = TrainConfig()
model_config = ModelConfig()
preproc_config = PreprocessingConfig()

dataset_path = "/Users/user/Downloads/receipt_annotations/receipt_top_bottom_5982"  # "/Volumes/tucan-SSD/datasets/coco_dataset" # "/Volumes/tucan-SSD/datasets/ai_challenger"
dataset_name = dataset_path.split("/")[-1]
current_time = datetime.datetime.now().strftime("%m%d%H%M")

output_path = "/Users/user/Project/ml-project/receipt/pose/outputs"
output_name = "test_datset_001"

def save_image_results(step, images, true_heatmaps):
    val_image_results_directory = "val_image_results"

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(os.path.join(output_path, output_name)):
        os.mkdir(os.path.join(output_path, output_name))
    if not os.path.exists(os.path.join(output_path, output_name, val_image_results_directory)):
        os.mkdir(os.path.join(output_path, output_name, val_image_results_directory))

    for i in range(images.shape[0]):
        image = np.array(images[i, :, :, :])
        heatmap = np.array(true_heatmaps[i, :, :, :])
        print(heatmap.max(), heatmap.min(), heatmap.argmax())
        # prediction = predicted_heatmaps[i, :, :, :]

        # result_image = display(i, image, heamap, prediction)
        result_image_path = os.path.join(output_path, output_name, val_image_results_directory, "result%d-%d.jpg" % (i, step))
        save_result_image(result_image_path, image, heatmap, title="img-idx:%d" % (step*32))
        # print("val_step: save result image on \"" + result_image_path + "\"")
        break

# ================================================
# ================= load dataset =================
# ================================================

from data_loader.data_loader import DataLoader

# dataloader instance gen
train_images_dir_path = os.path.join(dataset_path, "train/images")
train_annotation_json_filepath = os.path.join(dataset_path, "train/annotations.json")
print(">> LOAD TRAIN DATASET FORM:", train_annotation_json_filepath)
dataloader_train = DataLoader(
    images_dir_path=train_images_dir_path,
    annotation_json_path=train_annotation_json_filepath,
    train_config=train_config,
    model_config=model_config,
    preproc_config=preproc_config)

valid_images_dir_path = os.path.join(dataset_path, "valid/images")
valid_annotation_json_filepath = os.path.join(dataset_path, "valid/annotations.json")
print(">> LOAD VALID DATASET FORM:", valid_annotation_json_filepath)
dataloader_valid = DataLoader(
    images_dir_path=valid_images_dir_path,
    annotation_json_path=valid_annotation_json_filepath,
    train_config=train_config,
    model_config=model_config,
    preproc_config=preproc_config)

number_of_keypoints = dataloader_train.number_of_keypoints # 17

# train dataset
dataset_train = dataloader_train.input_fn()

# validation images
val_images, val_heatmaps = dataloader_valid.get_images(0, batch_size=25) # from 22 index 6 images and 6 labels

num_epochs = 1

if __name__ == '__main__':
    for epoch in range(num_epochs):
        print("-" * 10 + " " + str(epoch + 1) + " EPOCH " + "-" * 10)
        step = 0
        for images, heatmaps in dataset_train:
            step += 1
            if step % 10 == 0:
                print(step)
                # predictions = model(images)
                # predictions = np.array(predictions)
                save_image_results(step, images, heatmaps)  #, predictions)
