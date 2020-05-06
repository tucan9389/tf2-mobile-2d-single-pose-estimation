import os
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json

def get_bgimg(inp, target_size=None):
    inp = cv2.cvtColor(inp.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if target_size:
        inp = cv2.resize(inp, target_size, interpolation=cv2.INTER_AREA)
    return inp

def save_result_image(filepath=None, inp_image=None, true_heat=None, pred_heat=None):
    fig = plt.figure()

    if true_heat is not None:
        a = fig.add_subplot(1, 2, 1)
        a.set_title('True Heatmap')
        plt.imshow(get_bgimg(inp_image, target_size=(true_heat.shape[1], true_heat.shape[0])), alpha=0.5)
        tmp = np.amax(true_heat, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.7)
        plt.colorbar()

    if pred_heat is not None:
        a = fig.add_subplot(1, 2, 2)
        a.set_title('Pred Heatmap')
        plt.imshow(get_bgimg(inp_image, target_size=(pred_heat.shape[1], pred_heat.shape[0])), alpha=0.5)
        tmp = np.amax(pred_heat, axis=2)
        plt.imshow(tmp, cmap=plt.cm.gray, alpha=1, vmin=0.0, vmax=1.0)

        plt.colorbar()

    if filepath != None:
        fig.savefig(filepath)







def process(images_path, image_filename):
    image_filepath = os.path.join(images_path, image_filename)
    print(image_filepath)

    # Get an image
    image = Image.open(image_filepath)
    # image.show()

    resized_image = image.resize((128, 128))
    np_image = np.array(resized_image)
    np_image = np_image.reshape((1, 128, 128, 3))
    np_image = np_image.astype(np.float32)
    # print(np_image.shape)

    # Inference
    heatmaps = model(np_image)
    np_heatmaps = np.array(heatmaps)
    # print(heatmaps.shape)

    # for DEBUGGING
    # save_result_image(filepath="result.jpg", inp_image=np.array(resized_image), pred_heat=np_heatmaps.reshape((32, 32, 14)))

    # Post-processing
    np_heatmaps = np_heatmaps.reshape(32, 32, 14)
    for i in range(np_heatmaps.shape[2]):
        heatmap = np_heatmaps[:,:,i]
        idx = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
        # print(heatmap[idx], idx)
        if heatmap[idx] > 0.3:
            confidence = heatmap[idx]
            float_idx = np.array(idx).astype(np.float32) / np_heatmaps.shape[0]
            print(i, confidence, "->", "(%.3f,%.3f)"%(float_idx[1], float_idx[0]))
            return float_idx, confidence
        else:
            print("None")
            return None, None

def get_confidence_and_np_index(heatmap):
    idx = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
    # print(heatmap[idx], idx)
    if heatmap[idx] > 0.3:
        confidence = heatmap[idx]
        np_idx = np.array(idx).astype(np.float32)
        np_idx = np_idx + 0.3 # make middle point for one area
        np_idx = np_idx / heatmap.shape[0]
        # print(confidence, "->", "(%.3f,%.3f)" % (np_idx[1], np_idx[0]))
        return confidence, np_idx
    else:
        # print("None")
        return None, None


def process(model, images_path, images_info, batch_size=32):
    annotation_infos = []
    index = 0
    # image_filenames = os.listdir(images_path)
    number_of_images = len(images_info)
    batch = []
    batch_image = []
    for image_info in images_info:
        image_filename = image_info["file_name"]
        image_filepath = os.path.join(images_path, image_filename)
        # Get an image
        image = Image.open(image_filepath)
        # image.show()

        resized_image = image.resize((128, 128))
        np_image = np.array(resized_image)
        np_image = np_image.astype(np.float32)

        batch.append(np_image)
        batch_image.append(image_info)

        index += 1
        if len(batch) == batch_size or index == number_of_images:
            # Inference
            np_batch = np.array(batch)
            heatmaps = model(np_batch)
            np_batch_heatmaps = np.array(heatmaps)

            # Post-processing
            for idx in range(len(batch)):
                image_info = batch_image[idx]

                np_heatmaps =  np_batch_heatmaps[idx, :, :, :] # 32
                kp_count = np_heatmaps.shape[2] # 14: kp number
                keypoints = [0] * (kp_count*3)
                for i in range(kp_count):
                    heatmap = np_heatmaps[:, :, i]
                    conf, np_idx = get_confidence_and_np_index(heatmap)
                    if conf is not None:
                        # print(idx, i, conf, np_idx)
                        x = np_idx[1]
                        y = np_idx[0]
                        keypoints[i * 3 + 0] = int(x * image_info["width"])
                        keypoints[i * 3 + 1] = int(y * image_info["height"])
                        keypoints[i * 3 + 2] = 2

                    # else:
                    #     print("..")


                annotation_info = {}
                annotation_info["category_id"] = 1
                annotation_info["keypoints"] = keypoints
                annotation_info["image_id"] = image_info["id"]
                annotation_info["id"] = image_info["id"]
                annotation_info["num_keypoints"] = 0
                annotation_info["area"] = 0
                annotation_info["bbox"] = []
                annotation_infos.append(annotation_info)
            batch = []
            batch_image = []
            print("-"*20 + "(%d/%d) BATCH DONE" % ((index+1) / batch_size, ((number_of_images-1) / batch_size)+1) + "-"*20)
    print(annotation_infos)
    return annotation_infos







# Load model from saved_model_path
saved_model_path = os.path.join("/Users/user/Downloads", "saved_model 2")
model = tf.keras.models.load_model(saved_model_path)
model.return_heatmap = True

# Dataset path
datasets_path = "/Users/user/Downloads/Annotated"#"/Users/user/Project/dataset/LineReceipt/Annotated"

dataset_dirnames = os.listdir(datasets_path)
dataset_dirnames.sort()
# del dataset_dirnames[0]

for dataset_dirname in dataset_dirnames:
    dataset_path = os.path.join(datasets_path, dataset_dirname)
    print(dataset_path)
    images_path = os.path.join(dataset_path, "images")
    json_filepath = os.path.join(dataset_path, "annotation.json")
    with open(json_filepath) as f:
        annotation_info = json.load(f)

    print(annotation_info.keys())

    print()
    images_info = annotation_info["images"]
    annotations_info = annotation_info["annotations"]
    categories_info = annotation_info["categories"]

    # # RUN for test
    # image_filename = image_filenames[2]
    # float_idx, confidence = process(images_path, image_filename)

    annotation_infos = process(model, images_path, images_info)
    annotation_info["annotations"] = annotation_infos

    print("number of annotation information:", len(annotation_infos))

    # save
    with open(json_filepath, 'w') as json_file:
        json.dump(annotation_info, json_file)
    print("** WRITE ON", json_filepath)

    # break