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


def load_and_inference_saved_model(np_input):
    # Load model from saved_model_path
    saved_model_path = os.path.join("/Users/user/Downloads", "saved_model 2")
    model = tf.keras.models.load_model(saved_model_path)
    model.return_heatmap = True

    # Inference
    heatmaps = model(np_input)
    np_heatmaps = np.array(heatmaps)
    print(heatmaps.shape)
    return np_heatmaps

# for DEBUGGING
# save_result_image(filepath="result.jpg", inp_image=np.array(resized_image), pred_heat=np_heatmaps.reshape((32, 32, 14)))

def load_and_inference_tflite(np_input):
    import numpy as np
    import tensorflow as tf

    tflite_model_path = os.path.join("/Users/user/Downloads", "tflite", "sp_receipt_top_3307_2.tflite")

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], np_input)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    return output_data

def get_image_and_true_heatmap():


    datasets_path = "/Users/user/Project/dataset/LineReceipt/Annotated3/"
    dataset_dirname = "receipt01"

    dataset_path = os.path.join(datasets_path, dataset_dirname)
    print(dataset_path)
    images_path = os.path.join(dataset_path, "images")
    json_filepath = os.path.join(dataset_path, "annotation.json")
    with open(json_filepath) as f:
        annotation_info = json.load(f)

    print(annotation_info.keys())


    images_infos = annotation_info["images"]
    keypoints_infos = annotation_info["annotations"]
    categories_infos = annotation_info["categories"]

    kp_info = keypoints_infos[0]
    im_info = None
    for image_info in images_infos:
        if image_info["id"] == kp_info["image_id"]:
            im_info = image_info
    image_filename = im_info["file_name"]



    image_filepath = os.path.join(images_path, image_filename)
    print(image_filepath)

    # Get an image
    image = Image.open(image_filepath)
    # image.show()

    resized_image = image.resize((128, 128))
    np_image = np.array(resized_image)
    np_image = np_image.reshape((1, 128, 128, 3))
    np_image = np_image.astype(np.float32)

    return np_image



np_image = get_image_and_true_heatmap()

np_heatmaps_from_saved_model = load_and_inference_saved_model(np_image)
np_heatmaps_from_tflite = load_and_inference_tflite(np_image)

print(np_heatmaps_from_saved_model)
print(np_heatmaps_from_tflite)

save_result_image("result_saved_model.jpg", inp_image=np_image.reshape((128, 128, 3)), pred_heat=np_heatmaps_from_saved_model.reshape((32, 32, 14)))
save_result_image("result_tflite.jpg", inp_image=np_image.reshape((128, 128, 3)), pred_heat=np_heatmaps_from_tflite.reshape((32, 32, 14)))