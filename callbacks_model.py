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
# -*- coding: utf-8 -*-

import os
import tensorflow as tf


def get_check_pointer_callback(model_path, output_name):
    checkpoint_path = os.path.join(model_path, output_name + ".hdf5")  # ".ckpt"
    check_pointer_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                                save_weights_only=False,
                                                                verbose=1)
    return check_pointer_callback


def get_tensorboard_callback(log_path, output_name):
    log_path = os.path.join(log_path, output_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True,
                                                          write_images=True)

    return tensorboard_callback


def get_img_tensorboard_callback(log_path, output_name, images, labels, model):
    # tensorboard image
    def _show_image_for_debugging(numpy_img):
        from PIL import Image
        # import io
        # height, width, channel = numpy_img.shape
        img = Image.fromarray(numpy_img)

        img.save('my.png')
        img.show()

    file_writer = tf.summary.create_file_writer(os.path.join(log_path, output_name))

    def log_tensorboard_predicted_images(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        # batch_size = 6
        # images, labels = dataloader_valid.get_images(80, batch_size)
        predictions = model.predict(images)

        # Log the confusion matrix as an image summary.
        from data_loader.pose_image_processor import PoseImageProcessor

        # summary_str = []
        predicted_images = []
        for i in range(images.shape[0]):
            image = images[i, :, :, :]
            label = labels[i, :, :, :]
            prediction = predictions[i, :, :, :]

            numpy_img = PoseImageProcessor.display_image(image, true_heat=label, pred_heat=prediction, as_numpy=True)

            numpy_img = numpy_img / 255

            predicted_images.append(numpy_img)

        with file_writer.as_default():
            # Don't forget to reshape.
            tf.summary.image("predict from validation dataset", predicted_images, max_outputs=10, step=epoch)

    # Define the per-epoch callback.
    img_tensorboard_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_tensorboard_predicted_images)

    return img_tensorboard_callback
