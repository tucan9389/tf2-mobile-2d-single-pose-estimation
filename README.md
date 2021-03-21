# 💃 Mobile Pose Estimation for TensorFlow 2.0
> ~~This repository is forked from [edvardHua/PoseEstimationForMobile](https://github.com/edvardHua/PoseEstimationForMobile) when the original repository was closed.~~ <br>[edvardHua/PoseEstimationForMobile](https://github.com/edvardHua/PoseEstimationForMobile) repository is reopened! I'll maintain it separately. 👍


This repository currently implemented the Hourglass model using TensorFlow 2.0 with Keras API.

## Table of contents

- [Goals](#goals)
- [Getting Started](#getting-started)
- [Results](#results)
- [Converting To Mobile Model](#converting-to-mobile-model)
- [Tuning](#tuning)
- [Details](#details)
    - [Folder Structure](#folder-structure)
    - [Main Components](#main-components)
- [TODO](#todo)
- [Related Projects](#related-projects)
- [Acknowledgements](#acknowledgements)
- [Reference](#reference)
- [Contributing](#contributing)
- [License](#license)

## Goals

- 📚 Easy to train
- 🏃‍ Easy to use the model on mobile device

## Getting Started

### Install Anaconda (~10 min)

- [How To Install Anaconda on Ubuntu 18.04 [Quickstart]](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)
- [How to Install Anaconda on CentOS 7](https://linuxize.com/post/how-to-install-anaconda-on-centos-7/)

### Create Virtual Environment (~2 min)

Create new environment.
```shell
conda create -n {env_name} python={python_version} anaconda
# in my case
# conda create -n mpe-env-tf2-alpha0 python=3.7 anaconda
```

Start the environment.
```shell
source activate {env_name}
# in my case
# source activate mpe-env-tf2-alpha0
```

### Install the requirements (~1 min)
```shell
cd {tf2-mobile-pose-estimation_path}
pip install -r requirements.txt
pip install git+https://github.com/philferriere/cocoapi.git@2929bd2ef6b451054755dfd7ceb09278f935f7ad#subdirectory=PythonAPI
```


<details><summary>Download original COCO dataset.</summary>
<p>

### Download original COCO dataset

Special script that will help you to download and unpack
needed COCO datasets. Please fill COCO_DATASET_PATH with path
that is used in current version of repository.
You can check needed path in file train.py

**Warning** Your system should have approximately 40gb of free space for datasets

```shell
python downloader.py --download-path=COCO_DATASET_PATH
```

</p>
</details>

## Run The Project

In order to use the project you have to:
1. Prepare the dataset([ai_challenger dataset](https://drive.google.com/file/d/1rZng2KiEuyb-dev3HxJFYcZU4Il1VHqj/view?usp=sharing)) and unzip.
2. Run the model using:
```shell
python train.py
```

## Compatiable Datasets

Dataset Name | Doanload | Size | Number of images<br>train/valid | Number of Keypoints | Note
--- | --- | --- | --- | --- | ---
ai challenge | [google drive](https://drive.google.com/file/d/1rZng2KiEuyb-dev3HxJFYcZU4Il1VHqj/view?usp=sharing) | 2GB | 22k/1.5k | 14 | default dataset of this repo
coco single person only | [google drive](https://drive.google.com/file/d/1lwt3smqdJ2-ZuVCzgImEp8gw-RHuG-YR/view?usp=sharing) | 4GB | 25k/1k | 17 | filtered by showing only one person in an image which is from coco 2017 keypoint dataset

- ai challenge's keypoint names: `['top_head', 'neck', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']`
- coco's keypoint names: `['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']`

## Results

### AI Challenge Dataset

Model Name | Backbone | Stage Or Depth | PCH@.5 | Size | Total Epoch | Total Training Time | Note
--- | --- | --- | --- | --- | --- | --- | ---
MobileNetV2 based CPM | cpm-b0 | Stage 1 | .. | .. | .. | .. | Default CPM
MobileNetV2 based CPM | cpm-b0 | Stage 2 | .. | .. | .. | ..
MobileNetV2 based CPM | cpm-b0 | Stage 3 | .. | .. | .. | ..
MobileNetV2 based CPM | cpm-b0 | Stage 4 | .. | .. | .. | ..
MobileNetV2 based CPM | cpm-b0 | Stage 5 | .. | .. | .. | ..
MobileNetV2 based Hourglass | hg-b0 | Depth 4 | .. | .. | .. | .. | Default Hourglass

### COCO Single persononly Dataset

Model Name | Backbone | Stage Or Depth | OKS | Size | Total Epoch | Total Training Time | Note
--- | --- | --- | --- | --- | --- | --- | ---
MobileNetV2 based CPM | cpm-b0 | Stage 1 | .. | .. | .. | .. | Default CPM
MobileNetV2 based CPM | cpm-b0 | Stage 2 | .. | .. | .. | ..
MobileNetV2 based CPM | cpm-b0 | Stage 3 | .. | .. | .. | ..
MobileNetV2 based CPM | cpm-b0 | Stage 4 | .. | .. | .. | ..
MobileNetV2 based CPM | cpm-b0 | Stage 5 | .. | .. | .. | ..
MobileNetV2 based Hourglass | hg-b0 | Depth 4 | .. | .. | .. | .. | Default Hourglass

## Converting To Mobile Model

### TensorFLow Lite

If you train the model, it will create tflite models per evaluation step.

### Core ML

Check `convert_to_coreml.py` script. The converted `.mlmodel` support iOS14+.

## Details

> This section will be separated to other `.md` file.

### Folder Structure

```
tf2-mobile-pose-estimation
├── config
|   ├── model_config.py
|   └── train_config.py
├── data_loader
|   ├── data_loader.py
|   ├── dataset_augment.py
|   ├── dataset_prepare.py
|   └── pose_image_processor.py
├── models
|   ├── common.py
|   ├── mobilenet.py
|   ├── mobilenetv2.py
|   ├── mobilenetv3.py
|   ├── resnet.py
|   ├── resneta.py
|   ├── resnetd.py
|   ├── senet.py
|   ├── simplepose_coco.py
|   └── simpleposemobile_coco.py
├── train.py            - the main training script
├── common.py 
└── requirements.txt
└── outputs             - this folder will be generated automatically when start training
    ├── 20200312-sp-ai_challenger
    |   ├── saved_model
    |   └── image_results
    └── 20200312-sp-ai_challenger
        └── ...

My SSD    
└── datasets            - this folder contains the datasets of the project.
    └── ai_challenger
        ├── train.json
        ├── valid.json
        ├── train
        └── valid

```

## TODO

- ~~Save model to saved_model~~
- ~~Convert the model(saved_model) to TFLite model(`.tflite`)~~
- ~~Convert the model(saved_model) to Core ML model(`.mlmodel`)~~
- Run the model on Android
- ~~Run the model on iOS~~
- Make DEMO gif running on mobile device

## Reference

[1] [Paper of Convolutional Pose Machines](https://arxiv.org/abs/1602.00134) <br/>
[2] [Paper of Stack Hourglass](https://arxiv.org/abs/1603.06937) <br/>
[3] [Paper of MobileNet V2](https://arxiv.org/pdf/1801.04381.pdf) <br/>
[4] [Repository PoseEstimation-CoreML](https://github.com/tucan9389/PoseEstimation-CoreML) <br/>
[5] [Repository of tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation) <br>
[6] [Devlope guide of TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/docs_src/mobile/tflite) <br/>
[7] [Mace documentation](https://mace.readthedocs.io)

### Related Projects
- [tucan9389/PoseEstimation-CoreML](https://github.com/tucan9389/PoseEstimation-CoreML)
- [tucan9389/PoseEstimation-TFLiteSwift](https://github.com/tucan9389/PoseEstimation-TFLiteSwift) (Preparing...)
- [tucan9389/KeypointAnnotation](https://github.com/tucan9389/KeypointAnnotation)
- [osmr/imgclsmob](https://github.com/osmr/imgclsmob)
- [edvardHua/PoseEstimationForMobile](https://github.com/edvardHua/PoseEstimationForMobile)
- [jwkanggist/tf-tiny-pose-estimation](https://github.com/jwkanggist/tf-tiny-pose-estimatio)
- [dongseokYang/Body-Pose-Estimation-Android-gpu](https://github.com/dongseokYang/Body-Pose-Estimation-Android-gpu)

### Other Pose Estimation Projects

- [cbsudux/awesome-human-pose-estimation](https://github.com/cbsudux/awesome-human-pose-estimation)

## Contributing

> This section will be separated to other `.md` file.

Any contributions are welcome including improving the project.

# License

[Apache License 2.0](LICENSE)
