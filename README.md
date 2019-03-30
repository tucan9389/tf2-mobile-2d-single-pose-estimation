# 💃 Mobile Pose Estimation for TensorFlow 2.0
> ~~This repository is forked from [edvardHua/PoseEstimationForMobile](https://github.com/edvardHua/PoseEstimationForMobile) when the original repository was closed.~~ <br>[edvardHua/PoseEstimationForMobile](https://github.com/edvardHua/PoseEstimationForMobile) repository is reopened! I'll maintain it separately. 👍


This repository currently implemented the Hourglass model using TensorFlow 2.0 (and Keras API). Instead of normal convolution, inverted residuals (also known as Mobilenet V2) module has been used inside the model for **real-time** inference.

# Table of contents

- [Goals](#goals)
- [Getting Started](#getting-started)
- [Running The Project](#running-the-project)
- [Template Details](#template-details)
    - [Project Architecture](#project-architecture)
    - [Folder Structure](#folder-structure)
    - [Main Components](#main-components)
- [Example Projects](#example-projects)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

# Goals
- Train with your custom annotated dataset easily.<br>
  : [tucan9389/KeypointAnnotation](https://github.com/tucan9389/KeypointAnnotation) is annotation tool ran on mobile device. You can annotate on subway!.
- Support TF 2.0 and tf.keras API
- Sperate android project
- Refactoring<br>
  : Conform [the architecture for Keras project](https://github.com/Ahmkel/Keras-Project-Template)

# Getting Started
This template allows you to simply build and train deep learning models with checkpoints and tensorboard visualization.

In order to use the template you have to:
1. Prepare the dataset and locate the dataset.
2. Run the model using:
```shell
python train.py
```

# Running The Project
A simple model for the ai challenger dataset is available.
To run the project:
1. Start the training using:
```shell
python train.py
```

# Details

## Project Architecture(Preparing...)

<div align="center">
<img align="center" width="520" src="https://github.com/Ahmkel/Keras-Project-Template/blob/master/figures/ProjectArchitecture.jpg?raw=true">
</div>

## Folder Structure(Preparing...)

```
├── train.py             - here's an example of main that is responsible for the whole pipeline.
│
│
├── base                - this folder contains the abstract classes of the project components
│   ├── base_data_loader.py   - this file contains the abstract class of the data loader.
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│
│
├── model               - this folder contains the models of your project.
│   └── simple_mnist_model.py
│
│
├── trainer             - this folder contains the trainers of your project.
│   └── simple_mnist_trainer.py
│
|
├── data_loader         - this folder contains the data loaders of your project.
│   └── simple_mnist_data_loader.py
│
│
├── configs             - this folder contains the experiment and model configs of your project.
│   └── simple_mnist_config.json
│
│
├── datasets            - this folder might contain the datasets of your project.
│
│
└── utils               - this folder contains any utils you need.
     ├── config.py      - util functions for parsing the config files.
     ├── dirs.py        - util functions for creating directories.
     └── utils.py       - util functions for parsing arguments.
```


## Main Components(Preparing...)

### Models
You need to:
1. Create a model class that inherits from **BaseModel**.
2. Override the ***build_model*** function which defines your model.
3. Call ***build_model*** function from the constructor.


### Trainers
You need to:
1. Create a trainer class that inherits from **BaseTrainer**.
2. Override the ***train*** function which defines the training logic.

**Note:** To add functionalities after each training epoch such as saving checkpoints or logs for tensorboard using Keras callbacks:
1. Declare a callbacks array in your constructor.
2. Define an ***init_callbacks*** function to populate your callbacks array and call it in your constructor.
3. Pass the callbacks array to the ***fit*** function on the model object.

**Note:** You can use ***fit_generator*** instead of ***fit*** to support generating new batches of data instead of loading the whole dataset at one time.

### Data Loaders
You need to:
1. Create a data loader class that inherits from **BaseDataLoader**.
2. Override the ***get_train_data()*** and the ***get_test_data()*** functions to return your train and test dataset splits.

**Note:** You can also define a different logic where the data loader class has a function ***get_next_batch*** if you want the data reader to read batches from your dataset each time.

### Configs
You need to define a .json file that contains your experiment and model configurations such as the experiment name, the batch size, and the number of epochs.


### Main
Responsible for building the pipeline.
1. Parse the config file
2. Create an instance of your data loader class.
3. Create an instance of your model class.
4. Create an instance of your trainer class.
5. Train your model using ".Train()" function on the trainer object.

### From Config
We can now load models without having to explicitly create an instance of each class. Look at:
1. from_config.py: this can load any config file set up to point to the right modules/classes to import
2. Look at configs/simple_mnist_from_config.json to get an idea of how this works from the config. Run it with:
```shell
python from_config.py -c configs/simple_mnist_from_config.json
```
3. See conv_mnist_from_config.json (and the additional data_loader/model) to see how easy it is to run a different experiment with just a different config file:
```shell
python from_config.py -c configs/conv_mnist_from_config.json
```

# Related Projects
* [tucan9389/PoseEstimation-CoreML](https://github.com/tucan9389/PoseEstimation-CoreML)
* [edvardHua/PoseEstimationForMobile](https://github.com/edvardHua/PoseEstimationForMobile)
* [dongseokYang/Body-Pose-Estimation-Android-gpu](https://github.com/dongseokYang/Body-Pose-Estimation-Android-gpu)


# Contributing
Any contributions are welcome including improving the project.

# Acknowledgements
This project is based on [edvardHua/PoseEstimationForMobile](https://github.com/edvardHua/PoseEstimationForMobile).


# Reference

[1] [Paper of Convolutional Pose Machines](https://arxiv.org/abs/1602.00134) <br/>
[2] [Paper of Stack Hourglass](https://arxiv.org/abs/1603.06937) <br/>
[3] [Paper of MobileNet V2](https://arxiv.org/pdf/1801.04381.pdf) <br/>
[4] [Repository PoseEstimation-CoreML](https://github.com/tucan9389/PoseEstimation-CoreML) <br/>
[5] [Repository of tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation) <br>
[6] [Devlope guide of TensorFlow Lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/docs_src/mobile/tflite) <br/>
[7] [Mace documentation](https://mace.readthedocs.io)

# License

[Apache License 2.0](LICENSE)
