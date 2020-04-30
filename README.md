# Movidius Lunar Localization

## Background
This is the repository for our NASA Lunar Localization Senior Design Project, developed for Intel. It aims to build off of the NASA Frontier Development Lab's research into an approach to [Localization on the Moon Using a Neural Network Model](https://ieeexplore.ieee.org/document/8968124). The project aims to run a trained neural network model off of an Intel Neural Compute Stick 2 USB device. The model processes ground view images of the lunar landscape captured by a camera and then reprojects them into an approximated top-down view image of the surrounding landscape and then compares those images to satellite gathered aerial images via a neural network in order to determine a rover's location.

## Dependencies/Requirements for Development
The project requires a great number of dependencies as well as hardware in order to set up the environment for a local build that runs the neural network model off the Compute Stick. More detailed instructions for a local build will be compiled and uploaded later.

- [Ubuntu 18.4.3 LTS](http://old-releases.ubuntu.com/releases/18.04.3/)
- [Tensorflow](https://www.tensorflow.org/install/pip) & [TensorFlow Hub](https://www.tensorflow.org/hub/installation)
- [OpenCV 3.4 or higher](https://pypi.org/project/opencv-python/)
- Python 3.5 or higher
- CMake 2.8 or higher
- [Intel OpenVINO toolkit for Linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html#Install-Dependencies)

The project also requires these hardware components in order to run as intended:
- Intel Neural Compute Stick 2
- 1080p camera/webcam compatible with Linux
- Stepper motor and motor controller

## Project Components
**tf_model/** - This folder contains the previously trained TensorFlow model data.

**ovino_model/** - This folder contains the converted model data which was converted from TensorFlow data into Intel-compatible data with OpenVINO model conversion.

**image_match_train.py** - This is the script for training the lunar landscape dataset using a Convolutional Neural Network (CNN) model

**image_match_predict.py** - Contains the bulk of our implementation, including camera capture function, stepper motor functionality,  image reprojection, and final reprojected image display onto a GUI. The **run.py** script runs the whole process.

## Description
With the environment fully set up and all hardware connected, run "source /opt/intel/openvino/bin/setupvars.sh" to load the OpenVINO environment in order to utilize the Compute Stick. From this directory, executing the **run.py** file will open the GUI so that image capture and reprojection can begin.
