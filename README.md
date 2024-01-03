# Image Inpainting Detection

## Overview

This project implements an image inpainting detection model using a U-Net architecture with MobileNetV2 as the encoder. The model is trained to identify inpainted regions in images.

## Dataset

The dataset consists of pairs of original images and corresponding segmentation masks. Original images are located in the "Inpainted_Images" directory, and segmentation masks are in the "Mask_images" directory. The testing set is similarly structured in the "Test_Images" and "Test_Mask" directories.

##Model

The U-Net architecture with a MobileNetV2 encoder is used for image inpainting detection.
The model is trained using a custom Binary Focal Loss as the loss function.


## Prerequisites

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- NumPy
- Focal Loss Library (Install using `pip install focal-loss`)

## Usage

1. Set up the environment and install dependencies
   
Update the dataset paths and other parameters in the script (code.py) based on your directory structure.

Run the script to load the trained model and predict inpainted regions in the test set:
