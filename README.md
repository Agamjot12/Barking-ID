# Barking-ID
Deep Learning for Automated Dog Breed Identification
<br>It is an image classification model that can automatically identify dog breeds from images. It is trained to recognize 120 different breeds.

## Overview

This project implements a deep convolutional neural network using TensorFlow and transfer learning from MobileNetV2. The model achieves 82% accuracy in identifying dog breeds from images.

## Key features:
* Image preprocessing and data loading pipeline
* Efficient model architecture leveraging MobileNetV2
* Training using custom Keras callbacks and optimization
* Model saving and loading for inference

## Getting Started
The model is trained using a dataset of dog images labeled with their breeds. The training script train.py handles loading and preprocessing the data, defining and compiling the model, and training the model.
The trained model can be loaded to make predictions on new images using test.py.

## Usage
To train the model from scratch on the dataset:
````
python train.py
````
<br>To load the pretrained model and make predictions:
````
python test.py
````
This will run inference on the test dataset and save sample results to the save/ folder.

## Model Details
The model architecture uses transfer learning from a pretrained MobileNetV2 model on ImageNet. This base is trained on the dog breed dataset using an additional global average pooling layer and fully connected layers.

<br>Data augmentation and training optimization is implemented to improve model accuracy. The training loop uses custom Keras callbacks for checkpoints, learning rate adjustment, and early stopping.

## Results
The model achieves 82% top-1 accuracy on the test set of dog images. Sample prediction results on individual test images are saved to the save/ folder.
