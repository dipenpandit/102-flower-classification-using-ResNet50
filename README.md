# 102 Flower Classification using ResNet50

## Project Overview

This project aims to classify 102 different species of flowers using a pre-trained ResNet50 model. The goal of the project is to build a classifier that can identify various flower species from images. It can be taken as a simple learning project on transfer learning and how to use pretrained models with your custom dataset.

## Dataset
The project uses the Oxford 102 Flower Dataset, which contains images of flowers commonly found in the United Kingdom. 
Source - https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

The dataset includes:

- 102 different categories of flowers. The labels information were stored in the `imagelabels.mat` file.
- Total images: 8,189
- High-quality images with clear views of the flowers.

### Dataset Split
The dataset split information was stored in the `setid.mat` file. The data was divided into three sets:

- Training set: 1,020 images
- Validation set: 1,020 images
- Test set: 6,149 images

## Model Architecture
- Base model: ResNet50 pre-trained on ImageNet
- Modifications:
    - Froze all layers except the final fully connected layer
    - Replaced the final layer with a new one matching our 102 classes
    - Kept the pre-trained weights for better feature extraction

## Dependencies
- PyTorch
- torchvision
- numpy
- Pillow
- matplotlib
- tqdm
- torchmetrics