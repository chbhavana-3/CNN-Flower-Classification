# Flower Classification using Convolutional Neural Networks

## Introduction

This project aims to classify different types of flowers using Convolutional Neural Networks (CNNs). The dataset consists of images of various types of flowers, and the objective is to build a model that accurately predicts the type of flower present in each image.

## Model Building

The model architecture consists of three Conv2D layers followed by a Flatten layer to prepare the data for the fully connected layers. After the convolutional layers, the feature maps are flattened and passed through fully connected layers for classification. No max-pooling or augmentation techniques are used in this basic model.

## Model Evaluation

To evaluate the model, we compare its performance on both the training and test datasets using accuracy and ROC-AUC values. These metrics provide insights into the model's ability to correctly classify the flowers and its ability to distinguish between classes.

## Prediction

Once the model is trained and evaluated, we can use it to predict the types of flowers present in new images. The CHECK directory contains images for which we'll make predictions using the trained model.

We've implemented the model and evaluation steps as follows:
- Loaded the dataset and prepared it for modeling.
- Built a CNN model with three Conv2D layers and compared its performance on the training and test datasets.
- Plotted the ROC curve for the test dataset to visualize model performance.
- Predicted the type of all flowers in the CHECK directory using the trained model.

## Running the Code

Ensure you have the necessary dependencies installed, such as TensorFlow, scikit-learn, and Matplotlib. You can install them using pip:

```bash
pip install numpy matplotlib scikit-learn tensorflow
