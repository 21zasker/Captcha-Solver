# CAPTCHA Solver

## Overview
This Python project implements a text Captcha solver using two machine learning models: Support Vector Machine (SVM) from the `scikit-learn` library and a custom-built Learning Vector Quantization (LVQ-1) algorithm. The solver is designed to identify and decode alphanumeric characters from CAPTCHA images. The code processes each CAPTCHA image, extracts individual characters, and then predicts each character using the chosen model.

## Features
* CAPTCHA image processing and character extraction
* Character recognition using SVM and LVQ-1 models
* Custom implementation of the LVQ-1 algorithm
* User input to choose between SVM and LVQ-1 models for prediction
* Visualization of prediction results on the CAPTCHA image
* Visualization of class vectors in 2D using t-SNE

## Architecture
**SVM model:**

A linear kernel-based classifier that learns to distinguish between different characters by finding the optimal boundary between them in the feature space. It is trained on images of alphanumeric characters.

**LVQ-1 model:**

A model where each class is represented by a vector. These vectors are iteratively adjusted based on the training data, moving closer to similar input samples and away from dissimilar ones. t-SNE is used to visualize these vectors, revealing how closely related characters are positioned near each other.

**Input Processing:** 

Handles loading and preprocessing of CAPTCHA images, including grayscale conversion, thresholding, and contour detection to extract characters based on bounding boxes around the contours.

## Functions
* `train_lvq`: Trains the LVQ-1 model by adjusting the class vectors based on the training data.
* `resize_to_fit`: Resizes the extracted character images to a fixed size suitable for model input.
* `random_indexes`: Selects random indexes from the training set to initialize the LVQ-1 vectors.

## Dependencies
* Python
* Scikit-learn
* OpenCV
* Numpy
* Matplotlib
* Imutils
* Seaborn

## Results showcase
<img src="https://github.com/21zasker/Captcha-Solver/blob/main/Screenshot/Captcha_Solver_Showcase.jpg" width="65%" alt="Showcase">

## Note
The effectiveness of image processing may vary depending on the quality and characteristics of the CAPTCHA images. Factors such as noise, distortion, and image cleanliness can impact the results. Adjustments to preprocessing steps might be necessary based on the specific dataset and image conditions.
