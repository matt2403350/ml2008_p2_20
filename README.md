# ml2008
INF2008 Machine Learning Project on Country Classification

# Scripts for Image Classification using Convolutional Neural Networks

1. dataset.py:
Script for creating a dataset from the images in the dataset folder. The dataset is saved as a pickle file.

2. model.py:
Script for creating a Convolutional Neural Network for Feature Extraction, model is saved as a pickle file.

3. extract_features.py:
Script for extracting features from the CNN model and saving them as a pickle file. (features.npy and labels.npy)

4. train_classical.py:
Script for training/evaluating the classical machine learning models and classifying the images. 
utilizes feature extracted from the CNN model.

5. predict.py:
Script for predicting the class of an image using the saved .pkl models.