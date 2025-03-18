# ml2008
INF2008 Machine Learning Project on Country Classification

# Scripts for Image Classification using CNN as feature extractor and classical machine learning models for classification.

0. load_data.ipynb:
Jupyter notebook used for fetching data from Huggingface and the initial data exploration.

1. dataset.py:
Script for creating and splitting the image dataset into train and test dataset, with test dataset consisting of 5% of the images. Images saved in subfolders named after their respective labels.

2. model.py:
Script for creating a Convolutional Neural Network for Feature Extraction, model is saved as a pickle file.

3. extract_features.py:
Script for extracting features using the CNN model, from the raw image data in the train and test datasets. Then saving the extracted features as a pickle file for classification models down the line. (features.npy and labels.npy)

4. train_classical.py:
Script for training the classical machine learning models and classifying the images. 
utilizes feature extracted from the CNN model.

5. evaluate.py
Script used for evaluating the accuracy of trained machine learning models. Provides accuracy, confusion matrix, and the classification report.

6. predict.py:
Script for predicting the class of an image using the saved .pkl models.


To evaluate the models, run the following commands:
python/python3 train_classical.py

To Predict, run the following commands:
python3 predict.py "FILE-PATH" --model "rf"