# ml2008
INF2008 Machine Learning Project on Country Classification

# Scripts for Image Classification using CNN as feature extractor and classical machine learning models for classification.


1. dataset.py:
Script for creating and splitting the image dataset into train and test dataset, with test dataset consisting of 5% of the images. Images saved in subfolders named after their respective labels.

2. model.py:
Script for creating a Convolutional Neural Network for Feature Extraction, model is saved as a pickle file.

3. extract_features.py:
Script for extracting features using the CNN model, from the raw image data in the train and test datasets. Then saving the extracted features as a pickle file for classification models down the line. (features.npy and labels.npy)

4. train_classical.py:
Script for training the classical machine learning models and classifying the images. 
utilizes feature extracted from the CNN model.

5. predict.py:
Script for predicting the class of an image using the saved .pkl models.


# **STEP-BY-STEP GUIDE**
Take Note: Everything should be run through the terminal from main.py

Run main.py -h or --help if you need help with the commands.

Please Run the following in Order:

**STEP 1:**
Feature Extracting using CNN model (Will Take 1-5Minutes)
- Run in Terminal "python3/python main.py --extract_features"

**STEP 2:**
Training the Classical Machine Learning Models (Will Take 3-5Minutes)
- Run in Terminal " python3/python main.py --train_classical "

Evaluation of Confusion Matrix Should be Displayed

**STEP 3:**
Predicting the class of an image
- Run in Terminal " python3/python main.py --predict "FILE-PATH" --model "rf" "
- Sample Images are in src/predict

Image should be classified and displayed as well as the likelyhood of the country that will be predicted.
You may also add additional images to the predict folder and run the predict script to classify them.

**Done!**











