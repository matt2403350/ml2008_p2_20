"""
Predicts the country of an image using the trained SVM model.
"""

import joblib
import cv2
import numpy as np
from skimage.feature import hog
import sys

# Load trained model and preprocessing tools
svm_model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Define HOG feature extraction parameters (must match `dataset.py`)
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# Image size (must match dataset preprocessing)
image_size = (224, 224)

def predict_image(image_path):
    """
    Predicts the country of the given image using the trained SVM model.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"❌ Error: Cannot read image {image_path}. Please check the file path.")
        return None

    # Resize and extract HOG features
    image = cv2.resize(image, image_size)
    features = hog(image, **hog_params).reshape(1, -1)

    # Scale features
    features = scaler.transform(features)

    # Predict using SVM
    prediction = svm_model.predict(features)[0]  # ✅ Already a country name

    return prediction  # ✅ No need for indexing

# Run prediction from command line
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    result = predict_image(image_path)
    if result:
        print(f"🎯 Predicted Country: {result}")
else:
    print("Usage: python predict.py <image_path>")
