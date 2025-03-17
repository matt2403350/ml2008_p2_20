"""
loads a trained model and classifies new images
"""
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torchmetrics
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from model import CNNFeatureExtractor  # Import the same CNN model used in training

# 🔹 Load trained models & preprocessing tools
scaler = joblib.load("models/scaler.pkl")  # StandardScaler (used during training)
svm_model = joblib.load("models/svm_model.pkl")  # SVM Model
rf_model = joblib.load("models/random_forest_model.pkl")  # Random Forest Model
knn_model = joblib.load("models/knn_model.pkl")  # k-NN Model
#xgb_model = joblib.load("models/xgboost_model.pkl")  # XGBoost Model

# 🔹 Load dataset folder names dynamically (to map label index → country name)
dataset_path = "feature_extracting"  # Ensure this is the correct dataset directory

country_names = sorted([name for name in os.listdir(dataset_path) if not name.startswith('.')]) #ignore hidden files

# 🔹 Load the trained CNN model
num_classes = 6  # Update based on dataset size
model = CNNFeatureExtractor(num_classes)  # Load same CNN model used in training
model.load_state_dict(torch.load("models/country_classifier.pth"))
model.eval()  # Set to evaluation mode

# 🔹 Define preprocessing transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Same size as CNN training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_features(image_path):
    """ Extract CNN features from an external image (same as training). """
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Convert to tensor with batch dimension

    # Extract CNN features
    with torch.no_grad():
        features = model(image, extract_features=True).numpy()

    print(f"Extracted CNN Feature Shape: {features.shape}")  # Debugging print

    # Apply StandardScaler (to match training)
    features = scaler.transform(features)

    print(f"Feature Shape After Scaling: {features.shape}")  # Debugging print
    return features

def predict_country(image_path, model_name="svm"):
    """ Predicts the country of the given image using the selected model. """
    features = extract_features(image_path)

    # Select the classifier
    if model_name == "svm":
        model = svm_model
    elif model_name == "rf":
        model = rf_model
    elif model_name == "knn":
        model = knn_model
    #elif model_name == "xgb":
        #model = xgb_model
    else:
        raise ValueError("Invalid model choice. Use 'svm', 'rf' or 'knn'")

    # Predict
    prediction = model.predict(features)[0]

    # Get corresponding country name
    country_name = country_names[prediction]

    print(f"✅ Predicted Country ({model_name.upper()}): {country_name}")
    return country_name

def predict_image(image_path, model_name="rf"):
    """
    Predicts the class of the image
    :param image_path: path to the image
    :param model_name: model to use (default: svm)
    :return: class of the image
    """
    return predict_country(image_path, model_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict country from an image path.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("--model", type=str, choices=["svm", "rf", "knn"], default="rf",
                        help="Model to use for prediction (default: rf).")
    args = parser.parse_args()

    predict_image(args.image_path, args.model)
