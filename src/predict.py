"""
Loads a trained model and classifies new images with visualization.
"""
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
from model import CNNFeatureExtractor  # Import the same CNN model used in training

# 🔹 Load trained models & preprocessing tools
scaler = joblib.load("models/scaler.pkl")  # StandardScaler (used during training)
svm_model = joblib.load("models/svm_model.pkl")  # SVM Model
rf_model = joblib.load("models/random_forest_model.pkl")  # Random Forest Model
knn_model = joblib.load("models/knn_model.pkl")  # k-NN Model

# 🔹 Load dataset folder names dynamically (to map label index → country name)
dataset_path = "feature_extracting"  # Ensure this is the correct dataset directory
country_names = sorted([name for name in os.listdir(dataset_path) if not name.startswith('.')])  # Ignore hidden files

# 🔹 Load the trained CNN model (used for feature extraction)
num_classes = len(country_names)  # Update dynamically
model = CNNFeatureExtractor(num_classes)
model.load_state_dict(torch.load("models/country_classifier.pth"))
model.eval()  # Set to evaluation mode

# 🔹 Define preprocessing transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Must match training size
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

def predict_country(image_path, model_name="rf"):
    """ Predicts the country of the given image using the selected model and visualizes results. """
    features = extract_features(image_path)

    # Select the classifier
    model_dict = {
        "svm": svm_model,
        "rf": rf_model,
        "knn": knn_model
    }

    if model_name not in model_dict:
        raise ValueError("Invalid model choice. Use 'svm', 'rf', or 'knn'.")

    model = model_dict[model_name]
    prediction = model.predict(features)[0]  # Get the predicted label
    predicted_country = country_names[prediction]  # Map index to country name

    # 🔹 Display Image with Prediction
    image = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Predicted Country ({model_name.upper()}): {predicted_country}", fontsize=14, fontweight='bold')
    plt.show()

    # 🔹 If model supports probability prediction, show top predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]  # Top 3 predictions
        top_countries = [country_names[i] for i in top_indices]
        top_probs = probabilities[top_indices]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=top_probs, y=top_countries, palette="Blues_r")
        plt.xlabel("Prediction Probability")
        plt.ylabel("Country")
        plt.title(f"Top 3 Country Predictions ({model_name.upper()})")
        plt.show()

    print(f"✅ Predicted Country ({model_name.upper()}): {predicted_country}")
    return predicted_country

def predict_image(image_path, model_name="rf"):
    """
    Predicts the class of the image and visualizes results.
    :param image_path: path to the image
    :param model_name: model to use (default: rf)
    :return: class of the image
    """
    return predict_country(image_path, model_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict country from an image path with visualization.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("--model", type=str, choices=["svm", "rf", "knn"], default="rf",
                        help="Model to use for prediction (default: rf).")
    args = parser.parse_args()

    predict_image(args.image_path, args.model)
