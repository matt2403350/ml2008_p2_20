"""
Trains the model using SVM and saves it to the models directory.
"""

import joblib
import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the train-test split dataset prepared by dataset.py
train_data_path = "train_dataset"
test_data_path = "test_dataset"

# Image size from dataset.py
img_size = (224, 224)

# Define HOG feature extraction parameters
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

def extract_features_from_directory(directory):
    """
    Extracts HOG features from images in a given directory.
    """
    X, y = [], []
    for country in sorted(os.listdir(directory)):  # Sort to maintain label consistency
        country_path = os.path.join(directory, country)
        if os.path.isdir(country_path):
            for image_name in os.listdir(country_path):
                image_path = os.path.join(country_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"⚠️ Warning: Could not read {image_path}, skipping.")
                    continue

                image = cv2.resize(image, img_size)  # Resize for consistency
                features = hog(image, **hog_params)  # Extract HOG features

                X.append(features)
                y.append(country)

    return np.array(X), np.array(y)

# Extract features from train and test sets
print("🚀 Extracting HOG features from training images...")
X_train, y_train = extract_features_from_directory(train_data_path)
print("✅ Training features extracted.")

print("🚀 Extracting HOG features from test images...")
X_test, y_test = extract_features_from_directory(test_data_path)
print("✅ Test features extracted.")

# Standardize features (SVM performs better with scaled data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an optimized SVM model
print("🚀 Training SVM...")
svm_model = SVC(kernel='rbf', C=50, gamma='scale')  # RBF kernel for non-linear separation
svm_model.fit(X_train, y_train)

# Evaluate SVM model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ SVM Model Accuracy: {accuracy:.4f}")

# Save the trained SVM model and preprocessing tools
# Save the extracted dataset
os.makedirs("models", exist_ok=True)
joblib.dump((X_train, X_test, y_train, y_test), "models/dataset.pkl")  # ✅ Save dataset
print("✅ Dataset saved as dataset.pkl")

# Save the trained SVM model and preprocessing tools
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(sorted(os.listdir(train_data_path)), "models/label_encoder.pkl")  # Save label mapping
print("✅ SVM model and preprocessing tools saved successfully!")

print("✅ SVM model and preprocessing tools saved successfully!")
