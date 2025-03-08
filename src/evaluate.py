"""
Evaluates the trained SVM model and tests accuracy.
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset (preprocessed by dataset.py)
X_train, X_test, y_train, y_test = joblib.load("models/dataset.pkl")

# Load trained SVM model
svm_model = joblib.load("models/svm_model.pkl")

# Predict on test set
y_pred = svm_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {accuracy:.4f}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Load class labels (countries)
label_encoder = joblib.load("models/label_encoder.pkl")
class_names = label_encoder

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))
