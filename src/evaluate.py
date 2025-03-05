"""
Tests model accuracy
"""
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model import CNNClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load test dataset
test_dataset = datasets.ImageFolder(root="src/test_dataset", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
num_classes = len(test_dataset.classes)
model = CNNClassifier(num_classes)
model.load_state_dict(torch.load("src/models/country_classifier.pth"))
model.eval()


def evaluate_model():
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))


if __name__ == "__main__":
    evaluate_model()
