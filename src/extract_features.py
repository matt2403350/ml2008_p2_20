import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNNFeatureExtractor

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = datasets.ImageFolder(root="src/feature_extracting", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load trained CNN model
num_classes = len(dataset.classes)
model = CNNFeatureExtractor(num_classes)
model.load_state_dict(torch.load("src/models/country_classifier.pth"))
model.eval()

# Extract features
features = []
labels = []

with torch.no_grad():
    for images, lbls in dataloader:
        feats = model(images, extract_features=True)  # Extract features
        features.append(feats.numpy())
        labels.append(lbls.numpy())

# Convert lists to NumPy arrays
features = np.vstack(features)
labels = np.hstack(labels)

# Save extracted features for ML models
np.save("src/models/features.npy", features)
np.save("src/models/labels.npy", labels)
print("Feature extraction complete! Saved as features.npy & labels.npy")
