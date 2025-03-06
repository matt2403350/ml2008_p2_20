"""
trains the model and saves to models directory
"""
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from src.model import CNNClassifier
#from datasets import load_dataset

def hf_transform(dataset):
    dataset['country'] = dataset['label']
    
    return dataset

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

"""
# Load dataset
hf_ds = load_dataset("marcelomoreno26/geoguessr")
hf_train = hf_ds['train']
hf_train = hf_train.map(hf_transform, remove_columns=['label'], batched=True)
COUNTRY_LIST = ['Singapore', 'Malaysia', 'Indonesia', 'Thailand', 'Philippines', 'Cambodia']
hf_train = hf_train.filter(lambda x: x['country'] in COUNTRY_LIST)
"""

train_dataset = datasets.ImageFolder(root="src/train_dataset", transform=transform)
"""combined_dataset = ConcatDataset([hf_train, train_dataset])"""

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)



# Model setup
num_classes = len(train_dataset.classes)  # Automatically detect number of classes
model = CNNClassifier(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10  # Number of training epochs

loss_history = []


def train_model():

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Get file paths from dataset
            #file_paths = [train_dataset.samples[i][0] for i in labels.tolist()]

            # Print image paths and their corresponding labels
            print(f"Training Labels: {images}")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        with open ("loss_history.txt", "a") as f:
            f.write(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}\n")

    print(f"Loss History saved to {os.path.abspath('loss_history.txt')}")



    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/country_classifier.pth")
    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()
