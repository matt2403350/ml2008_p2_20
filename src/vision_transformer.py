import torch
import os
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.utils.data import DataLoader, Dataset

# Custom dataset to combine raw images and pre-extracted features
class CombinedDataset(Dataset):
    def __init__(self, image_dataset, pre_extracted_features=None):
        """
        Args:
            image_dataset (Dataset): A PyTorch dataset (e.g., ImageFolder) for raw images.
            pre_extracted_features (np.ndarray): Pre-extracted features corresponding to the images.
        """
        self.image_dataset = image_dataset
        self.pre_extracted_features = pre_extracted_features

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        if self.pre_extracted_features is not None:
            feature = self.pre_extracted_features[idx]
            return image, feature, label
        else:
            return image, label
        
# Load pre-extracted .npy features
def npy_loader(path):
        return np.load(path)

# Extract features from ViT Model
def extract_features(model, images):
    with torch.no_grad():
        outputs = model(images, output_hidden_states=True)
        features = outputs.hidden_states[-1][:, 0, :]
    return features

# Save extracted features
def save_features(model, dataloader, save_path, pre_extracted_features=None):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if pre_extracted_features is not None:
                images, features, labels = batch
                features = features
            else:
                images, labels = batch

            # No need for device handling, assume running on CPU
            vit_features = extract_features(model, images)

            if pre_extracted_features is not None:
                # Combine ViT features with pre-extracted features
                combined_features = torch.cat((vit_features, features), dim=1)
                all_features.append(combined_features.cpu().numpy())
            else:
                all_features.append(vit_features.cpu().numpy())

            all_labels.append(labels.cpu().numpy())

    # Concatenate all features and labels
    all_features = np.vstack(all_features)
    all_labels = np.hstack(all_labels)

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save features and labels
    np.save(os.path.join(save_path, "features.npy"), all_features)
    np.save(os.path.join(save_path, "labels.npy"), all_labels)
    print(f"Features saved to {save_path}")

def main():
    # Define transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ViT input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Load the pre-extracted features (if available)
    pre_extracted_features = None
    if os.path.exists("src/models/features.npy"):
        pre_extracted_features = npy_loader("src/models/features.npy")
        print("Pre-extracted features shape:", pre_extracted_features.shape)

    # Load the raw image dataset
    train_dataset = datasets.ImageFolder(root="train_dataset", transform=transform)
    print("Number of images in dataset:", len(train_dataset))

    # Create the combined dataset
    combined_dataset = CombinedDataset(train_dataset, pre_extracted_features)

    # Create a DataLoader for the dataset
    train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=False)

    # Load a pre-trained Vision Transformer model
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)

    # Extract and save features
    save_features(model, train_loader, "src/models/features", pre_extracted_features)

if __name__ == "__main__":
    main()

