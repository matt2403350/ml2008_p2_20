import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.metrics import log_loss

# Custom dataset to combine raw images and pre-extracted features
class CombinedDataset(Dataset):
    def __init__(self, image_dataset, features):
        """
        Args:
            image_dataset (Dataset): A PyTorch dataset (e.g., ImageFolder) for raw images.
            features (np.ndarray): Pre-extracted features corresponding to the images.
        """
        self.image_dataset = image_dataset
        self.features = features

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Get the raw image and label
        image, label = self.image_dataset[idx]

        # Get the corresponding feature vector
        feature = self.features[idx]

        # Return the combined data
        return image, feature, label

# Load pre-extracted features
def npy_loader(path):
    return np.load(path)

def main():
    # Define transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Load the pre-extracted features
    features = npy_loader("src/models/features/features.npy")
    print("Features shape:", features.shape)

    # Load the raw image dataset
    train_dataset = datasets.ImageFolder(root="src/train_dataset", transform=transform)
    print("Number of images in dataset:", len(train_dataset))

    # Create the combined dataset
    combined_dataset = CombinedDataset(train_dataset, features)

    # Create a DataLoader for the combined dataset
    train_loader = DataLoader(combined_dataset, shuffle=True)

    # Example: Iterate through the DataLoader
    # for batch_idx, (images, features, labels) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}:")
    #     print("Images shape:", images.shape)  # Should be (batch_size, 3, 224, 224)
    #     print("Features shape:", features.shape)  # Should be (batch_size, feature_dim)
    #     print("Labels shape:", labels.shape)  # Should be (batch_size,)
    #     break  # Just show the first batch for demonstration

    # Flatten the images and concatenate with features
    def prepare_data(images, features):
        images_flat = images.view(images.size(0), -1)  # Flatten images to (batch_size, 3*224*224)
        combined_data = torch.cat((images_flat, features), dim=1)  # Concatenate along feature dimension
        return combined_data

    # Collect training data
    X_list, y_list = [], []
    for images, features, labels in train_loader:
        X = prepare_data(images, features).numpy()
        y = labels.numpy()
        X_list.append(X)
        y_list.append(y)

    # Convert collected batches into single arrays
    X_train = np.vstack(X_list)
    y_train = np.hstack(y_list)

    print("Final training data shape:", X_train.shape, y_train.shape)

    # Train decision tree
    dt_model = DecisionTreeClassifier(max_depth=12, random_state=42)
    dt_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(dt_model, "src/models/decision_tree2.pkl")

    # Performance check on training set
    train_accuracy = dt_model.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")

if __name__ == "__main__":
    main()