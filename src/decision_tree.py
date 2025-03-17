import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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

# Flatten the images and concatenate with features
def prepare_data(images, features):
    images_flat = images.view(images.size(0), -1)  # Flatten images to (batch_size, 3*224*224)
    combined_data = torch.cat((images_flat, features), dim=1)  # Concatenate along feature dimension
    return combined_data


def train(train_dataset):
    # Create a DataLoader for the train dataset
    train_loader = DataLoader(train_dataset, shuffle=True)

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
    joblib.dump(dt_model, "src/models/decision_tree3.pkl")

    # Performance check on training set
    train_accuracy = dt_model.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy:.4f}")

def test(test_dataset):
    # load trained model
    dt_model = joblib.load("src/models/decision_tree3.pkl")

    test_loader = DataLoader(test_dataset, shuffle=True)

    # Collect testing data
    X_list, y_list = [], []
    for images, features, labels in test_loader:
        X = prepare_data(images, features).numpy()
        y = labels.numpy()
        X_list.append(X)
        y_list.append(y)

    # Convert collected batches into single arrays
    X_test = np.vstack(X_list)
    y_test = np.hstack(y_list)

    print("Final testing data shape:", X_test.shape, y_test.shape)

    y_pred = dt_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Decision Tree Classification Report:")
    print(classification_report(y_test, y_pred))

    return

def main():
    # Define transformations for image preprocessing
    transform = transforms.Compose([
        transforms.Resize((768, 331)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    # Load the pre-extracted features
    features = npy_loader("src/models/features/features.npy")
    print("Features shape:", features.shape)

    # Load the raw image dataset
    images = datasets.ImageFolder(root="src/feature_extracting", transform=transform)
    print("Number of images in dataset:", len(images))

    # Create the combined dataset
    combined_dataset = CombinedDataset(images, features)

    train_dataset, test_dataset = train_test_split(combined_dataset, test_size=0.2, random_state=42)

    train(train_dataset)
    test(test_dataset)

if __name__ == "__main__":
    main()