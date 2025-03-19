import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from transformers import ViTForImageClassification, ViTConfig
import torchvision.transforms as transforms
from PIL import Image

# Load extracted features and labels
features = np.load("models/features.npy")
labels = np.load("models/labels.npy")

# Check the shape of features and calculate expected size
print("Shape of features before reshaping:", features.shape)

expected_size_per_image = 316 * 316 * 3  # RGB image of size 316x316

# Ensure the number of features is compatible with expected size
num_elements = features.size
num_samples = num_elements // expected_size_per_image

if num_elements % expected_size_per_image == 0:
    # Reshape the features into images
    features = features.reshape(num_samples, 316, 316, 3)
    print("Features reshaped to:", features.shape)
else:
    print(f"The total number of elements ({num_elements}) is not divisible by {expected_size_per_image}.")
    print(f"Consider trimming or padding the data to match the expected number of elements.")
    exit()

# Reshape and preprocess the features (make sure they are in RGB format and 316x316 if flattened) ---
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),  # Resize to 32x32
    transforms.ToTensor()  # Convert to Tensor
])

# Ensure that each image is properly reshaped and resized
features_resized = []
for image in features:
    image_pil = Image.fromarray(image.astype(np.uint8))  # Convert to PIL image
    image_resized = resize_transform(image_pil)  # Apply the resize transform
    features_resized.append(image_resized)

# Convert list to a numpy array for further processing
features_resized = np.array(features_resized)

# Ensure that the shape is (num_samples, 3, 32, 32)
features_resized = features_resized.reshape(-1, 3, 32, 32)

# Standardize features 
features_resized_flat = features_resized.reshape(features_resized.shape[0], -1)  # Flatten the images
scaler = StandardScaler()
features_resized_flat = scaler.fit_transform(features_resized_flat)  # Standardize the features
features_resized = features_resized_flat.reshape(-1, 3, 32, 32)  # Reshape back to 3-channel images

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_resized, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for batches
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the Vision Transformer model 
class VisionTransformerModel(nn.Module):
    def __init__(self, num_classes=6):
        super(VisionTransformerModel, self).__init__()
        config = ViTConfig(
            image_size=32,  # Image size after resizing
            patch_size=16,  # Patch size, should be smaller than the image
            num_classes=num_classes,  # Number of output classes
            num_hidden_layers=12,  # Transformer depth
            num_attention_heads=12,  # Number of attention heads
            hidden_size=768,  # Hidden size of the transformer
            intermediate_size=3072,  # Feedforward layer size
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.model = ViTForImageClassification(config)

    def forward(self, x):
        return self.model(x).logits  # We only need the logits (raw predictions)

# Initialize the model, loss function, and optimizer 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformerModel(num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model():
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# Evaluate the model function
def evaluate_model():
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    return np.array(predictions)

# Train and evaluate the model
train_model()
predictions = evaluate_model()
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
print("Classification Report:")
print(classification_report(y_test, predictions))
