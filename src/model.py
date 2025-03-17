"""
defines a model class that will be used to train the model
CNN Model - Convoluted Neural Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):

    def __init__(self, num_classes=6):
        super(CNNFeatureExtractor, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 128)  # Adjust based on input image size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, extract_features=True):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten

        if extract_features:
            return x

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



