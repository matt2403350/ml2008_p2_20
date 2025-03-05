"""
loads a trained model and classifies new images
"""
import os
import torch
import torch.nn as nn
import torchmetrics
from torchvision import transforms, datasets
from torch.utils.data import DataLoader



def predict_image(image_path):
    """
    Predicts the class of the image
    :param image_path: path to the image
    :return: class of the image
    """
    return None

