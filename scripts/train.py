# Imports
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import torchvision
from torchvision import transforms 

# Script Imports
import data_setup, engine
from model_builder import (
    model_beit, 
    model_cswin, 
    model_swin, 
    model_inception_resnet_v2,
    model_resnet18,
    model_mobilenet_v2, 
)
from utils import *

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_UNITS = 32
LEARNING_RATE = 0.003
NUM_WORKERS = 0

# Setup directories
data_dir = "../capsule-vision-2024/data/Dataset"
train_dir = "training"
test_dir = "validation"
train_xlsx_filename = "training_data.xlsx"
test_xslx_filename = "validation_data.xslx"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3)
])

# Create DataLoaders with help from data_setup.py
train_loader, test_loader = data_setup.create_dataloaders(
    train_xlsx=train_xlsx_filename,
    test_xlsx=test_xslx_filename,
    train_root_dir=train_dir,
    test_root_dir=test_dir,
    data_root_dir=data_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

# Class labels (assuming these are your target classes)
class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

# Define a list of models for training
model_list = {
    "ResNet18": model_resnet18(pretrained=True, num_classes=len(class_columns)),
    "InceptionResNetV2": model_inception_resnet_v2(pretrained=True, num_classes=len(class_columns)),
    "MobileNetV2": model_mobilenet_v2(pretrained=True, num_classes=len(class_columns)),
    "Swin Transformer": model_swin(pretrained=True, num_classes=len(class_columns)),
    "CSwin Transformer": model_cswin(pretrained=True, num_classes=len(class_columns)),
    "BEiT": model_beit(pretrained=True, num_classes=len(class_columns))
}

# Dictionary to store results for each model
results_dict = {}

# Loop through each model and train them
for model_name, model in model_list.items():
    print(f"\nTraining model: {model_name}")
    
    # Move model to target device
    model = model.to(device)
    
    # Define optimizer (AdamW as an example) and loss function (Cross Entropy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    loss_fn = F.cross_entropy
    
    # Train the model using engine.train function
    results = engine.train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device, 
        model_name=model_name,
    )
    
    # Store the results
    results_dict[model_name] = results

# Results will contain training history for each model