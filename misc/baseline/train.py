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
from docs.baseline.model_builder import (
    model_beit, 
    model_swin, 
    model_inception_resnet_v2,
    model_resnet18,
    model_mobilenet_v2, 
)
from utils import *
from metrics import *
print("Libraries Imported Successfuly!\n\n")

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_UNITS = 32
LEARNING_RATE = 0.003
NUM_WORKERS = 4

# Reproducibility
seed_everything(seed=42)

# Setup directories
train_dir = "training"
test_dir = "validation"
train_xlsx_filename = "training_data.xlsx"
test_xlsx_filename = "validation_data.xlsx"

# data_dir = "../capsule-vision-2024/data/Dataset"
data_dir="kaggle/input/capsule-vision-2024-data/Dataset"
# save_dir = "../capsule-vision-2024/models"
save_dir="kaggle/working/models"
# logging_dir = "../capsule-vision-2024/logs"
logging_dir="kaggle/working/logs"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n\n")


data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),  # Convert the image to a tensor here
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3)
])

# Create DataLoaders with help from data_setup.py
train_loader, test_loader = data_setup.create_dataloaders(
    train_xlsx=train_xlsx_filename,
    test_xlsx=test_xlsx_filename,
    train_root_dir=train_dir,
    test_root_dir=test_dir,
    data_root_dir=data_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
print("Data Loaded!\n\n")


# Class labels (assuming these are your target classes)
class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

# Define a list of models for training
model_list = {
    "ResNet18": model_resnet18(pretrained=True, num_classes=len(class_columns)),
    "InceptionResNetV2": model_inception_resnet_v2(pretrained=True, num_classes=len(class_columns)),
    "MobileNetV2": model_mobilenet_v2(pretrained=True, num_classes=len(class_columns)),
    "Swin Transformer": model_swin(pretrained=True, num_classes=len(class_columns)),
    "BEiT": model_beit(pretrained=True, num_classes=len(class_columns))
}
print("Models Loaded!\n\n")


# Dictionary to store results for each model
results_dict = {}

# Loop through each model and train them
for model_name, model in model_list.items():
    print(f"\nTraining model: {model_name}")
    
    # Move model to target device
    model = model.to(device)
    
    # Define optimizer (AdamW as an example) and loss function (Cross Entropy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    loss_fn = FocalLoss() # CrossEntropyLoss()
    
    # Train the model using engine.train function
    results = engine.train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
        model_name=model_name,  # Model name to pass into logger and model saving
        save_dir=save_dir,  # Directory to save models after every 5 epochs
    )
    
    # Store the results
    results_dict[model_name] = results

# After the training loop, results_dict will contain training history for each model
print("Training complete for all models.")