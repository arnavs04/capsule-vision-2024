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
from model_builder import *
from utils import *
from metrics import *
print("Libraries Imported Successfully!\n\n")

# Hyperparameters setup
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
KAGGLE = True  # Set to True when running on Kaggle

# Reproducibility for consistent results
seed_everything(seed=42)

# Directory setup
train_dir = "training"
test_dir = "validation"
train_xlsx_filename = "training_data.xlsx"
test_xlsx_filename = "validation_data.xlsx"

# Local directories
data_dir = "../capsule-vision-2024/data/Dataset"
save_dir = "../capsule-vision-2024/models"
logging_dir = "../capsule-vision-2024/logs"

# Kaggle-specific directories (if KAGGLE = True)
if KAGGLE:
    data_dir = "kaggle/input/capsule-vision-2024-data/Dataset"
    save_dir = "kaggle/working/models"
    logging_dir = "kaggle/working/logs"

# Setup device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n\n")

# Data augmentation and normalization for training
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
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
class_columns = [
    'Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
    'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms'
]
num_classes = len(class_columns)

# Define a list of models for training
model_list = {
    # CNN-based models
    "EfficientNet": model_efficientnet(pretrained=True, num_classes=num_classes),
    "ResNet": model_resnet(pretrained=True, num_classes=num_classes),
    "MobileNetV3": model_mobilenetv3(pretrained=True, num_classes=num_classes),
    "RegNet": model_regnet(pretrained=True, num_classes=num_classes),
    "DenseNet": model_densenet(pretrained=True, num_classes=num_classes),
    "InceptionV4": model_inception_v3(pretrained=True, num_classes=num_classes),
    "ResNeXt": model_resnext(pretrained=True, num_classes=num_classes),
    "WideResNet": model_wide_resnet(pretrained=True, num_classes=num_classes),
    "MNASNet": model_mnasnet(pretrained=True, num_classes=num_classes),
    "SEResNet50": model_seresnet50(pretrained=True, num_classes=num_classes),
    "ConvNeXt": model_convnext(pretrained=True, num_classes=num_classes),

    # Transformer-based models
    "ViT": model_vit(pretrained=True, num_classes=num_classes),
    "SwinTransformer": model_swin(pretrained=True, num_classes=num_classes),
    "DeiT": model_deit(pretrained=True, num_classes=num_classes),
    "BEiT": model_beit(pretrained=True, num_classes=num_classes),
    "CaiT": model_cait(pretrained=True, num_classes=num_classes),
    "TwinsSVT": model_twins_svt(pretrained=True, num_classes=num_classes),
    "EfficientFormer": model_efficientformer(pretrained=True, num_classes=num_classes),
}

print("Models Loaded!\n\n")

# Dictionary to store results for each model
results_dict = {}

# Loop through each model and train them
for model_name, model in model_list.items():
    print(f"\nTraining model: {model_name}")
    
    # Move model to target device (GPU/CPU)
    model = model.to(device)
    
    # Define optimizer (AdamW as an example) and loss function (FocalLoss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    loss_fn = FocalLoss()  # Optionally switch to CrossEntropyLoss if needed
    
    # Train the model using engine.train function
    results = engine.train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
        model_name=model_name,  # Model name for logging and saving
        save_dir=save_dir,  # Directory to save models every 5 epochs
    )
    
    # Store training results
    results_dict[model_name] = results

# After the loop, results_dict contains the training history for all models
print("Training complete for all models.")