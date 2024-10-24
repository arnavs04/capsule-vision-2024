import torch
from torchvision import transforms
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from model_builder import *  # Import model architecture definitions
from utils import save_predictions_to_excel  # Import save function

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KAGGLE = True  # Set to True if running on Kaggle

# Define paths for test images and the output Excel file
test_path = "../capsule-vision-2024/data/Testing set/Images"
output_test_predictions = "../capsule-vision-2024/reports/test_excel.xlsx"

# Define paths for saved model weights
model_paths = [
    '../capsule-vision-2024/models/SwinTransformer_best.pth',
    '../capsule-vision-2024/models/ResNeXt_best.pth',
    '../capsule-vision-2024/models/WideResNet_best.pth',
    '../capsule-vision-2024/models/ResNet_best.pth',
    '../capsule-vision-2024/models/ViT_best.pth',
    '../capsule-vision-2024/models/RegNet_best.pth',
    '../capsule-vision-2024/models/BEiT_best.pth',
    '../capsule-vision-2024/models/TwinsSVT_best.pth',
    '../capsule-vision-2024/models/SEResNet50_best.pth',
    '../capsule-vision-2024/models/MobileNetV3_best.pth',
    '../capsule-vision-2024/models/MNASNet_best.pth',
    '../capsule-vision-2024/models/CaiT_best.pth',
    '../capsule-vision-2024/models/DeiT_best.pth',
    '../capsule-vision-2024/models/DenseNet_best.pth',
    '../capsule-vision-2024/models/EfficientFormer_best.pth',
    '../capsule-vision-2024/models/InceptionV3_best.pth',
    '../capsule-vision-2024/models/ConvNeXt_best.pth',
    '../capsule-vision-2024/models/EfficientNet_best.pth'
]

# Update paths if running on Kaggle
if KAGGLE:
    model_paths = [
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/SwinTransformer_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/ResNeXt_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/WideResNet_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/ResNet_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/ViT_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/RegNet_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/BEiT_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/TwinsSVT_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/SEResNet50_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/MobileNetV3_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/MNASNet_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/CaiT_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/DeiT_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/DenseNet_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/EfficientFormer_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/InceptionV3_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/ConvNeXt_best.pth',
        '/kaggle/input/capsule-vision-2024-models/pytorch/updated/1/EfficientNet_best.pth'
    ]
    output_test_predictions = "/kaggle/working/Seq2Cure.xlsx"
    test_path = "/kaggle/input/capsule-vision-2020-test/Testing set/Images"

# Define corresponding model architectures
model_classes = [
    model_swin, model_resnext, model_wide_resnet, model_resnet, model_vit, model_regnet, 
    model_beit, model_twins_svt, model_seresnet50, model_mobilenetv3, model_mnasnet, 
    model_cait, model_deit, model_densenet, model_efficientformer, model_inception_v3, 
    model_convnext, model_efficientnet
]

# Function to load a model with the specified class and weights
def load_model(model_class, model_path, device):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

# Function to load and preprocess images for inference
def load_and_preprocess_image(full_path, target_size=(224, 224)):
    img = Image.open(full_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

# Custom Dataset for loading test data
class TestDataset(Dataset):
    def __init__(self, image_paths, image_size=(224, 224)):
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = load_and_preprocess_image(image_path, self.image_size)
        return img, image_path

# Function to load test images from the directory
def load_test_data(test_dir, image_size=(224, 224)):
    full_image_paths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.lower().endswith(('jpg'))]
    dataset = TestDataset(full_image_paths, image_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    image_paths = [os.path.basename(path) for path in full_image_paths]
    return dataloader, image_paths

# Perform ensemble inference using multiple models
def ensemble_test_inference(models, dataloader, device):
    all_predictions = []
    all_image_paths = []

    # Set all models to evaluation mode
    for model_idx, model in enumerate(models):
        model.eval()

    # Inference using the ensemble of models
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Ensemble Inference", unit="batch", leave=True) as pbar:
            for X, image_paths in dataloader:
                X = X.to(device)
                
                # Ensemble predictions by averaging softmax outputs
                ensemble_preds = torch.stack([torch.softmax(model(X), dim=1) for model in models]).mean(dim=0)

                all_predictions.append(ensemble_preds.cpu())  # Move predictions to CPU for processing
                all_image_paths.extend([os.path.basename(path) for path in image_paths])  # Collect image paths

                pbar.update(1)

    predictions = torch.cat(all_predictions, dim=0)  # Concatenate all batch predictions
    return predictions, all_image_paths

# Main function to run the ensemble inference and save results
def main():
    # Load all models
    models = [load_model(cls, path, device) for cls, path in zip(model_classes, model_paths)]

    # Load test data
    dataloader, image_paths = load_test_data(test_path)

    # Run ensemble inference
    predictions, image_paths = ensemble_test_inference(models, dataloader, device)

    # Save predictions to Excel file
    save_predictions_to_excel(image_paths, predictions, output_test_predictions)

    print(f"Predictions saved to {output_test_predictions}")

if __name__ == "__main__":
    main()