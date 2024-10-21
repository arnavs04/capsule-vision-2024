import torch
from torchvision import transforms
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from model_builder import *
from utils import save_predictions_to_excel  # Keep the same save function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KAGGLE = True

# Model paths
test_path = "../capsule-vision-2024/data/Testing set/Images"
output_test_predictions = "../capsule-vision-2024/reports/test_excel.xlsx"

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

if KAGGLE is True:
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
    test_path = "/kaggle/input/capsule-vision-2020-test/Testing set/Images"  # Update with actual path


# Model classes
model_classes = [
    model_swin,         # Swin Transformer
    model_resnext,      # ResNeXt
    model_wide_resnet,  # Wide ResNet
    model_resnet,       # ResNet
    model_vit,          # Vision Transformer
    model_regnet,       # RegNet
    model_beit,         # BEiT
    model_twins_svt,    # Twins-SVT
    model_seresnet50,   # SEResNet50
    model_mobilenetv3,  # MobileNetV3
    model_mnasnet,      # MNASNet
    model_cait,         # CaiT
    model_deit,         # DeiT
    model_densenet,     # DenseNet
    model_efficientformer, # EfficientFormer
    model_inception_v3, # Inception v3
    model_convnext,     # ConvNeXt
    model_efficientnet  # EfficientNet
]


# Function to load PyTorch model
def load_model(model_class, model_path, device):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

# Preprocess the image as per PyTorch model requirements
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

# Function to load test data
def load_test_data(test_dir, image_size=(224, 224)):
    full_image_paths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.lower().endswith(('jpg'))]
    dataset = TestDataset(full_image_paths, image_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Return the dataset loader and image file names (without full path)
    image_paths = [os.path.basename(path) for path in full_image_paths]
    return dataloader, image_paths

# Ensemble Inference on the test set
def ensemble_test_inference(models, dataloader, device):
    all_predictions = []
    all_image_paths = []

    # Set all models to eval mode once before inference
    for model_idx, model in enumerate(models):
        model.eval()
        print(f"Model {model_idx + 1} set to eval mode.")
    
    with torch.no_grad():
        # Initialize the progress bar with total number of batches
        with tqdm(total=len(dataloader), desc="Ensemble Inference", unit="batch", leave=True, miniters=1, smoothing=0) as pbar:
            for batch_idx, (X, image_paths) in enumerate(dataloader):
                
                # Move data to device (GPU or CPU)
                X = X.to(device)

                # Calculate predictions for all models and average them
                ensemble_preds = torch.stack([
                    torch.softmax(model(X), dim=1)  # Softmax for probabilities
                    for model_idx, model in enumerate(models)
                ]).mean(dim=0)  # Average over the models

                # Append predictions to list (move them to CPU for easier processing)
                all_predictions.append(ensemble_preds.cpu())

                # Extract and save only the file names (not full paths)
                all_image_paths.extend([os.path.basename(path) for path in image_paths])

                # Update the progress bar for each batch
                pbar.update(1)

    # Concatenate all predictions
    predictions = torch.cat(all_predictions, dim=0)
    
    return predictions, all_image_paths


def main():
    # Load ensemble models
    models = [load_model(cls, path, device) for cls, path in zip(model_classes, model_paths)]

    # Directory containing test images
    dataloader, image_paths = load_test_data(test_path)

    # Run ensemble inference on the test set
    predictions, image_paths = ensemble_test_inference(models, dataloader, device)

    save_predictions_to_excel(image_paths, predictions, output_test_predictions)

    print(f"Predictions saved to {output_test_predictions}")

if __name__ == "__main__":
    main()