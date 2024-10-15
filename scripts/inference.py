import torch
from torchvision import transforms
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from model_builder import (
    model_resnet18,
    model_beit,
    model_inception_resnet_v2,
    model_mobilenet_v2,
    model_swin,
)
from utils import save_predictions_to_excel  # Keep the same save function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths

# model_paths = [
#     '../capsule-vision-2024/models/ResNet18_best.pth',
#     '../capsule-vision-2024/models/BEiT_best.pth',
#     '../capsule-vision-2024/models/InceptionResNetV2_best.pth',
#     '../capsule-vision-2024/models/MobileNetV2_best.pth',
#     '../capsule-vision-2024/models/SwinTransformer_best.pth'
# ]

model_paths = [
    '/kaggle/input/capsule-vision-2024-models/pytorch/default/1/ResNet18_best.pth',
    '/kaggle/input/capsule-vision-2024-models/pytorch/default/1/BEiT_best.pth',
    '/kaggle/input/capsule-vision-2024-models/pytorch/default/1/InceptionResNetV2_best.pth',
    '/kaggle/input/capsule-vision-2024-models/pytorch/default/1/MobileNetV2_best.pth',
    '/kaggle/input/capsule-vision-2024-models/pytorch/default/1/SwinTransformer_best.pth'
]

# Model classes
model_classes = [model_resnet18, model_beit, model_inception_resnet_v2, model_mobilenet_v2, model_swin]

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
    image_paths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.lower().endswith(('jpg'))]
    dataset = TestDataset(image_paths, image_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    return dataloader, image_paths

# Ensemble Inference on the test set
def ensemble_test_inference(models, dataloader, device):
    all_predictions = []
    all_image_paths = []

    with torch.no_grad():
        for X, image_paths in tqdm(dataloader, total=len(dataloader), desc="Ensemble Test Inference"):
            X = X.to(device)

            # Initialize ensemble predictions
            ensemble_preds = None

            # Accumulate predictions from each model
            for model in models:
                model.eval()
                y_pred_logits = model(X)
                y_pred_probs = torch.softmax(y_pred_logits, dim=1)  # Convert logits to probabilities

                if ensemble_preds is None:
                    ensemble_preds = y_pred_probs
                else:
                    ensemble_preds += y_pred_probs

            # Average predictions over all models
            ensemble_preds /= len(models)

            all_predictions.append(ensemble_preds.cpu())  # Collect predictions
            all_image_paths.extend(image_paths)  # Collect image paths

    predictions = torch.cat(all_predictions, dim=0)
    return predictions, all_image_paths

def main():
    # Load ensemble models
    models = [load_model(cls, path, device) for cls, path in zip(model_classes, model_paths)]

    # Directory containing test images
    # test_path = "../capsule-vision-2024/data/Testing set/Images"
    test_path = "/kaggle/input/capsule-vision-2020-test/Testing set/Images"  # Update with actual path
    dataloader, image_paths = load_test_data(test_path)

    # Run ensemble inference on the test set
    predictions, image_paths = ensemble_test_inference(models, dataloader, device)

    # Save predictions to Excel
    # output_test_predictions = "../capsule-vision-2024/submission/test_excel.xlsx"
    output_test_predictions = "/kaggle/working/test_excel.xlsx"
    save_predictions_to_excel(image_paths, predictions, output_test_predictions)

    print(f"Predictions saved to {output_test_predictions}")

if __name__ == "__main__":
    main()