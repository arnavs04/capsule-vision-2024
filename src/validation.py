import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
from metrics import generate_metrics_report
from data_setup import VCEDataset
from model_builder import *

# Hyperparameters
batch_size = 32
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KAGGLE = True

# File paths
test_xlsx = 'validation_data.xlsx'
test_root_dir = 'validation'
data_root_dir = "../capsule-vision-2024/data/Dataset"
metrics_report_dir = "../capsule-vision-2024/reports/metrics_report.json"
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
    data_root_dir = '/kaggle/input/capsule-vision-2024-data/Dataset'
    metrics_report_dir = "/kaggle/working/metrics_report.json"





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

# New Dataset class to include image paths
class VCEDatasetWithPaths(torch.utils.data.Dataset):
    def __init__(self, xlsx_file, root_dir, train_or_test: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.xlsx_file_path = os.path.join(self.root_dir, train_or_test, xlsx_file)
        self.annotations = pd.read_excel(io=self.xlsx_file_path, sheet_name=0)
        self.class_columns = self.annotations.columns[2:]  # Assuming class columns start from the 3rd column
        self.num_classes = len(self.class_columns)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get image path
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0].replace("\\", "/"))
        only_image_path = self.annotations.iloc[index, 0]
        # Load the image and ensure it's in RGB format
        image = Image.open(img_path).convert('RGB')

        # Get the target label, assuming one-hot encoding in the Excel
        target = self.annotations.iloc[index, 2:].values
        y_label = torch.tensor(target.argmax(), dtype=torch.long)

        # Apply transformation, if provided
        if self.transform:
            image = self.transform(image)

        # Return image, label, and the image path
        return image, y_label, only_image_path

# Preprocess data using VCEDatasetWithPaths
def preprocess_data_with_paths(xlsx_file, data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = VCEDatasetWithPaths(
        xlsx_file=xlsx_file,
        root_dir=data_dir,
        train_or_test='validation',
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return dataloader, val_dataset


# Load models
def load_model(model_class, model_path, device):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model


# Ensemble inference step
def ensemble_test_step(models, dataloader, device):
    all_predictions = []
    all_labels = []
    all_image_paths = []

    # Set all models to eval mode once before inference
    for model_idx, model in enumerate(models):
        model.eval()
        print(f"Model {model_idx + 1} set to eval mode.")

    with torch.no_grad():
        # Initialize the progress bar with total number of batches, leave=True ensures it stays after completion
        with tqdm(total=len(dataloader), desc="Ensemble Inference", unit="batch", leave=True, miniters=1, smoothing=0) as pbar:
            for batch_idx, (X, y, image_paths) in enumerate(dataloader):

                # Move data to device (GPU or CPU)
                X, y = X.to(device), y.to(device)

                # Calculate predictions for all models and average them
                ensemble_preds = torch.stack([
                    torch.softmax(model(X), dim=1)  # Softmax for probabilities
                    for model_idx, model in enumerate(models)
                ]).mean(dim=0)  # Average over the models

                
                # Print first 5 predictions for debugging

                # Append predictions, labels, and image paths to lists
                all_predictions.append(ensemble_preds.cpu())
                all_labels.append(y.cpu())
                all_image_paths.extend(image_paths)


                # Update the progress bar for each batch
                pbar.update(1)

    # Concatenate all predictions and labels
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return predictions, labels, all_image_paths


# Save predictions to Excel
def save_predictions_to_excel(image_paths, y_pred: torch.Tensor, output_path: str):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    
    # Convert logits to class predictions
    y_pred_classes = y_pred.argmax(dim=1).cpu().numpy()
    
    # Create a DataFrame to store image paths, predicted class, and prediction probabilities
    df = pd.DataFrame({
        'image_path': image_paths,
        'predicted_class': [class_columns[i] for i in y_pred_classes],
        **{col: y_pred[:, i].cpu().numpy() for i, col in enumerate(class_columns)}
    })
    
    # Save to Excel file
    df.to_excel(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# Main function
def main():
    models = [load_model(cls, path, device) for cls, path in zip(model_classes, model_paths)]

    # Use the new preprocessing function that returns image paths
    dataloader, dataset = preprocess_data_with_paths(test_xlsx, data_root_dir)

    # Run the ensemble test step, which now also returns image paths
    predictions, true_labels, image_paths = ensemble_test_step(models, dataloader, device)

    # Generate metrics report (using logits as predictions)
    metrics_report = generate_metrics_report(true_labels, predictions)
    print("Metrics Report:\n", metrics_report)

    with open(metrics_report_dir, 'w') as f:
        f.write(metrics_report)

    print(f"Metrics report saved to {metrics_report_dir}.")

    # Save predictions to Excel (using argmax of predictions)
    # output_val_predictions = "../capsule-vision-2024/reports/validation_excel.xlsx"
    output_val_predictions = "/kaggle/working/validation_excel.xlsx"
    save_predictions_to_excel(image_paths, predictions, output_val_predictions)


# Run the script
if __name__ == "__main__":
    main()