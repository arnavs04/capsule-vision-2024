import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
from metrics import generate_metrics_report
from data_setup import VCEDataset
from docs.baseline.model_builder import (
    model_resnet18,
    model_beit,
    model_inception_resnet_v2,
    model_mobilenet_v2,
    model_swin,
)

# Hyperparameters
batch_size = 32
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
test_xlsx = 'validation_data.xlsx'
test_root_dir = 'validation'
#data_root_dir = "../capsule-vision-2024/data/Dataset"
data_root_dir = '/kaggle/input/capsule-vision-2024-data/Dataset'

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
    '/kaggle/input/capsule-vision-2024-models/pytorch/default/1/Swin Transformer_best.pth'
]

# metrics_report_dir = "../capsule-vision-2024/reports/metrics_report.json"
metrics_report_dir = "/kaggle/working/metrics_report.json"

model_classes = [model_resnet18, model_beit, model_inception_resnet_v2, model_mobilenet_v2, model_swin]


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


# Load models
def load_model(model_class, model_path, device):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model


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


# Ensemble inference step
def ensemble_test_step(models, dataloader, device):
    all_predictions = []
    all_labels = []
    all_image_paths = []  # To store image paths

    with torch.no_grad():
        for batch_idx, (X, y, image_paths) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Ensemble Inference"):
            X, y = X.to(device), y.to(device)

            # Initialize the ensemble predictions
            ensemble_preds = None

            # For each model, get predictions and accumulate them
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

            # Store predictions and image paths
            all_predictions.append(ensemble_preds.cpu())
            all_labels.append(y.cpu())
            all_image_paths.extend(image_paths)

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

    # output_val_predictions = "../capsule-vision-2024/reports/validation_excel.xlsx"
    output_val_predictions = "/kaggle/working/validation_excel.xlsx"
    save_predictions_to_excel(image_paths, predictions, output_val_predictions)


# Run the script
if __name__ == "__main__":
    main()