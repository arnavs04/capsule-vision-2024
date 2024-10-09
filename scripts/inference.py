import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os
import json
from metrics import generate_metrics_report  # Assuming this is your metrics reporting function

from model_builder import (
    model_resnet18,
    model_beit,
    model_inception_resnet_v2,
    model_mobilenet_v2,
    model_swin,
)

## INCOMPLETE


def load_model(model_class, model_path, device):
    """Load the trained model from a specified path."""
    model = model_class()  # Replace with your model initialization
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model


def preprocess_data(data_dir):
    """Preprocess the VCE dataset for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size according to your model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Adjust based on your dataset
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # Adjust batch size as necessary
    return dataloader, dataset.class_to_idx


def run_inference(model, dataloader, device):
    """Run inference on the dataset and collect predictions and true labels."""
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Running Inference"):
            X = X.to(device)
            y_pred_logits = model(X)  # Get model logits

            # Apply softmax to get probabilities
            y_pred_probs = torch.softmax(y_pred_logits, dim=1)

            # Collect predictions and true labels
            all_predictions.append(y_pred_probs)
            all_labels.append(y)

    # Concatenate results
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return labels, predictions


def main(data_dir, model_path):
    """Main function to run inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(model_class=model, model_path=model_path, device=device)

    # Preprocess the data
    dataloader, class_names = preprocess_data(data_dir)

    # Run inference
    true_labels, predictions = run_inference(model, dataloader, device)

    # Generate metrics report
    metrics_report = generate_metrics_report(true_labels, predictions)
    print("Metrics Report:\n", metrics_report)

    # Save the predictions to a file
    predictions_numpy = predictions.cpu().numpy()  # Move to CPU and convert to NumPy
    predictions_dict = {
        "predictions": predictions_numpy.tolist(),  # Convert to list for JSON serialization
        "true_labels": true_labels.cpu().numpy().tolist()  # True labels
    }
    
    with open('predictions.json', 'w') as f:
        json.dump(predictions_dict, f)

    print("Inference completed. Predictions saved to predictions.json.")

if __name__ == "__main__":
    DATA_DIR = 'path/to/your/vce/dataset'  # Replace with your VCE dataset path
    MODEL_PATH = 'path/to/your/model.pth'  # Replace with your model path
    main(DATA_DIR, MODEL_PATH)