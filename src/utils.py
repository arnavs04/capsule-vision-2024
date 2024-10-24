import os
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict
from logging import getLogger, Logger, INFO, StreamHandler, FileHandler, Formatter

import numpy as np
import pandas as pd
import torch
from torch import nn

# Flag to check if running in a Kaggle environment
KAGGLE = True

# Define directories for logging and saving reports
logging_dir = "../capsule-vision-2024/logs"
save_report_dir = "../capsule-vision-2024/reports"

# Modify paths for Kaggle environment
if KAGGLE is True:
    logging_dir = "kaggle/working/logs"
    save_report_dir = "kaggle/working/reports"


def setup_logger(model_name: str) -> Logger:
    """
    Set up a logger for tracking model performance and events.
    """
    log_dir = os.path.join(logging_dir, model_name)
    os.makedirs(log_dir, exist_ok=True)  # Ensure the log directory exists
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = getLogger(model_name)
    
    # Only set up the logger if it hasn't been set up before
    if not logger.handlers:
        logger.setLevel(INFO)

        file_handler = FileHandler(log_file)  # Save logs to file
        stream_handler = StreamHandler()  # Output logs to console

        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def save_model(model: nn.Module, target_dir: str, model_name: str):
    """
    Save the model's state dictionary to a given directory.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)  # Save the model's state dictionary


def load_model(model: nn.Module, target_dir: str, model_name: str):
    """
    Load the model's state dictionary from a given directory.
    """
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    target_dir_path = Path(target_dir)
    
    model_load_path = target_dir_path / model_name
    assert model_load_path.is_file(), f"Model file not found at: {model_load_path}"
    
    print(f"[INFO] Loading model from: {model_load_path}")
    model.load_state_dict(torch.load(model_load_path))  # Load the model's state dictionary
    return model


def save_metrics_report(report: Dict, model_name: str, epoch: int, save_dir: str = save_report_dir):
    """
    Save the metrics report as a JSON file for each epoch.
    """
    report_dir = os.path.join(save_dir, model_name)
    os.makedirs(report_dir, exist_ok=True)  # Create directory if it doesn't exist

    report_filename = f"metrics_epoch_{epoch+1}.json"  # Name the report file based on the epoch
    report_path = os.path.join(report_dir, report_filename)
    
    # Save the report as a JSON file
    with open(report_path, 'w') as report_file:
        json.dump(report, report_file, indent=4)
    
    print(f"[INFO] Saved metrics report for {model_name}, epoch {epoch+1} at {report_path}")


def save_predictions_to_excel(image_paths, y_pred: torch.Tensor, output_path: str):
    """
    Save model predictions to an Excel file.
    """
    # Define class names corresponding to the model's output
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 
                     'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    
    # Convert predicted logits to class indices
    y_pred_classes = y_pred.argmax(dim=1).cpu().numpy()
    
    # Create a DataFrame with image paths, predicted classes, and prediction probabilities for each class
    df = pd.DataFrame({
        'image_path': image_paths,
        'predicted_class': [class_columns[i] for i in y_pred_classes],
        **{col: y_pred[:, i].cpu().numpy() for i, col in enumerate(class_columns)}
    })
    
    # Save the predictions to an Excel file
    df.to_excel(output_path, index=False)


def seed_everything(seed=42):
    """
    Set a seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Set the PYTHONHASHSEED environment variable
    np.random.seed(seed)  # Set the seed for NumPy
    torch.manual_seed(seed)  # Set the seed for PyTorch
    torch.cuda.manual_seed(seed)  # Set the seed for CUDA (if using GPU)
    torch.backends.cudnn.deterministic = True  # Make CuDNN deterministic