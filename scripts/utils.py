import os
import gc
import time
import math
import numpy as np
import pandas as pd
import random
from pathlib import Path
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

import torch
from torch import nn


def save_model(model: nn.Module, target_dir: str, model_name: str):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    
    model_save_path = target_dir_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def is_torch_available():
    return torch is not None


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_bytes = total_params * 4
    size_in_mb = size_in_bytes / (1024 ** 2)
    return size_in_mb

def save_predictions_to_excel(image_paths, y_pred: torch.Tensor, output_path: str):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    y_pred_classes = y_pred.argmax(dim=1).cpu().numpy()
    df = pd.DataFrame({
        'image_path': image_paths,
        'predicted_class': [class_columns[i] for i in y_pred_classes],
        **{col: y_pred[:, i].cpu().numpy() for i, col in enumerate(class_columns)}
    })
    df.to_excel(output_path, index=False)
