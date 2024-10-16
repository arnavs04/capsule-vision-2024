import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms


class VCEDataset(Dataset):
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
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0].replace("\\", "/"))
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format

        target = self.annotations.iloc[index, 2:].values
        y_label = torch.tensor(target.argmax(), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, y_label

    def get_class_weights(self):
        class_counts = self.annotations.iloc[:, 2:].sum().values
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum()  # Normalize
        return torch.FloatTensor(class_weights)

def create_dataloaders(
    train_xlsx: str,
    test_xlsx: str,
    train_root_dir: str,
    test_root_dir: str,
    data_root_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = 4
):
    # Create datasets
    train_dataset = VCEDataset(
        xlsx_file=train_xlsx,
        root_dir=data_root_dir,
        train_or_test=train_root_dir,
        transform=transform,
    )

    test_dataset = VCEDataset(
        xlsx_file=test_xlsx,
        root_dir=data_root_dir,
        train_or_test=test_root_dir,
        transform=transform,
    )

    # Calculate sample weights for training set
    class_weights = train_dataset.get_class_weights()
    train_targets = [train_dataset.annotations.iloc[i, 2:].values.argmax() for i in range(len(train_dataset))]
    sample_weights = [class_weights[t] for t in train_targets]

    # Create weighted sampler for training set
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader