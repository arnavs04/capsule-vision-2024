import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

NUM_WORKERS =  0 # os.cpu_count()


class VCEDataset(Dataset):
    def __init__(self, xlsx_file, root_dir, train_or_test: str, transform=None):
        # Load annotations from the XLSX file
        self.root_dir = root_dir
        self.transform = transform
        self.xlsx_file_path = os.path.join(self.root_dir, train_or_test, xlsx_file)
        self.annotations = pd.read_excel(io=self.xlsx_file_path, sheet_name=0)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        target = self.annotations.iloc[index, 2:].values
        y_label = torch.tensor(target.argmax(), dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
    

def create_dataloaders(
    train_xlsx: str,
    test_xlsx: str,
    train_root_dir: str,
    test_root_dir: str,
    data_root_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
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

    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader
