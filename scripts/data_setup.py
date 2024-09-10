import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS =  0 #os.cpu_count()

class VCEDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # Load annotations from the CSV file
        self.annotations = pd.read_xslx(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get the image path and load the image
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        # Read the target columns (assumed to be from the third column onward)
        target = self.annotations.iloc[index, 2:].values
        # Convert one-hot encoded target to a single class index
        y_label = torch.tensor(target.argmax(), dtype=torch.long)

        # Apply any provided transformations
        if self.transform:
            image = self.transform(image)

        return (image, y_label)
    

def create_dataloaders(
    train_csv: str,
    test_csv: str,
    root_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    # Create datasets
    train_dataset = VCEDataset(
        csv_file=train_csv,
        root_dir=root_dir,
        transform=transform,
    )

    test_dataset = VCEDataset(
        csv_file=test_csv,
        root_dir=root_dir,
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
