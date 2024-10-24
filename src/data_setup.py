import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms


class VCEDataset(Dataset):
    """
    A custom dataset class for Visual Capsule Endoscopy (VCE) data.

    Args:
        xlsx_file (str): The Excel file containing image paths and annotations.
        root_dir (str): The root directory where the images are stored.
        train_or_test (str): Subdirectory for training or testing (used to navigate inside root_dir).
        transform (torchvision.transforms.Compose, optional): Transformations to apply to the images.
    """
    def __init__(self, xlsx_file, root_dir, train_or_test: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.xlsx_file_path = os.path.join(self.root_dir, train_or_test, xlsx_file)
        self.annotations = pd.read_excel(io=self.xlsx_file_path, sheet_name=0)
        self.class_columns = self.annotations.columns[2:]  # Assuming class labels start from the 3rd column
        self.num_classes = len(self.class_columns)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Fetches an image and its corresponding label based on the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            image (torch.Tensor): The transformed image.
            y_label (torch.Tensor): The label associated with the image, as a tensor.
        """
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0].replace("\\", "/"))
        image = Image.open(img_path).convert('RGB')  # Ensures that the image is in RGB format

        # Extract class labels (assumes the labels are in one-hot encoding in the Excel file)
        target = self.annotations.iloc[index, 2:].values
        y_label = torch.tensor(target.argmax(), dtype=torch.long)  # Converts one-hot encoding to class index

        # Apply the transformations to the image, if provided
        if self.transform:
            image = self.transform(image)

        return image, y_label

    def get_class_weights(self):
        """
        Calculates the class weights based on the distribution of class labels in the dataset.

        Returns:
            class_weights (torch.FloatTensor): Weights for each class, useful for imbalanced datasets.
        """
        class_counts = self.annotations.iloc[:, 2:].sum().values  # Sum across all samples for each class
        class_weights = 1.0 / class_counts  # Inverse of class frequency
        class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1
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
    """
    Creates DataLoader objects for training and testing datasets.

    Args:
        train_xlsx (str): The Excel file for the training dataset.
        test_xlsx (str): The Excel file for the testing dataset.
        train_root_dir (str): Subdirectory for training data.
        test_root_dir (str): Subdirectory for testing data.
        data_root_dir (str): Root directory where images are stored.
        transform (torchvision.transforms.Compose): Transformations to apply to the images.
        batch_size (int): Number of samples per batch.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.

    Returns:
        train_loader (DataLoader): DataLoader for the training dataset with weighted sampling.
        test_loader (DataLoader): DataLoader for the testing dataset (no sampling).
    """
    # Create the training dataset
    train_dataset = VCEDataset(
        xlsx_file=train_xlsx,
        root_dir=data_root_dir,
        train_or_test=train_root_dir,
        transform=transform,
    )

    # Create the testing dataset
    test_dataset = VCEDataset(
        xlsx_file=test_xlsx,
        root_dir=data_root_dir,
        train_or_test=test_root_dir,
        transform=transform,
    )

    # Get class weights to handle class imbalance
    class_weights = train_dataset.get_class_weights()

    # For each sample, get the target class index for weighting the sampler
    train_targets = [train_dataset.annotations.iloc[i, 2:].values.argmax() for i in range(len(train_dataset))]
    sample_weights = [class_weights[t] for t in train_targets]  # Assign class weight to each sample

    # Create a weighted random sampler for the training set
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

    # Create DataLoaders for training and testing sets
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Uses the weighted sampler to balance classes
        num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers
    )

    return train_loader, test_loader