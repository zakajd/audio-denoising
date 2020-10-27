"""
Init datasets and dataloaders
"""
import os
import cv2
import math
import random
import pathlib
from loguru import logger
import numpy as np
import torch
import torchvision
import albumentations as albu
import albumentations.pytorch as albu_pt

from src.utils import ToCudaLoader

# Default ImageNet mean and std
# MEAN = (0.485, 0.456, 0.406)
# STD = (0.229, 0.224, 0.225)

# No normalization
MEAN = (0., 0., 0.)
STD = (1.0, 1.0, 1.0)


def get_dataloaders(
    root: str = "data/raw",
    aug_type: str = "light",
    task: str = "classification",
    batch_size: int = 64,
    size: int = 128,
    workers: int = 6,
):
    """
    Args:
        root: Path to folder with data
        aumentation: Type of aug defined in `src.data.augmentations.py`
        task: One of {`classification`, `denoising`}
        batch_size: Number of images in stack
        size: Size of images used for training
        workers: Number of CPU threads used to load images
    Returns:
        train_dataloader
        val_dataloader
    """
    root = pathlib.Path(root)
    print(root)
    # root = pathlib.Path(os.path.join(root, 'train' if train else 'val'))

    # Get augmentations
    train_aug = get_aug(aug_type, size=size)
    logger.info(f"Using {aug_type} augs: {train_aug}")

    # Get dataset
    dataset_class = {
        'classification':  ClassificationDataset,
        'denoising': DenoisingDataset,
    }[task]

    train_dataset = dataset_class(
        root=root / 'train', transform=train_aug, train=True)
    logger.info(f"Train size ({task}): {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=workers,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

    train_loader = ToCudaLoader(train_loader)
    val_loader, _ = get_val_dataloader(
        root / 'val',
        aug_type="val",
        task=task,
        batch_size=batch_size,
        size=size,
        workers=workers,
    )

    return train_loader, val_loader


def get_val_dataloader(
        root="data/raw/val", aug_type="val", task='classification', batch_size=64, size=128, workers=6):
    """
    Returns:
        val_loader (DataLoader)
        val_targets
    """
    val_aug = get_aug(aug_type, size=size)

    # Get dataset
    dataset_class = {
        'classification':  ClassificationDataset,
        'denoising': DenoisingDataset,
    }[task]

    val_dataset = dataset_class(
        root=root, transform=val_aug, train=False)

    logger.info(f"Val size {task}: {len(val_dataset)}")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        shuffle=False,
    )

    val_loader = ToCudaLoader(val_loader)
    return val_loader, val_dataset.targets


def get_aug(aug_type: str = "val", size: int = 128):
    """Return augmentations by type
    Args:
        aug_type : one of `val`, `test`, `light`, `medium`
        size: final size of the crop
    """
    N_FFT = 96 # Image height
    # N_FFT = 80 # Image height

    NORM_TO_TENSOR = albu.Compose([
        # albu.Normalize(mean=MEAN, std=STD),  # No normalization for now
        albu_pt.ToTensorV2()])

    CROP_AUG = albu.Compose([
        albu.PadIfNeeded(N_FFT, size, border_mode=0),
        albu.RandomCrop(N_FFT, size),
    ])

    LIGHT_AUG = albu.Compose(
        [   
            CROP_AUG,
            albu.Cutout(num_holes=8, max_h_size=size // 16, max_w_size=size // 16, fill_value=0, p=0.3),
            # Add noise
            albu.GaussNoise(var_limit=(0.1, 0.3)),
            NORM_TO_TENSOR
        ],
        p=1.0,
    )

    VAL_AUG = albu.Compose([
        CROP_AUG,
        NORM_TO_TENSOR
    ])

    types = {
        "light": LIGHT_AUG,
        "val": VAL_AUG,
        "test": VAL_AUG,
    }

    return types[aug_type]


class ClassificationDataset(torch.utils.data.Dataset):
    """Dataset class for training and validation data

    Args:
        root: Path to directory with melspectograms splitted into clean / noisy foldes
        transform: albumentations.Transform object
        train: Flag to switch between training and validation
    
    Returns:
        image: Mel-spectogram
        target: Binary label. 0 - clean, 1 - noisy
    """
    def __init__(self, root: str = "data/raw/train", transform=None, train: bool = True):
        root = pathlib.Path(root)

        self.clean_files = sorted((root / "clean").glob("*/*.npy"))
        self.noisy_files = sorted((root / "noisy").glob("*/*.npy"))
        assert len(self.clean_files) == len(self.noisy_files), "Clean and noisy files doesn't match!"

        self.files = self.clean_files + self.noisy_files
        self.targets = [0] * len(self.clean_files) + [0] * len(self.noisy_files)

        # Сheck that all images exist
        assert map(lambda x: pathlib.Path(x).exists(), self.files), "Found missing images!"

        self.transform = albu.Compose([albu_pt.ToTensorV2()]) if transform is None else transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # Read image and convert to full precision
        image = np.load(self.files[index]).astype('float32')
        # One channel -> Fake RGB
        image = np.repeat(image[np.newaxis, :, :], 3, axis=0)

        # Transform
        image = self.transform(image=image.T)["image"]
        target = torch.tensor(self.targets[index], dtype=image.dtype)
        return image, target

    def __len__(self):
        return len(self.files)


class DenoisingDataset(ClassificationDataset):
    """Denoising dataset class for training and validation data

    Args:
        root: Path to directory with melspectograms
        transform: albumentations.Transform object
        train: Flag to switch between training and validation
        to_rgb: Flag to make fake channels to use pretrained models
    
    Returns:
        noisy_image: Mel-spectogram with noise
        clean_image: Mel-spectogram without noise
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # Read images and convert to full precision
        clean_image = np.load(self.clean_files[index]).astype('float32')
        noisy_image = np.load(self.noisy_files[index]).astype('float32')

        # One channel -> Fake RGB
        noisy_image = np.repeat(noisy_image[np.newaxis, :, :], 3, axis=0)

        # Transform
        transform = self.transform(image=noisy_image.T, mask=clean_image.T)
        clean_image = transform["mask"].unsqueeze(0)  # Add channel dim
        noisy_image = transform["image"]

        # print("Noisy", index, noisy_image.shape)
        # print("Clean", index, clean_image.shape)
        return noisy_image, clean_image

    def __len__(self):
        return len(self.clean_files)
