
"""
  Contains functionnalities for get, preprocess data and create pytorch Dataloaders
  for image classification.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import os
import kagglehub
import pathlib
from pathlib import Path
from shutil import copy
from typing import Tuple, List

def copy_image_from_raw_to_dest_path(raw_path: pathlib.Path, dest_parent_path: pathlib.Path, class_names: List[str]):

    """
    Copies image files to their respective class folders.

    Args:
        raw_path: The path to the source image folder.
        dest_parent_path: The path to the destination parent folder.
        class_names: A list or dictionary mapping class names.
    """

    path_list = list(raw_path.glob("*/*.png"))

    print(f"Copy is still running ...")

    for i, path in enumerate(path_list):
        parent_dir_name = path.parent.stem
        parent_name = class_names[parent_dir_name]

        destination_parent_name = dest_parent_path / parent_name

        destination_parent_name.mkdir(parents=True, exist_ok=True)

        destination_path = destination_parent_name / path.name
        copy(path, destination_path)

    print("Copy is finished!")

def get_preprocess_data():
    """
    Downloads and preprocesses the dataset, creating train, test, and validation sets.

    Downloads the CIFAR-10 dataset from Kaggle, organizes the images into
    train, test, and validation directories, and copies them to class-specific folders.
    """
    raw_path = kagglehub.dataset_download("blourdhuraju/cifar-10")
    data_path = Path("data/")
    image_path = data_path / "cifar10"

    train_dir = image_path / "train"
    test_dir = image_path / "test"
    val_dir = image_path / "val"

    if not (train_dir.is_dir() and test_dir.is_dir() and val_dir.is_dir()):
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"{train_dir}, {val_dir} and {test_dir} are already exists!")

    print(f"Data preprocessing is running ...")

    raw_path = Path(raw_path)
    class_names_file = raw_path / "cifar10" / "class_names.txt"

    with open(class_names_file, "r") as file:
        class_names = {str(i): item_class.strip() for i, item_class in enumerate(file)}

    cifar10_path = raw_path / "cifar10"
    cifar10_val_path = cifar10_path / "val"
    cifar10_train_path = cifar10_path / "train"
    cifar10_test_path = cifar10_path / "test"

    copy_image_from_raw_to_dest_path(cifar10_train_path, train_dir, class_names)
    copy_image_from_raw_to_dest_path(cifar10_test_path, test_dir, class_names)
    copy_image_from_raw_to_dest_path(cifar10_val_path, val_dir, class_names)

    print(f"Data preprocessing is finished!")


NUM_WORKERS = os.cpu_count()

def create_dataloader(train_dir: str, val_dir: str,
                      transforms: transforms.Compose,
                      batch_size: int, num_workers: int=NUM_WORKERS) -> Tuple[DataLoader, DataLoader, List[str]]:

    """
    Creates training and validation DataLoaders.

    Args:
        train_dir: The path to the training directory.
        val_dir: The path to the validation directory.
        transforms: Torchvision transforms to apply to the data.
        batch_size: The number of samples per batch in each DataLoader.
        num_workers: The number of workers for data loading.

    Returns:
        A tuple of (train_dataloader, val_dataloader, class_names).
    """

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transforms
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=transforms
    )

    class_names = train_dataset.classes

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle = True,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, class_names