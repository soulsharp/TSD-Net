import os

import torch
import torch.utils.data as data
import torchvision
from PIL import Image, ImageTransform
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def build_transforms(cfg, is_train=True):
    """
    Builds a torchvision transform pipeline based on configuration settings.

    Args:
        cfg (dict): Configuration dictionary containing the following keys:
            - "mean" (list of float): Per-channel means for normalization.
            - "std" (list of float): Per-channel standard deviations for normalization.
            - "im_size" (int): Final image size after cropping/resizing.
            - "interpolation_mode" (str): Interpolation method name for resizing.
              Must be one of the keys in torchvision.transforms.InterpolationMode (e.g., "BICUBIC", "BILINEAR").
            - Optional (for training only):
                - "cj_brightness" (float): Brightness jitter strength for ColorJitter. Default: 0.4.
                - "cj_contrast" (float): Contrast jitter strength for ColorJitter. Default: 0.4.
                - "cj_saturation" (float): Saturation jitter strength for ColorJitter. Default: 0.4.
                - "cj_hue" (float): Hue jitter strength for ColorJitter. Default: 0.1.

        is_train (bool, optional): Whether to build transforms for training or evaluation. Default is True.

    Returns:
        torchvision.transforms.Compose: Composed transform pipeline.
            - Training: Resize → RandomCrop → RandomHorizontalFlip → ColorJitter → ToTensor → Normalize
            - Evaluation: Resize → ToTensor → Normalize
    """

    normalize = T.Normalize(mean=cfg["mean"], std=cfg["std"])
    transforms = None

    # if is_train:
    #     crop = cfg["im_size"]
    #     precrop = crop + 32
    #     transforms = T.Compose([
    #         T.Resize(
    #             (precrop, precrop),
    #             interpolation=InterpolationMode[cfg["interpolation_mode"]]
    #         ),
    #         T.RandomCrop((crop, crop)),
    #         T.RandomHorizontalFlip(),
    #         T.ColorJitter(
    #             cfg.get("cj_brightness", 0.4),
    #             cfg.get("cj_contrast", 0.4),
    #             cfg.get("cj_saturation", 0.4),
    #             cfg.get("cj_hue", 0.1),
    #         ),
    #         T.ToTensor(),
    #         normalize,
    #     ])

    # else:
    #     transforms = T.Compose([
    #         T.Resize(
    #             (cfg["im_size"], cfg["im_size"]),
    #             interpolation=InterpolationMode[cfg["interpolation_mode"]]
    #         ),
    #         T.ToTensor(),
    #         normalize
    #         ])

    # Testing different augmentations
    if is_train:
        transforms = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ]
        )
    else:
        transforms = T.Compose([T.ToTensor(), normalize])

    return transforms


def prepare_cifar10_dataset(cfg):
    """
    Prepares the CIFAR-10 training and test datasets with configurable transforms.

    Args:
        cfg (dict): Configuration dictionary.
            Required keys include:
                - "mean": List of 3 floats for channel-wise mean normalization.
                - "std": List of 3 floats for channel-wise std normalization.
                - "im_size": Integer target image size.
                - "interpolation_mode": String name of the interpolation mode (e.g., "BICUBIC").
                - (Optional, for training):
                    - "cj_brightness", "cj_contrast", "cj_saturation", "cj_hue"

    Returns:
        tuple:
            - dataset_train (torchvision.datasets.CIFAR10): Training dataset with augmentation.
            - dataset_test (torchvision.datasets.CIFAR10): Test dataset with deterministic transforms.
    """

    transform_train = build_transforms(cfg, is_train=True)
    transform_test = build_transforms(cfg, is_train=False)

    dataset_train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    dataset_test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    return dataset_train, dataset_test
