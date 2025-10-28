import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from lr_finder import LRFinder, train_with_one_cycle, plot_training_history


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ResNet-50 training with One-Cycle Policy"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_dir", type=str, default="./data", help="Path to the dataset directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "CIFAR100", "ImageNet", "custom"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for training (default: 224)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of classes in the dataset (default: 10)",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer (default: 0.9)",
    )

    # Data augmentation parameters
    parser.add_argument(
        "--cutout",
        action="store_true",
        help="Use Cutout data augmentation (for ImageNet and custom datasets)",
    )
    parser.add_argument(
        "--cutout_holes",
        type=int,
        default=1,
        help="Number of holes to cut out from image (default: 1)",
    )
    parser.add_argument(
        "--cutout_length",
        type=int,
        default=16,
        help="Length of the holes (default: 16)",
    )
    parser.add_argument(
        "--rotate_degrees",
        type=float,
        default=7.0,
        help="Max rotation angle in degrees for augmentation (default: 10.0)",
    )
    parser.add_argument(
        "--use_albumentation",
        action="store_true",
        help="Use Albumentation library for augmentations",
    )

    # One-cycle policy parameters
    parser.add_argument(
        "--find_lr",
        action="store_true",
        help="Run learning rate finder before training",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.01,
        help="Maximum learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--div_factor",
        type=float,
        default=25.0,
        help="Initial learning rate is max_lr/div_factor",
    )
    parser.add_argument(
        "--pct_start",
        type=float,
        default=0.3,
        help="Percentage of training spent increasing the learning rate (default: 0.3)",
    )

    # Model parameters
    parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint to resume training from",
    )

    # Hardware parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (default: cuda if available, else cpu)",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )

    return parser.parse_args()


def get_transforms(args):
    """
    Get data transformations based on the selected dataset.
    """
    # Mean and std for ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if args.use_albumentation:
        # Using Albumentations for more efficient augmentations
        if args.dataset in ["CIFAR10", "CIFAR100"]:
            # Transforms for CIFAR datasets
            train_transform = AlbumentationTransform(
                A.Compose(
                    [
                        A.RandomResizedCrop(height=args.img_size, width=args.img_size),
                        A.HorizontalFlip(),
                        A.Normalize(mean=mean, std=std),
                        ToTensorV2(),
                    ]
                )
            )

            val_transform = AlbumentationTransform(
                A.Compose(
                    [
                        A.Resize(height=args.img_size + 32, width=args.img_size + 32),
                        A.CenterCrop(height=args.img_size, width=args.img_size),
                        A.Normalize(mean=mean, std=std),
                        ToTensorV2(),
                    ]
                )
            )
        else:
            # ImageNet-style transforms with enhanced augmentation
            train_transform_list = [
                A.RandomResizedCrop(height=args.img_size, width=args.img_size),
                A.HorizontalFlip(),
                A.Rotate(limit=args.rotate_degrees, p=0.7),
                A.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8
                ),
                A.Normalize(mean=mean, std=std),
            ]

            # Add Cutout if requested
            if args.cutout:
                train_transform_list.append(
                    A.CoarseDropout(
                        max_holes=args.cutout_holes,
                        max_height=args.cutout_length,
                        max_width=args.cutout_length,
                        min_height=args.cutout_length // 2,
                        min_width=args.cutout_length // 2,
                        fill_value=128,
                        p=0.5,
                    )
                )

            train_transform_list.append(ToTensorV2())

            train_transform = AlbumentationTransform(A.Compose(train_transform_list))

            val_transform = AlbumentationTransform(
                A.Compose(
                    [
                        A.Resize(
                            height=int(args.img_size * 1.14),
                            width=int(args.img_size * 1.14),
                        ),
                        A.CenterCrop(height=args.img_size, width=args.img_size),
                        A.Normalize(mean=mean, std=std),
                        ToTensorV2(),
                    ]
                )
            )
    else:
        # Standard PyTorch transforms
        if args.dataset in ["CIFAR10", "CIFAR100"]:
            # Standard transforms for CIFAR datasets
            train_transforms_list = [
                transforms.RandomResizedCrop(args.img_size),
                transforms.RandomHorizontalFlip(),
            ]

            if args.cutout:
                train_transforms_list.append(
                    Cutout(n_holes=args.cutout_holes, length=args.cutout_length)
                )

            train_transforms_list.extend(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            )

            train_transform = transforms.Compose(train_transforms_list)

            val_transform = transforms.Compose(
                [
                    transforms.Resize(args.img_size + 32),
                    transforms.CenterCrop(args.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            # ImageNet-style transforms for larger images with enhanced augmentation
            train_transforms_list = [
                transforms.RandomResizedCrop(args.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(args.rotate_degrees, fill=(128, 128, 128)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            ]

            if args.cutout:
                train_transforms_list.append(
                    Cutout(n_holes=args.cutout_holes, length=args.cutout_length)
                )

            train_transforms_list.extend(
                [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
            )

            train_transform = transforms.Compose(train_transforms_list)

            val_transform = transforms.Compose(
                [
                    transforms.Resize(int(args.img_size * 1.14)),
                    transforms.CenterCrop(args.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

    return train_transform, val_transform


class Cutout:
    """
    Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = torch.ones((h, w), dtype=torch.float32)

        for _ in range(self.n_holes):
            # Random position of the cutout patch
            y = np.random.randint(h)
            x = np.random.randint(w)

            # Ensure the patch is within bounds
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            # Create the cutout
            mask[y1:y2, x1:x2] = 0.0

        mask = mask.unsqueeze(0).repeat(img.size(0), 1, 1)
        img = img * mask

        return img


class AlbumentationTransform:
    """
    Wrapper class for Albumentation transforms to be compatible with PyTorch datasets.
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Apply albumentation transform
        result = self.transform(image=img)
        return result["image"]


# This section has been replaced with the Cutout implementation above


def get_datasets(args):
    """
    Get training and validation datasets based on the selected dataset.
    """
    train_transform, val_transform = get_transforms(args)

    if args.dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=val_transform
        )
    elif args.dataset == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=val_transform
        )
    elif args.dataset == "ImageNet":
        train_dir = os.path.join(args.data_dir, "train")
        val_dir = os.path.join(args.data_dir, "val")

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise ValueError(
                f"ImageNet data directory structure not found in {args.data_dir}"
            )

        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    elif args.dataset == "custom":
        # For custom datasets, assume ImageFolder structure
        train_dir = os.path.join(args.data_dir, "train")
        val_dir = os.path.join(args.data_dir, "val")

        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise ValueError(
                f"Custom dataset directory structure not found in {args.data_dir}"
            )

        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return train_dataset, val_dataset


def get_model(args):
    """
    Get ResNet-50 model with proper final layer for the given number of classes.
    """
    if args.pretrained:
        print("=> using pre-trained ResNet-50 model")
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        print("=> creating ResNet-50 model from scratch")
        model = models.resnet50(weights=None)

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, args.num_classes)

    return model


def find_learning_rate(model, train_loader, args):
    """
    Run learning rate finder to determine the optimal max learning rate.
    """
    # Define criterion
    criterion = nn.CrossEntropyLoss()

    # Create optimizer for the finder with a placeholder learning rate
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,  # This will be overridden by the finder
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Create and run the learning rate finder
    print("Running learning rate finder...")
    lr_finder = LRFinder(model, optimizer, criterion, args.device)
    lrs, losses = lr_finder.range_test(
        train_loader, start_lr=1e-7, end_lr=1.0, num_iter=100, step_mode="exp"
    )

    # Plot the results and get suggested learning rate
    min_grad_idx = lr_finder.plot()
    if min_grad_idx is not None and min_grad_idx < len(lrs):
        suggested_lr = lrs[min_grad_idx]
        print(f"Suggested maximum learning rate: {suggested_lr:.6f}")
        return suggested_lr
    else:
        print(
            f"Could not automatically determine optimal learning rate, using default: {args.max_lr}"
        )
        return args.max_lr


def main():
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Get datasets
    print(f"Loading {args.dataset} dataset...")
    train_dataset, val_dataset = get_datasets(args)

    # Log the augmentation settings
    if args.cutout:
        print(
            f"Using Cutout augmentation with {args.cutout_holes} holes of size {args.cutout_length}"
        )

    if args.use_albumentation:
        print("Using Albumentations library for advanced image augmentations")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(args.device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(args.device == "cuda"),
    )

    print(
        f"Dataset loaded with {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
    )

    # Create model
    model = get_model(args)
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Find learning rate if requested
    if args.find_lr:
        max_lr = find_learning_rate(model, train_loader, args)
    else:
        max_lr = args.max_lr
        print(f"Using specified maximum learning rate: {max_lr}")

    # Create optimizer with initial learning rate = max_lr / div_factor
    initial_lr = max_lr / args.div_factor
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    print(f"Initial learning rate: {initial_lr:.6f}")
    print(f"Maximum learning rate: {max_lr:.6f}")

    # Train with one-cycle policy
    print("\nStarting training with one-cycle policy...")

    history = train_with_one_cycle(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        max_lr=max_lr,
        num_epochs=args.epochs,
        device=device,
    )

    # Save the final model
    final_model_path = os.path.join(args.output_dir, "resnet50_final.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        final_model_path,
    )
    print(f"Final model saved to {final_model_path}")

    # Plot the training history
    plot_training_history(history)

    # Save the plot
    plt.savefig(os.path.join(args.output_dir, "training_history.png"))

    print("Training complete!")


if __name__ == "__main__":
    main()
