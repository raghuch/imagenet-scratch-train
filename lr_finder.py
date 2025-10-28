import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm


class LRFinder:
    """
    Learning rate finder for PyTorch models, implementing the one-cycle policy approach
    described in Leslie Smith's paper "Cyclical Learning Rates for Training Neural Networks".

    This implementation helps find the optimal learning rate for training deep neural networks
    by gradually increasing the learning rate during training and observing the loss.
    """

    def __init__(self, model, optimizer, criterion, device):
        """Initialize the learning rate finder.
        Parameters:
        model : torch.nn.Module -- The PyTorch model to train
        optimizer : torch.optim.Optimizer --The optimizer to use for training
        criterion : callable -- Loss function
        device : torch.device --Device to use for training (cpu or cuda)"""
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Training history
        self.history = {"lr": [], "loss": []}

        # Best loss tracking
        self.best_loss = float("inf")

    def range_test(
        self,
        train_loader,
        start_lr=1e-7,
        end_lr=10,
        num_iter=100,
        step_mode="exp",
        smooth_f=0.05,
        diverge_th=5.0,
    ):
        """Perform the learning rate range test.
        Parameters:
        train_loader : DataLoader -- PyTorch DataLoader for training data
        start_lr : float, default=1e-7 -- Starting learning rate
        end_lr : float, default=10 --Maximum learning rate
        num_iter : int, default=100 --Number of iterations to run the test
        step_mode : str, default="exp" --"exp" for exponential increase, "linear" for linear increase
        smooth_f : float, default=0.05 --Smoothing factor for loss values
        diverge_th : float, default=5.0 --Threshold for detecting divergence (loss > diverge_th * best_loss)

        Returns:
        tuple (learning_rates, losses)"""
        # Reset model and optimizer state
        self.model.train()
        self.history = {"lr": [], "loss": []}
        self.best_loss = float("inf")

        # Set the starting learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr

        # Calculate the learning rate multiplier per iteration
        if step_mode == "exp":
            lr_schedule = lambda i: start_lr * (end_lr / start_lr) ** (i / num_iter)
        elif step_mode == "linear":
            lr_schedule = lambda i: start_lr + (end_lr - start_lr) * (i / num_iter)
        else:
            raise ValueError(f"step_mode must be 'exp' or 'linear', got {step_mode}")

        # Initialize iterator and progress bar
        iterator = iter(train_loader)
        pbar = tqdm(range(num_iter), desc="LR Finder", dynamic_ncols=True)

        # Run the learning rate search
        for iteration in pbar:
            # Get the next batch of data
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)

            # Move data to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update the learning rate
            lr = lr_schedule(iteration)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Record the loss and learning rate
            loss_value = loss.item()

            # Apply smoothing to the loss
            if iteration > 0:
                loss_value = (
                    smooth_f * loss_value + (1 - smooth_f) * self.history["loss"][-1]
                )

            self.history["lr"].append(lr)
            self.history["loss"].append(loss_value)

            # Update progress bar
            pbar.set_postfix(loss=loss_value, lr=lr)

            # Check if the loss is getting too high
            if loss_value < self.best_loss:
                self.best_loss = loss_value

            if loss_value > diverge_th * self.best_loss:
                print(
                    f"\nStopping early: loss {loss_value:.4f} > {diverge_th} * best_loss {self.best_loss:.4f}"
                )
                break

        return self.history["lr"], self.history["loss"]

    def plot(
        self, skip_start=10, skip_end=5, log_lr=True, title="Learning Rate Finder"
    ):
        """
        Plot the learning rate vs. loss.

        Parameters:
        -----------
        skip_start : int, default=10
            Number of batches to skip at the start
        skip_end : int, default=5
            Number of batches to skip at the end
        log_lr : bool, default=True
            Whether to use a logarithmic x-axis for learning rates
        title : str, default="Learning Rate Finder"
            Title for the plot
        """
        lrs = self.history["lr"]
        losses = self.history["loss"]

        # Skip the specified number of batches at the start and end
        if skip_start > 0:
            lrs = lrs[skip_start:]
            losses = losses[skip_start:]
        if skip_end > 0:
            lrs = lrs[:-skip_end] if skip_end < len(lrs) else lrs
            losses = losses[:-skip_end] if skip_end < len(losses) else losses

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title(title)

        if log_lr:
            plt.xscale("log")

        plt.grid(True, which="both", ls="-")

        # Find the point of steepest descent for the loss
        min_grad_idx = None
        try:
            # Calculate the gradients
            gradients = np.gradient(losses)
            # Find the index with the steepest negative gradient
            smooth_grads = np.convolve(gradients, np.ones(5) / 5, mode="valid")
            min_grad_idx = np.argmin(smooth_grads) + 2  # +2 due to convolution
            min_grad_idx = min(min_grad_idx, len(lrs) - 1)  # Ensure valid index

            # Mark the suggested learning rate on the plot
            suggested_lr = lrs[min_grad_idx]
            plt.axvline(x=suggested_lr, color="r", linestyle="--")
            plt.text(
                suggested_lr,
                plt.ylim()[0],
                f" {suggested_lr:.2E}",
                horizontalalignment="left",
                verticalalignment="bottom",
            )
            print(f"Suggested learning rate: {suggested_lr:.2E}")
        except:
            print("Could not determine the suggested learning rate automatically.")

        plt.show()

        return min_grad_idx


def create_one_cycle_policy(
    optimizer,
    max_lr,
    total_steps,
    pct_start=0.3,
    div_factor=25.0,
    final_div_factor=1e4,
    anneal_strategy="cos",
):
    """
    Creates a one-cycle learning rate scheduler.

    Parameters:
    -----------
    optimizer : torch.optim.Optimizer
        The optimizer to use
    max_lr : float or list
        Maximum learning rate(s)
    total_steps : int
        Total number of training steps
    pct_start : float, default=0.3
        Percentage of training spent increasing the learning rate
    div_factor : float, default=25.0
        Initial learning rate is max_lr/div_factor
    final_div_factor : float, default=1e4
        Final learning rate is max_lr/final_div_factor
    anneal_strategy : str, default='cos'
        Annealing strategy: 'cos' for cosine, 'linear' for linear

    Returns:
    --------
    torch.optim.lr_scheduler
        One-cycle learning rate scheduler
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        anneal_strategy=anneal_strategy,
    )


def train_with_one_cycle(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    max_lr,
    num_epochs,
    device="cuda",
):
    """
    Train a model using the one-cycle policy.

    Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    optimizer : torch.optim.Optimizer
        The optimizer to use
    criterion : callable
        Loss function
    max_lr : float
        Maximum learning rate (found using the LR finder)
    num_epochs : int
        Number of epochs to train for
    device : str, default='cuda'
        Device to use for training

    Returns:
    --------
    dict
        Training history
    """
    # Calculate total steps for the scheduler
    total_steps = len(train_loader) * num_epochs

    # Create the one-cycle scheduler
    scheduler = create_one_cycle_policy(
        optimizer=optimizer, max_lr=max_lr, total_steps=total_steps
    )

    # Initialize history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    # Move model to device
    model = model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
        ):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update learning rate
            scheduler.step()

            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100.0 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"
            ):
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Update metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Update history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(current_lr)

        # Print progress
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%, "
            f"LR: {current_lr:.6f}"
        )

    return history


def plot_training_history(history):
    """
    Plot the training history.

    Parameters:
    -----------
    history : dict
        Training history from train_with_one_cycle
    """
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # Plot loss
    axs[0, 0].plot(history["train_loss"], label="Train Loss")
    axs[0, 0].plot(history["val_loss"], label="Validation Loss")
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot accuracy
    axs[0, 1].plot(history["train_acc"], label="Train Accuracy")
    axs[0, 1].plot(history["val_acc"], label="Validation Accuracy")
    axs[0, 1].set_title("Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy (%)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot learning rate
    axs[1, 0].plot(history["lr"])
    axs[1, 0].set_title("Learning Rate")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Learning Rate")
    axs[1, 0].grid(True)

    # Make the 4th subplot empty or use it for another metric
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations for ImageNet-style datasets
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # You would load your actual dataset here
    # For example, with ImageNet or a similar dataset:
    # train_dataset = datasets.ImageFolder(root='path/to/train', transform=train_transform)
    # val_dataset = datasets.ImageFolder(root='path/to/val', transform=val_transform)

    # For demonstration, we'll use CIFAR-10 which is much smaller
    print("Loading CIFAR-10 dataset (for demonstration)...")
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=val_transform
    )

    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Create ResNet-50 model
    print("Creating ResNet-50 model...")
    model = models.resnet50(weights=None)  # Start with random weights

    # Adjust the final layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Create optimizer (without setting learning rate yet)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Create LR finder and run the range test
    print("\nRunning learning rate finder...")
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lrs, losses = lr_finder.range_test(
        train_loader, start_lr=1e-7, end_lr=1.0, num_iter=100, step_mode="exp"
    )

    # Plot the results and get suggested learning rate
    min_grad_idx = lr_finder.plot()
    if min_grad_idx is not None:
        suggested_lr = lrs[min_grad_idx]
    else:
        # Fallback to a reasonable default
        suggested_lr = 0.01

    print(f"\nTraining with one-cycle policy using max_lr = {suggested_lr:.4f}...")

    # Reset the model and optimizer for actual training
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # For CIFAR-10
    optimizer = optim.SGD(
        model.parameters(), lr=suggested_lr / 25, momentum=0.9, weight_decay=5e-4
    )

    # Train with one-cycle policy
    history = train_with_one_cycle(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        max_lr=suggested_lr,
        num_epochs=10,
        device=device,
    )

    # Plot the training history
    plot_training_history(history)

    print("Training complete!")
