# ResNet-50 Training with One-Cycle Learning Rate Policy

This module provides implementation of the one-cycle learning rate policy for training neural networks, specifically optimized for ResNet-50 architectures. The one-cycle policy, introduced by Leslie Smith, is an effective method for fast and stable neural network training. The implementation includes advanced data augmentation techniques like Cutout, random rotations, and enhanced image transforms via the Albumentations library to improve model generalization.

## What is One-Cycle Policy?

The one-cycle policy involves:

1. **Learning Rate Scheduling**: Starting with a low learning rate, increasing it to a maximum value, and then decreasing it back to an extremely low value
2. **Momentum Scheduling**: Inverse scheduling of momentum (decreasing while learning rate increases, and vice versa)
3. **Cosine Annealing**: Smooth transitions using cosine functions rather than linear changes

## Key Benefits

- **Faster Training**: Often reduces training time significantly
- **Better Convergence**: Can lead to better final accuracy
- **Regularization**: Acts as an implicit regularizer, often reducing the need for other regularization techniques
- **Avoids Saddle Points**: The varying learning rate helps escape saddle points in the loss landscape
- **Enhanced Data Augmentation**: Cutout, random rotations, and other transforms via Albumentations improve generalization

## Usage

### Quick Start

```python
from train_resnet50_one_cycle import main

# Run with default parameters (CIFAR-10)
main()
```

### Command Line Usage

```bash
# Find optimal learning rate and train ResNet-50 on CIFAR-10
python train_resnet50_one_cycle.py --find_lr --epochs 20

# Train on CIFAR-100
python train_resnet50_one_cycle.py --dataset CIFAR100 --num_classes 100 --epochs 30

# Train on custom dataset
python train_resnet50_one_cycle.py --dataset custom --data_dir /path/to/data \
    --num_classes 10 --batch_size 128 --find_lr --epochs 50

# Train on ImageNet with Cutout and enhanced augmentations using Albumentations
python train_resnet50_one_cycle.py --dataset ImageNet --data_dir /path/to/imagenet \
    --num_classes 1000 --batch_size 64 --cutout --cutout_holes 1 --cutout_length 16 \
    --rotate_degrees 10 --use_albumentation --find_lr --epochs 90
```

### Learning Rate Finder

The module includes a learning rate finder that helps determine the optimal maximum learning rate:

```python
from lr_finder import LRFinder

# Create model, optimizer, criterion, and dataloader
# ...

lr_finder = LRFinder(model, optimizer, criterion, device)
lrs, losses = lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=1.0)
lr_finder.plot()  # Visualize and find optimal learning rate
```

### One-Cycle Training

Once you've found the optimal learning rate:

```python
from lr_finder import train_with_one_cycle

history = train_with_one_cycle(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    max_lr=0.01,  # The maximum learning rate found using LR finder
    num_epochs=30,
    device=device
)

# Visualize training history
from lr_finder import plot_training_history
plot_training_history(history)
```

## Command Line Arguments

The `train_resnet50_one_cycle.py` script supports the following arguments:

### Dataset Parameters
- `--data_dir`: Path to the dataset directory (default: "./data")
- `--dataset`: Dataset to use (choices: "CIFAR10", "CIFAR100", "ImageNet", "custom"; default: "CIFAR10")
- `--img_size`: Image size for training (default: 224)
- `--num_classes`: Number of classes in the dataset (default: 10)

### Training Parameters
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of epochs to train (default: 10)
- `--workers`: Number of data loading workers (default: 4)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--momentum`: Momentum for SGD optimizer (default: 0.9)

### Augmentation Parameters
- `--cutout`: Enable Cutout data augmentation (especially effective for ImageNet)
- `--cutout_holes`: Number of holes to cut out from each image (default: 1)
- `--cutout_length`: Length of the holes in pixels (default: 16)
- `--rotate_degrees`: Maximum rotation angle in degrees for augmentation (default: 10.0)
- `--use_albumentation`: Use Albumentations library for more efficient augmentations

### One-Cycle Policy Parameters
- `--find_lr`: Run learning rate finder before training
- `--max_lr`: Maximum learning rate (default: 0.01)
- `--div_factor`: Initial learning rate is max_lr/div_factor (default: 25.0)
- `--pct_start`: Percentage of training spent increasing the learning rate (default: 0.3)

### Model Parameters
- `--pretrained`: Use pretrained model
- `--checkpoint`: Path to checkpoint to resume training from

### Hardware Parameters
- `--device`: Device to use for training (default: "cuda" if available, else "cpu")

### Output Parameters
- `--output_dir`: Directory to save output files (default: "./output")
- `--save_every`: Save checkpoint every N epochs (default: 5)

## Learning Rate Finder API Reference

### `LRFinder` Class

```python
class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        """
        Initialize the learning rate finder.
        
        Parameters:
            model (torch.nn.Module): PyTorch model
            optimizer (torch.optim.Optimizer): PyTorch optimizer
            criterion (callable): Loss function
            device (torch.device): Device for training
        """
        pass
        
    def range_test(self, train_loader, start_lr=1e-7, end_lr=10, 
                   num_iter=100, step_mode="exp", smooth_f=0.05, diverge_th=5.0):
        """
        Run learning rate range test.
        
        Parameters:
            train_loader (DataLoader): PyTorch DataLoader
            start_lr (float): Starting learning rate
            end_lr (float): Maximum learning rate to test
            num_iter (int): Number of iterations
            step_mode (str): "exp" or "linear" increase
            smooth_f (float): Smoothing factor
            diverge_th (float): Divergence threshold
            
        Returns:
            tuple: (learning_rates, losses)
        """
        pass
        
    def plot(self, skip_start=10, skip_end=5, log_lr=True, title="Learning Rate Finder"):
        """
        Plot learning rate vs loss.
        
        Parameters:
            skip_start (int): Batches to skip at start
            skip_end (int): Batches to skip at end
            log_lr (bool): Use log scale for learning rate
            title (str): Plot title
            
        Returns:
            int: Index of suggested learning rate
        """
        pass
```

## One-Cycle Training Function

```python
def train_with_one_cycle(model, train_loader, val_loader, optimizer, criterion, 
                        max_lr, num_epochs, device="cuda"):
    """
    Train with one-cycle policy.
    
    Parameters:
        model (torch.nn.Module): PyTorch model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (torch.optim.Optimizer): PyTorch optimizer
        criterion (callable): Loss function
        max_lr (float): Maximum learning rate
        num_epochs (int): Number of training epochs
        device (str): Device for training
        
    Returns:
        dict: Training history
    """
    pass
```

## References

1. Smith, Leslie N. "A disciplined approach to neural network hyper-parameters: Part 1 - learning rate, batch size, momentum, and weight decay." arXiv preprint arXiv:1803.09820 (2018).
2. Smith, Leslie N., and Nicholay Topin. "Super-convergence: Very fast training of neural networks using large learning rates." arXiv preprint arXiv:1708.07120 (2017).
3. DeVries, Terrance, and Graham W. Taylor. "Improved regularization of convolutional neural networks with cutout." arXiv preprint arXiv:1708.04552 (2017).
4. Buslaev, A., et al. "Albumentations: Fast and flexible image augmentations." Information 11.2 (2020): 125.

## License

This implementation is provided under the MIT License.