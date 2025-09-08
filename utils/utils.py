import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch import nn
from torch.utils.data import Subset, random_split


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by the divisor
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def load_yaml(path):
    """
    Loads a YAML file from the given path.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict or None: Parsed YAML contents, or None if loading fails.
    """
    try:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"File not found: {path}")
    except yaml.YAMLError as exc:
        print(f"YAML error: {exc}")
    return None


def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        float: Number of parameters (in millions).
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def resume_checkpoint(model, optimizer, config, output_dir):
    """
    Loads model and optimizer state from a checkpoint, if available.

    Args:
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer): Optimizer to restore.
        config (dict): Configuration dictionary.
        output_dir (str): Directory containing the checkpoint.
        which(str) : Resume training from latest or best checkpoint.

    Returns:
        tuple: (best_perf, begin_epoch)
    """
    best_perf = 0.0
    begin_epoch = 0

    if not config["TRAIN_CHECKPOINT"]:
        checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    else:
        checkpoint_path = config["TRAIN_CHECKPOINT"]

    print(f"Looking for a checkpoint at {checkpoint_path} ...")

    if config["AUTO_RESUME"] and os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}")
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
        best_perf = checkpoint_dict["perf"]
        begin_epoch = checkpoint_dict["epoch"]
        state_dict = checkpoint_dict["state_dict"]
        model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    return best_perf, begin_epoch


def save_checkpoint(model, optimizer, output_dir, epoch_num, best_perf):
    """
    Saves a checkpoint of the model and optimizer.

    Args:
        model (torch.nn.Module): Model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        output_dir (str): Directory to save the checkpoint.
        epoch_num (int): Current training epoch.
        best_perf (float): Best performance metric so far.
    """
    save_dict = {
        "epoch": epoch_num,
        "state_dict": model.state_dict(),
        "perf": best_perf,
        "optimizer": optimizer.state_dict()
    }

    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")

    try:
        torch.save(save_dict, checkpoint_path)
        print(f"=> Checkpoint saved at epoch {epoch_num} to '{checkpoint_path}'")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def save_best_model(model, save_dir):
    """
    Saves the model weights as the best-performing model.

    Args:
        model (torch.nn.Module): Model to save.
        save_dir (str): Directory to save the model.
    """
    try:
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        print("=> Best model updated.")
    except Exception as e:
        print(f"Error saving best model: {e}")


def build_dataloader(dataset, config, is_train):
    """
    Creates a DataLoader from a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load from.
        config (dict): Config with keys : 'batch_size', 'num_workers', 'pin_memory'.
        is_train (bool): Whether the loader is for training (affects shuffle and drop_last).

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """

    required_keys = ["train_batch_size", "val_batch_size", "num_workers", "pin_memory"]
    for key in required_keys:
        assert key in config, f"Missing key in config: {key}"

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train_batch_size"] if is_train else config["val_batch_size"],
        shuffle=is_train,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        drop_last=is_train,
    )

    return data_loader


def _is_depthwise(module):
    """
    Check if a given module is a depthwise convolution layer.

    A depthwise convolution is characterized by the number of groups being
    equal to both the number of input and output channels.

    Args:
        module (nn.Module): Module to check.

    Returns:
        bool: True if the module is a depthwise Conv2d layer, False otherwise.
    """
    return (
        isinstance(module, nn.Conv2d)
        and module.groups == module.in_channels
        and module.groups == module.out_channels
    )


def set_decay(config, model):
    """
    Separate model parameters into two groups: with and without weight decay.

    Applies rules from `config["TRAIN_WITHOUT_WD"]` to determine which parameters
    should not receive weight decay (e.g., depthwise convs, norms, biases).

    Args:
        config (dict): Configuration dictionary containing training preferences.
        model (nn.Module): The model whose parameters are to be grouped.

    Returns:
        List[dict]: A list containing two dictionaries:
            - One with parameters using default weight decay.
            - One with parameters using zero weight decay.
    """
    without_decay_list = config["TRAIN_WITHOUT_WD"]
    without_decay = set()

    # Adds the following modules to the without_decay set
    for m in model.modules():
        if _is_depthwise(m) and "dw" in without_decay_list:
            without_decay.add(m.weight)
        elif isinstance(m, nn.BatchNorm2d) and "bn" in without_decay_list:
            without_decay.update([m.weight, m.bias])
        elif isinstance(m, nn.GroupNorm) and "gn" in without_decay_list:
            without_decay.update([m.weight, m.bias])
        elif isinstance(m, nn.LayerNorm) and "ln" in without_decay_list:
            without_decay.update([m.weight, m.bias])

    # Adds named parameters ending with .bias to the without_decay set
    if "bias" in config["TRAIN_WITHOUT_WD"]:
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                without_decay.add(param)

    params_with_decay = []
    params_without_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in without_decay:
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    return [
        {"params": params_with_decay},
        {"params": params_without_decay, "weight_decay": 0.0},
    ]


def build_optimizer(config, model):
    """
    Create an optimizer with custom weight decay settings.

    Constructs an optimizer (Adam or AdamW) and applies selective weight decay
    based on configuration and model structure.

    Args:
        config (dict): Configuration containing optimizer name, learning rate,
                       weight decay, and parameter exclusion rules.
        model (nn.Module): The model to optimize.

    Returns:
        torch.optim.Optimizer: Instantiated optimizer with parameter groups.
    """
    optimizer = None
    params = set_decay(config, model)

    if config["optimizer_name"] == "Adam":
        optimizer = optim.Adam(
            params,
            lr=config["train_lr"],
            weight_decay=config["train_wd"],
        )
    elif config["optimizer_name"] == "AdamW":
        optimizer = optim.AdamW(
            params,
            lr=config["train_lr"],
            weight_decay=config["train_wd"],
        )
    else:
        raise ValueError(f"Unknown optimizer name {config['optimizer_name']}")

    return optimizer


def build_criterion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.125).to(device)
    return criterion


def build_lr_scheduler(config, optimizer, begin_epoch, prev_best=None):
    """
    Builds and returns a learning rate scheduler based on the specified configuration.

    Supports:
        - CosineAnnealingLR: Cosine annealing learning rate with optional warm start.
        - ReduceLROnPlateau: Reduces learning rate when a metric has stopped improving.

    Args:
        config (dict): Configuration dictionary containing scheduler type and parameters.
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        begin_epoch (int): The starting epoch index (used to initialize scheduler state).

    Returns:
        _LRScheduler or ReduceLROnPlateau: Configured PyTorch learning rate scheduler.

    Raises:
        ValueError: If an unknown scheduler name is provided in config["lr_scheduler_name"].
    """
    lr_scheduler = None
    if config["lr_scheduler_name"] == "CosineAnnealing":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config["train_epochs"], float(config["eta_min"]), begin_epoch - 1
        )
    elif config["lr_scheduler_name"] == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["reduce_lr_factor"],
            patience=config["patience_epochs"],
            threshold=float(config["loss_reduction_threshold"]),
        )
    else:
        raise ValueError(f"Unknown lr_scheduler {config['lr_scheduler_name']}")

    return lr_scheduler


def step_scheduler(scheduler, val_loss=None):
    """
    Steps the learning rate scheduler based on its type.

    - For ReduceLROnPlateau, `val_loss` must be provided and is used to determine if LR should be reduced.
    - For other schedulers (like CosineAnnealingLR), `step()` is called without arguments.

    Args:
        scheduler (torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau):
            The learning rate scheduler to step.
        val_loss (float, optional):
            Validation loss to pass to ReduceLROnPlateau. Required if scheduler is of that type.

    Raises:
        AssertionError: If scheduler is ReduceLROnPlateau and val_loss is not provided.
    """

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        assert val_loss is not None, "val_loss required for ReduceLROnPlateau"
        scheduler.step(val_loss)
    else:
        scheduler.step()


class AverageMeter(object):
    """
    Tracks and updates the average of a scalar quantity (e.g., loss or accuracy).

    Useful during training or evaluation to maintain running averages over time.
    """

    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes the fraction of correct predictions in a batch.

    Args:
        preds (Tensor): Model output probabilities of shape (B, C).
        targets (Tensor): Ground truth labels of shape (B,).

    Returns:
        float: Accuracy as a fraction between 0 and 1.
    """
    # Argmax to be used in case of multi-class classification
    predicted_labels = torch.argmax(preds, dim=1)

    correct = (predicted_labels == targets).sum().item()
    total = targets.size(0)
    return correct / total

def get_topk_accuracy(logits, labels, k):
    """
    Computes the Top-K classification accuracy.

    Args:
        logits (torch.Tensor): Model output logits of shape (batch_size, num_classes).
        labels (torch.Tensor): Ground truth class labels of shape (batch_size,).
        k (int): The number of top predictions to consider (Top-K).

    Returns:
        float: Top-K accuracy as a fraction between 0 and 1.
    """
    assert (
        isinstance(logits, torch.Tensor) and logits.ndim == 2
    ), "Logits must be 2-dimensional tensors"
    assert (
        isinstance(labels, torch.Tensor) and labels.ndim == 1
    ), "Labels must be 1-dimensional tensors"
    assert isinstance(k, int) and k > 0, "K must be an integer greater than 0"

    # Top-k indices along classes
    topk_preds = torch.topk(logits, k, dim=1).indices
    labels = labels.view(-1, 1).expand_as(topk_preds)

    correct = (topk_preds == labels).any(dim=1).sum().item()
    total = labels.size(0)

    return correct / total


@torch.no_grad()
def validate_model(val_loader, device, model, criterion, k=3, amp_enabled=True):
    model.eval()
    losses = AverageMeter()
    accuracy = AverageMeter()

    for images, targets in val_loader:
        x = images.to(device, non_blocking=True)
        y = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(x, return_logits=True)
            loss = criterion(outputs, y)

        losses.update(loss.item(), x.size(0))
        acc = get_topk_accuracy(outputs, y, k)
        accuracy.update(acc, x.size(0))

    print(f"Avg Loss: {losses.avg:.3f}\n" f"Validation Accuracy: {accuracy.avg:.3f}")
    return losses.avg, accuracy.avg
