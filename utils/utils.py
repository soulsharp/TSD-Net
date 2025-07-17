import torch
from torch import nn
import yaml
import os
import torch.optim as optim
from torch.utils.data import random_split, Subset
import json
import time
import numpy as np


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
        with open(path, 'r') as file:
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
    return params/1000000

def resume_checkpoint(model, optimizer, config, output_dir, which="latest"):
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
        checkpoint = os.path.join(output_dir, 'checkpoint.pth')
    else:
        checkpoint = config["TRAIN_CHECKPOINT"]

    if config["AUTO_RESUME"] and os.path.exists(checkpoint):

        checkpoint_dict = torch.load(checkpoint, map_location='cpu')
        best_perf = checkpoint_dict['perf']
        begin_epoch = checkpoint_dict['epoch']
        state_dict = checkpoint_dict['state_dict']
        model.load_state_dict(state_dict)

        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    
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
        'epoch': epoch_num,
        'state_dict': model.state_dict(),
        'perf': best_perf,
        'optimizer': optimizer.state_dict(),
    }

    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')

    try:
        torch.save(save_dict, checkpoint_path)
        print(f"=> Checkpoint saved at epoch {epoch_num} to '{checkpoint_path}'")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def save_best_model(model, output_dir):
    """
    Saves the model weights as the best-performing model.

    Args:
        model (torch.nn.Module): Model to save.
        output_dir (str): Directory to save the model.
    """
    try:
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
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
            batch_size = config["train_batch_size"] if is_train else config["val_batch_size"],
            shuffle=is_train,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            drop_last=is_train,
        )

    return data_loader

def save_splits(split_dict, save_path):
    with open(save_path, 'w') as f:
        json.dump(split_dict, f)

def load_splits(load_path):
    with open(load_path, 'r') as f:
        split_dict = json.load(f)
    return {k: list(map(int, v)) for k, v in split_dict.items()}

def split_dataset(im_dataset, train_config, split_file_path="dataset/splits.json", seed=42):
    """
    Splits a dataset into training, validation, and test sets using a fixed seed
    and optionally saves/loads the split indices to/from disk.

    Args:
        im_dataset (Dataset): The full dataset to be split.
        train_config (dict): Dictionary with "val_split" and "test_split" float values.
        split_file_path (str): Path to a .json file for saving/loading split indices.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset), each a Subset of im_dataset.
    """
    dataset_len = len(im_dataset)
    val_size = int(dataset_len * train_config["val_split"])
    test_size = int(dataset_len * train_config["test_split"])
    train_size = dataset_len - val_size - test_size

    if os.path.exists(split_file_path):
        split_dict = load_splits(split_file_path)
    else:
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = random_split(
            im_dataset, [train_size, val_size, test_size], generator=generator
        )

        split_dict = {
            "train": train_ds.indices,
            "val": val_ds.indices,
            "test": test_ds.indices,
        }
        save_splits(split_dict, split_file_path)

    train_dataset = Subset(im_dataset, split_dict["train"])
    val_dataset = Subset(im_dataset, split_dict["val"])
    test_dataset = Subset(im_dataset, split_dict["test"])

    return train_dataset, val_dataset, test_dataset

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
    return(
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
        if _is_depthwise(m) and 'dw' in without_decay_list:
            without_decay.add(m.weight)
        elif isinstance(m, nn.BatchNorm2d) and 'bn' in without_decay_list:
            without_decay.update([m.weight, m.bias])
        elif isinstance(m, nn.GroupNorm) and 'gn' in without_decay_list:
            without_decay.update([m.weight, m.bias])
        elif isinstance(m, nn.LayerNorm) and 'ln' in without_decay_list:
            without_decay.update([m.weight, m.bias])
    
    # Adds named parameters ending with .bias to the without_decay set
    if 'bias' in config["TRAIN_WITHOUT_WD"]:
        for name, param in model.named_parameters():
            if name.endswith('.bias'):
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
        {'params': params_with_decay},
        {'params': params_without_decay, 'weight_decay': 0.0}
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

# Could possibly build custom criterions in the future for other training tasks
def build_criterion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 15.0]).to(device))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]))
    return criterion

def build_lr_scheduler(config, optimizer, begin_epoch):
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
            optimizer,
            config["train_epochs"],
            float(config["eta_min"]),
            begin_epoch - 1
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
    # # Argmax to be used in case of multi-class classification
    # predicted_labels = torch.argmax(preds, dim=1)
    
    # This works only in the case of binary classification
    predicted_labels = preds > 0.35
    correct = (predicted_labels == targets).sum().item()
    total = targets.size(0)
    return correct / total

def compute_f1_score(preds: torch.Tensor, targets: torch.Tensor):
    """
    Computes the precision, recall, and F1 score for binary classification.

    Args:
        preds (torch.Tensor): Predicted binary labels (0 or 1), of shape (N,).
        targets (torch.Tensor): Ground-truth binary labels (0 or 1), of shape (N,).

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - precision (float): True positives / (True positives + False positives)
            - recall (float): True positives / (True positives + False negatives)
            - f1 (float): Harmonic mean of precision and recall
    """
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.squeeze(1)
    
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    preds = preds.int()
    targets = targets.int()

    assert preds.ndim == 1 and targets.ndim == 1, "Both inputs must be 1D tensors"
    assert torch.all((preds == 0) | (preds == 1)), "Predictions must be binary (0 or 1)"
    assert torch.all((targets == 0) | (targets == 1)), "Targets must be binary (0 or 1)"
    
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    
    epsilon = float(1e-8)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    
    return precision.item(), recall.item(), f1.item()

@torch.no_grad
def test(test_loader, model, criterion, threshold):
    """
    Runs final evaluation of a trained model on the test set using a fixed threshold.

    This function:
        - Computes loss, accuracy, precision, recall, and F1 score.
        - Applies a fixed threshold to logits for binary classification.
        - Assumes the model is already trained and uses no gradient tracking.

    Args:
        test_loader (DataLoader): DataLoader for the test set.
        model (nn.Module): Trained model to evaluate.
        criterion (nn.Module): Loss function used to compute evaluation loss.
        threshold (float): Threshold for converting logits to binary predictions.

    Returns:
        tuple: A 5-tuple containing:
            - avg_loss (float): Mean loss over all test batches.
            - avg_accuracy (float): Mean accuracy.
            - precision (float): Precision over the full test set.
            - recall (float): Recall over the full test set.
            - f1 (float): F1 score over the full test set.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    all_preds = []
    all_targets = []

    batch_time = AverageMeter()
    losses = AverageMeter()
    val_accuracy = AverageMeter()

    end = time.time()
    for batch in test_loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device).float()

        outputs = model(x, return_logits=True)

        #Squeeze applied for BCE consistency
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, y)
        losses.update(loss.item(), x.size(0))

        # preds = torch.argmax(outputs, dim=1)
        preds = outputs > threshold
        acc = compute_accuracy(outputs, y)
        val_accuracy.update(acc, x.size(0))

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

        batch_time.update(time.time() - end)
        end = time.time()

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    p, r, f = compute_f1_score(all_preds, all_targets)

    print(
        f"Avg_time per batch : {batch_time.avg:.3f}s\n"
        f"Avg Loss: {losses.avg:.3f}\n"
        f"Avg Accuracy: {val_accuracy.avg:.3f}\n"
        f"Precision: {p:.3f}, Recall: {r:.3f}, F1: {f:.3f}"
    )

    return losses.avg, val_accuracy.avg, p, r, f

@torch.no_grad
def validate_model(val_loader, device, model, criterion):
    """
    Evaluates the model on the validation set using multiple thresholds to find the best F1-score.

    Args:
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run inference on (CPU or CUDA).
        model (torch.nn.Module): The model to evaluate. Must support return_logits=True if needed.
        criterion (torch.nn.Module): Loss function used for evaluation.

    Returns:
        tuple:
            - max_f1 (float): Maximum F1-score achieved across thresholds.
            - best_threshold (float): Threshold value that yielded max F1-score.
            - avg_loss (float): Average validation loss.
    """
    model.eval()
    logits = []
    targets = []
    losses = AverageMeter()

    for batch in val_loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device).float()
        outputs = model(x, return_logits=True)

        #Squeeze applied for BCE consistency
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, y)
        losses.update(loss.item(), x.size(0))
        logits.append(outputs.cpu())
        targets.append(y.cpu())
    
    logits = torch.cat(logits)
    targets = torch.cat(targets) 

    thresholds = torch.arange(0.1, 0.91, 0.05)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for t in thresholds:
        preds = (logits > t)
        p, r, f1 = compute_f1_score(preds, targets)
        f1_scores.append(f1)
        precision_scores.append(p)
        recall_scores.append(r)
    
    max_f1_idx = np.argmax(f1_scores)
    max_precision = precision_scores[max_f1_idx]
    max_recall = recall_scores[max_f1_idx]
    max_f1 = f1_scores[max_f1_idx]

    print(
        f"Avg Loss: {losses.avg:.3f}\n"
        f"Precision: {max_precision:.3f}, Recall: {max_recall:.3f}, F1: {max_f1:.3f}"
    )
    
    return max_f1, thresholds[max_f1_idx].item(), losses.avg