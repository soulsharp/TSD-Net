import argparse
import os
import pprint
import random
import time

import numpy as np
import torch
import yaml
from torch import GradScaler

from data.dataset import prepare_cifar10_dataset
from model.classifier import TSD_Classifier
from utils.utils import (
    AverageMeter,
    build_criterion,
    build_dataloader,
    build_lr_scheduler,
    build_optimizer,
    compute_accuracy,
    count_parameters,
    load_yaml,
    resume_checkpoint,
    save_best_model,
    save_checkpoint,
    step_scheduler,
    validate_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(config, train_loader, model, criterion, optimizer, epoch, scaler):
    """
    Trains the model for a single epoch using mixed precision (if enabled), gradient scaling,
    and optional gradient clipping.

    Tracks and prints batch-wise and average metrics like loss and accuracy.

    Args:
        config (dict): Training configuration containing:
            - "amp_enabled" (bool): Whether to use automatic mixed precision.
            - "clip_grad_norm" (float): Max norm for gradient clipping. Set to 0 to disable.
        train_loader (DataLoader): DataLoader for the training data.
        model (nn.Module): The model to train.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer to update model weights.
        epoch (int): Current epoch index (for logging).
        scaler (GradScaler): PyTorch GradScaler for AMP training.

    Returns:
        tuple: (avg_loss, avg_accuracy) for the epoch.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.train()
    end = time.time()

    print_frequency = len(train_loader) // 5

    for i, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x = images.to(device)
        y = targets.to(device)

        with torch.autocast(device.type, enabled=config["amp_enabled"]):
            # In multi-class classification return_logits is to be set to False
            out = model(x, return_logits=False)
            loss = criterion(out, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if config.get("clip_grad_norm", 0.0) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])

        scaler.step(optimizer)
        scaler.update()

        # Metrics
        losses.update(loss.item(), x.size(0))
        acc = compute_accuracy(out, y)
        accuracy.update(acc, x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_frequency == 0:
            print(
                f"Epoch[{epoch}][{i}/{len(train_loader)}]\n"
                f"Avg_time per batch : {batch_time.avg:.3f}s\n"
                f"Avg dataloading time : {data_time.avg:.3f}\n"
                f"Loss: Current_batch = {losses.val:.3f}, Avg={losses.avg:.3f}\n"
                f"Accuracy : Current_batch = {accuracy.val:.3f}, Avg={accuracy.avg:.3f}"
            )


def train(args):
    """
    Executes the full training loop for the TSD_Classifier model.

    This includes:
      - Parsing configuration from a YAML file.
      - Setting deterministic seeds for reproducibility.
      - Initializing the model, optimizer, loss function, AMP scaler, and learning rate scheduler.
      - Loading and splitting the dataset into train/val/test sets (with persistent split saving).
      - Resuming from a checkpoint if available.
      - Iteratively training and validating the model for the specified number of epochs.
      - Performing threshold sweeping on validation set to find the best F1 score and corresponding threshold.
      - Saving the latest model checkpoint and best-performing model (based on validation F1 score).
      - Writing the best threshold to the config file for later use during testing.

    Args:
        args (argparse.Namespace): Parsed command-line arguments. Should include:
            - config_path (str): Path to YAML config file with training parameters.
            - num_classes (int): Number of target classes for classification.

    Raises:
        ValueError: If the config file fails to load or is malformed.

    Effects:
        - Creates model checkpoints in the specified checkpoint directory.
        - Modifies the YAML config to record the best validation threshold.
    """

    # Reads the config file #
    config = load_yaml(args.config_path)
    print("Training config: \n")
    pprint.pprint(config)

    if config is None:
        raise ValueError(f"Failed to load config file: {args.config_path}")

    # Config params
    dataset_config = config["dataset_params"]
    tsd_config = config["tsd_params"]
    train_config = config["train_params"]

    # # Sets the desired seed value
    # seed = train_config['seed']
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # if device == 'cuda':
    #     torch.cuda.manual_seed_all(seed)

    # model = TSD_Classifier(args.num_classes, tsd_config["emb_dim"],
    #                        tsd_config["num_heads_tiny"], tsd_config["num_encoder_layers_tiny"],
    #                        tsd_config["expansion_ratio"], tsd_config["cte_output_channels"])

    # Using TSD-B (TSD-Big)
    model = TSD_Classifier(
        args.num_classes,
        tsd_config["emb_dim"],
        tsd_config["num_heads_big"],
        tsd_config["num_encoder_layers_big"],
        tsd_config["expansion_ratio"],
        tsd_config["cte_output_channels"],
    )

    model = model.to(device)
    print(f"Num_parameters : {count_parameters(model)}M")
    if train_config.get("compile_model", True):
        model = torch.compile(model, mode="reduce-overhead")

    train_dataset, val_dataset = prepare_cifar10_dataset(dataset_config)

    best_perf = 0.0
    begin_epoch = train_config["begin_train_epoch"]
    num_epochs = train_config["train_epochs"]
    optimizer = build_optimizer(train_config, model)

    # Resumes training from a checkpoint
    checkpoint_dir = train_config["checkpoint_dir"]
    checkpoint_save_path = os.path.join(checkpoint_dir, "latest")
    best_model_path = os.path.join(checkpoint_dir, "best")

    best_perf, begin_epoch = resume_checkpoint(
        model, optimizer, train_config, checkpoint_save_path
    )

    # Builds dataloaders, criterion and scheduler
    train_loader = build_dataloader(train_dataset, train_config, is_train=True)
    val_loader = build_dataloader(val_dataset, train_config, is_train=False)
    criterion = build_criterion().to(device)
    lr_scheduler = build_lr_scheduler(train_config, optimizer, begin_epoch)

    scaler = GradScaler(device.type, enabled=train_config["amp_enabled"])

    print("Start of training...")
    for epoch in range(begin_epoch, num_epochs):
        start_time = time.time()
        train_one_epoch(
            train_config, train_loader, model, criterion, optimizer, epoch, scaler
        )
        print(f"End of training epoch {epoch}.Took {(time.time() - start_time):.3f}s")

        print("Starting validation...")
        val_loss, acc = validate_model(val_loader, device, model, criterion)

        if acc > best_perf:
            best_perf = acc
            save_best_model(model, best_model_path)

        print(f"Current epoch acc: {acc:.3f}, Best acc: {best_perf:.3f}")

        step_scheduler(lr_scheduler, val_loss)

        save_checkpoint(model, optimizer, checkpoint_save_path, epoch, best_perf)

    print("End of training...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for TSD classifier training"
    )
    parser.add_argument(
        "--config", dest="config_path", default="config/config.yaml", type=str
    )
    parser.add_argument("--num_classes", default=2, type=int)
    args = parser.parse_args()
    train(args)
