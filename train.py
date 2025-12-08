"""
Training script for channel-adaptive ViT models.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import sys
import random
from typing import Any, Optional

# Disable tqdm's default file writing to ensure proper in-place updates
tqdm.monitor_interval = 0

from config import Config
from channel_adaptive_pipeline.models.early_fusion_vit import EarlyFusionViT
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_grouped_chammi_dataloaders
from channel_adaptive_pipeline.losses import LabelSmoothingCrossEntropy
from channel_adaptive_pipeline.schedulers import get_optimizer, get_scheduler
from channel_adaptive_pipeline.model_utils import (
    get_num_classes_from_metadata,
    save_model_checkpoint,
    load_model_checkpoint,
    count_parameters,
    get_class_to_idx_mapping,
)
from channel_adaptive_pipeline.logging_utils import Logger, print_metrics

def cleanup_old_runs(config: Config):
    """Remove existing checkpoints and logs for a fresh start."""
    import shutil
    
    # Remove checkpoints
    if os.path.exists(config.checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(config.checkpoint_dir) 
                           if f.endswith('.pth')]
        if checkpoint_files:
            print(f"Removing {len(checkpoint_files)} existing checkpoint(s)...")
            for f in checkpoint_files:
                os.remove(os.path.join(config.checkpoint_dir, f))
            print("Checkpoints cleaned.")
        else:
            print("No existing checkpoints found.")
    else:
        print("Checkpoint directory doesn't exist yet.")
    
    # Remove logs
    if os.path.exists(config.log_dir):
        log_files = [f for f in os.listdir(config.log_dir) 
                     if f.endswith('.csv') or f.endswith('.log')]
        if log_files:
            print(f"Removing {len(log_files)} existing log file(s)...")
            for f in log_files:
                os.remove(os.path.join(config.log_dir, f))
            print("Logs cleaned.")
        else:
            print("No existing logs found.")
    else:
        print("Log directory doesn't exist yet.")
    
    # Remove TensorBoard logs
    tb_log_dir = os.path.join(config.log_dir, config.experiment_name)
    if os.path.exists(tb_log_dir):
        print("Removing TensorBoard logs...")
        shutil.rmtree(tb_log_dir)
        print("TensorBoard logs cleaned.")

def train_epoch(
    model: nn.Module,
    dataloaders: dict,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: Config,
    epoch: int,
    logger: Logger,
    class_to_idx: dict,
):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloaders: Dictionary of dataloaders by channel count
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Config object
        epoch: Current epoch number
        logger: Logger instance
        class_to_idx: Mapping from class names to indices
    
    Returns:
        Dictionary of metrics
    """
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Metrics per channel count
    metrics_per_channel = {3: {'loss': 0.0, 'correct': 0, 'samples': 0},
                          4: {'loss': 0.0, 'correct': 0, 'samples': 0},
                          5: {'loss': 0.0, 'correct': 0, 'samples': 0}}
    
    # Option 2: Randomize batch-by-batch - create iterators for each channel group
    channel_counts = list(dataloaders.keys())
    iterators = {ch: iter(dl) for ch, dl in dataloaders.items()}
    
    # Track which channels are exhausted and batch counts
    exhausted_channels = set()
    batch_counts = {ch: 0 for ch in channel_counts}
    total_batches = sum(len(dl) for dl in dataloaders.values())
    
    # Create a single progress bar for the entire epoch
    pbar = tqdm(
        total=total_batches,
        desc=f"Epoch {epoch+1}/{config.num_epochs} [Random]",
        unit="batch",
        file=sys.stdout,
        ncols=120,
        mininterval=1.0,
        maxinterval=10.0,
        miniters=10,
        dynamic_ncols=False,
        leave=False,
        smoothing=0.1,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    global_batch_idx = 0
    
    # Process batches in random order until all channels are exhausted
    while len(exhausted_channels) < len(channel_counts):
        # Get available (non-exhausted) channels
        available_channels = [ch for ch in channel_counts if ch not in exhausted_channels]
        
        if not available_channels:
            break
        
        # Randomly select which channel to process next
        channel_count = random.choice(available_channels)
        dataloader = dataloaders[channel_count]
        
        try:
            batch_images, batch_metadatas, batch_labels = next(iterators[channel_count])
            
            # Move to device
            batch_images = batch_images.to(device)
            
            # Convert labels to indices if they're strings
            if isinstance(batch_labels[0], str):
                batch_labels_tensor = torch.tensor([
                    class_to_idx[label] if label in class_to_idx else 0
                    for label in batch_labels
                ], dtype=torch.long, device=device)
            else:
                batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
            
            # Forward pass
            logits = model(batch_images)
            
            # Compute loss
            loss = criterion(logits, batch_labels_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == batch_labels_tensor).sum().item()
            
            batch_size = batch_images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size
            
            metrics_per_channel[channel_count]['loss'] += loss.item() * batch_size
            metrics_per_channel[channel_count]['correct'] += correct
            metrics_per_channel[channel_count]['samples'] += batch_size
            
            batch_counts[channel_count] += 1
            global_batch_idx += 1
            
            # Update progress bar
            if global_batch_idx % 10 == 0 or global_batch_idx == total_batches:
                current_loss = total_loss / total_samples
                current_acc = total_correct / total_samples
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.4f}',
                    'C': channel_count
                })
            
            pbar.update(1)
            
            # Log metrics periodically
            if global_batch_idx % config.log_freq == 0:
                current_loss = total_loss / total_samples
                current_acc = total_correct / total_samples
                step = epoch * total_batches + global_batch_idx
                logger.log({
                    'train_loss': current_loss,
                    'train_acc': current_acc,
                    'lr': config.learning_rate,  # Will be updated after epoch
                }, step)
            
        except StopIteration:
            # This channel is exhausted
            exhausted_channels.add(channel_count)
    
    pbar.close()
    
    # Compute final metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Per-channel metrics
    channel_metrics = {}
    for ch, m in metrics_per_channel.items():
        if m['samples'] > 0:
            channel_metrics[f'loss_ch{ch}'] = m['loss'] / m['samples']
            channel_metrics[f'acc_ch{ch}'] = m['correct'] / m['samples']
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        **channel_metrics,
    }
    
    return metrics


def validate(
    model: nn.Module,
    dataloaders: dict,
    criterion: nn.Module,
    device: torch.device,
    class_to_idx: dict,
):
    """
    Validate the model.
    
    Args:
        model: Model to validate
        dataloaders: Dictionary of dataloaders by channel count
        criterion: Loss function
        device: Device to validate on
        class_to_idx: Mapping from class names to indices
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for channel_count, dataloader in dataloaders.items():
            for batch_images, batch_metadatas, batch_labels in dataloader:
                batch_images = batch_images.to(device)
                
                # Convert labels to indices
                if isinstance(batch_labels[0], str):
                    batch_labels_tensor = torch.tensor([
                        class_to_idx[label] if label in class_to_idx else 0
                        for label in batch_labels
                    ], dtype=torch.long, device=device)
                else:
                    batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long, device=device)
                
                # Forward pass
                logits = model(batch_images)
                
                # Compute loss
                loss = criterion(logits, batch_labels_tensor)
                
                # Update metrics
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == batch_labels_tensor).sum().item()
                
                batch_size = batch_images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += correct
                total_samples += batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description='Train channel-adaptive ViT model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--clean', action='store_true', help='Clean existing checkpoints and logs before training')
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Override config from file if provided
    if args.config and os.path.exists(args.config):
        # Could load from JSON/YAML here if needed
        pass

    if args.clean:
        cleanup_old_runs(config)
    
    # Set device
    device = torch.device(config.device if config.device == 'cpu' else f'cuda:{config.gpu_id}')
    print(f"Using device: {device}")
    
    # Auto-detect number of classes
    print("Auto-detecting number of classes from metadata...")
    num_classes = get_num_classes_from_metadata(
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        target_labels=config.target_labels,
    )
    config.num_classes = num_classes
    print(f"Found {num_classes} classes")
    
    # Get class to index mapping
    class_to_idx = get_class_to_idx_mapping(
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        target_labels=config.target_labels,
    )
    
    # Initialize model
    print(f"Initializing {config.model_type} ViT-{config.vit_size}...")
    if config.model_type == 'early_fusion':
        model = EarlyFusionViT(
            num_classes=num_classes,
            img_size=config.img_size,
            patch_size=config.patch_size,
        )
    else:
        raise ValueError(f"Model type {config.model_type} not implemented yet")
    
    model = model.to(device)
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Initialize loss function
    criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    
    # Initialize optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_metric = 0.0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_info = load_model_checkpoint(model, args.resume, device, optimizer, scheduler)
        start_epoch = checkpoint_info['epoch'] + 1
        best_metric = checkpoint_info['metrics'].get('val_accuracy', 0.0)
        print(f"Resumed from epoch {start_epoch}, best metric: {best_metric:.4f}")
    
    # Create dataloaders
    print("Creating training dataloaders...")
    train_dataloaders = create_grouped_chammi_dataloaders(
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        batch_size=config.batch_size,
        shuffle=True,
        target_labels=config.target_labels,
        split='train',
        resize_to=config.img_size,
        augment=config.augment,
        normalize=config.normalize,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    print("Creating validation dataloaders...")
    val_dataloaders = create_grouped_chammi_dataloaders(
        csv_file=config.csv_file,
        root_dir=config.root_dir,
        batch_size=config.batch_size,
        shuffle=False,
        target_labels=config.target_labels,
        split=config.sd_split,
        resize_to=config.img_size,
        augment=False,
        normalize=config.normalize,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    # Initialize logger
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name=config.experiment_name,
        use_tensorboard=config.use_tensorboard,
        use_csv=True,
    )
    
    # Training loop
    print(f"\nStarting training for {config.num_epochs} epochs...")
    
    for epoch in range(start_epoch, config.num_epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloaders=train_dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,  # Don't pass scheduler to train_epoch, step it after epoch
            device=device,
            config=config,
            epoch=epoch,
            logger=logger,
            class_to_idx=class_to_idx,
        )
        
        # Update learning rate (once per epoch, after all channel groups)
        if scheduler is not None:
            scheduler.step()
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloaders=val_dataloaders,
            criterion=criterion,
            device=device,
            class_to_idx=class_to_idx,
        )
        
        # Log metrics
        all_metrics = {**{f'train_{k}': v for k, v in train_metrics.items()},
                       **{f'val_{k}': v for k, v in val_metrics.items()}}
        all_metrics['lr'] = scheduler.get_last_lr()[0] if scheduler else config.learning_rate
        logger.log(all_metrics, epoch)
        
        # Print metrics
        print_metrics(train_metrics, prefix=f"Epoch {epoch+1} - Train")
        print_metrics(val_metrics, prefix=f"Epoch {epoch+1} - Val")
        if scheduler:
            print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_metric
        if is_best:
            best_metric = val_metrics['accuracy']
        
        if (epoch + 1) % config.save_freq == 0 or is_best:
            checkpoint_path = config.get_checkpoint_path(epoch=epoch + 1)
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=all_metrics,
                save_path=checkpoint_path,
                is_best=is_best,
            )
            if is_best:
                print(f"Saved best model (val_acc: {best_metric:.4f})")
    
    # Close logger
    logger.close()
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

