"""
Training script for supervised Bag-of-Channels ViT on CHAMMI.

Supports:
- Cross-entropy (CE) training
- ProxyNCA++ metric learning
- Mixed precision on MPS
- Gradient accumulation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import sys
import os

# Handle autocast for MPS (PyTorch 2.0+)
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.boc_vit import BoCViT
from losses.metric_losses import ProxyNCA
from channel_adaptive_pipeline.chammi_grouped_dataloader import (
    create_grouped_chammi_dataloaders,
    create_dataset_ordered_training_iterator,
)


def train_epoch(
    model: nn.Module,
    iterator,  # Can be DataLoader or iterator
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    scaler: Optional[GradScaler] = None,
    label_to_idx: Optional[Dict] = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    batch_idx = 0
    print("Starting to iterate over training batches...")
    import sys
    sys.stdout.flush()  # Force flush output
    
    for batch_data in tqdm(iterator, desc="Training", file=sys.stdout):
        # Handle different iterator formats
        if len(batch_data) == 4:
            # Dataset-ordered iterator: (images, metadatas, labels, dataset_source)
            images, metadatas, labels, dataset_source = batch_data
        elif len(batch_data) == 3:
            # Standard DataLoader: (images, metadatas, labels)
            images, metadatas, labels = batch_data
        else:
            raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
        
        images = images.to(device, non_blocking=True)
        
        # Convert labels to tensor if needed
        if isinstance(labels, list):
            # Encode string labels to integers using label_to_idx mapping
            if label_to_idx is not None:
                # Handle None labels
                encoded_labels = []
                for label in labels:
                    if label is None:
                        encoded_labels.append(0)  # Default to class 0
                    elif isinstance(label, dict):
                        # If label is dict, use the first value
                        label = list(label.values())[0] if label else None
                        encoded_labels.append(label_to_idx.get(str(label), 0))
                    else:
                        encoded_labels.append(label_to_idx.get(str(label), 0))
                labels = torch.tensor(encoded_labels, dtype=torch.long, device=device)
            else:
                # Fallback: create mapping on the fly (not ideal but works)
                unique_labels = sorted(set(str(l) if l is not None else "None" for l in labels))
                temp_label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                labels = torch.tensor([temp_label_to_idx.get(str(l) if l is not None else "None", 0) for l in labels], dtype=torch.long, device=device)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        else:
            labels = labels.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        # MPS doesn't support float16 autocast, so disable it for MPS
        if device.type == "mps":
            # MPS: no autocast (MPS handles precision internally)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps
        else:
            # CPU/CUDA: use autocast
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / grad_accum_steps
        
        # Backward pass
        if device.type != "mps" and use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (batch_idx + 1) % grad_accum_steps == 0:
            if device.type != "mps" and use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps  # Unscale for logging
        num_batches += 1
        batch_idx += 1
        
        # Debug: print first batch info
        if batch_idx == 1:
            print(f"\n✓ First batch processed: images={images.shape}, labels={labels.shape if isinstance(labels, torch.Tensor) else len(labels)}")
            sys.stdout.flush()
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    label_to_idx: Optional[Dict] = None,
) -> tuple[float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, metadatas, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            
            # Convert labels to tensor if needed
            if isinstance(labels, list):
                if label_to_idx is not None:
                    encoded_labels = []
                    for label in labels:
                        if label is None:
                            encoded_labels.append(0)
                        elif isinstance(label, dict):
                            label = list(label.values())[0] if label else None
                            encoded_labels.append(label_to_idx.get(str(label), 0))
                        else:
                            encoded_labels.append(label_to_idx.get(str(label), 0))
                    labels = torch.tensor(encoded_labels, dtype=torch.long, device=device)
                else:
                    labels = torch.tensor([0] * len(labels), dtype=torch.long, device=device)
            elif not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long, device=device)
            else:
                labels = labels.to(device, non_blocking=True)
            
            # MPS doesn't support float16 autocast
            if device.type == "mps":
                # MPS: no autocast
                outputs = model(images)
                loss = criterion(outputs, labels)
            else:
                # CPU/CUDA: use autocast
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Compute accuracy (for CE mode)
            if outputs.dim() > 1 and outputs.shape[1] > 1:  # Logits
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train supervised BoC-ViT on CHAMMI")
    
    # Data arguments
    parser.add_argument("--csv-file", type=str, required=True, help="Path to combined_metadata.csv")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to CHAMMI root directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per channel group")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--target-labels", type=str, default="Label", help="Label column name")
    
    # Model arguments
    parser.add_argument("--img-size", type=int, default=128, help="Image size")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
    parser.add_argument("--embed-dim", type=int, default=192, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=6, help="Number of transformer blocks")
    parser.add_argument("--num-heads", type=int, default=3, help="Number of attention heads")
    parser.add_argument("--aggregator-mode", type=str, default="mean", choices=["mean", "attn"], help="Aggregation mode")
    parser.add_argument("--head-mode", type=str, default="ce", choices=["ce", "proxynca"], help="Head mode")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (required for CE)")
    parser.add_argument("--metric-embed-dim", type=int, default=256, help="Embedding dimension for metric learning")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use mixed precision")
    
    # Loss arguments (for ProxyNCA)
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for ProxyNCA")
    
    # Other arguments
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate")
    
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    
    # Determine number of classes if not provided
    if args.num_classes is None:
        if args.head_mode == "ce":
            # Try to infer from dataset
            from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset
            temp_dataset = CHAMMIDataset(
                csv_file=args.csv_file,
                root_dir=args.root_dir,
                target_labels=args.target_labels,
                transform=None,
                split="train",
            )
            # Get unique labels
            unique_labels = set()
            for i in range(min(1000, len(temp_dataset))):  # Sample to infer
                _, _, label = temp_dataset[i]
                if isinstance(label, (list, tuple)):
                    unique_labels.update(label)
                else:
                    unique_labels.add(label)
            args.num_classes = len(unique_labels)
            print(f"Inferred num_classes: {args.num_classes}")
        else:
            args.num_classes = 10  # Default for metric learning (not used)
    
    # Create model
    model = BoCViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        aggregator_mode=args.aggregator_mode,
        head_mode=args.head_mode,
        num_classes=args.num_classes,
        metric_embed_dim=args.metric_embed_dim,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    if args.head_mode == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.head_mode == "proxynca":
        criterion = ProxyNCA(
            embed_dim=args.metric_embed_dim,
            num_classes=args.num_classes,
            temperature=args.temperature,
        ).to(device)
    else:
        raise ValueError(f"Unknown head_mode: {args.head_mode}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler (cosine with warmup)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + torch.cos(torch.tensor((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * 3.14159)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler (MPS doesn't support GradScaler, so disable)
    scaler = GradScaler() if args.use_amp and device.type != "mps" else None
    if device.type == "mps" and args.use_amp:
        print("Note: Mixed precision autocast disabled for MPS (not supported)")
        args.use_amp = False  # Disable autocast for MPS
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch}")
    
    # Create label encoder (map string labels to integers)
    print("Creating label encoder...")
    from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset
    temp_dataset = CHAMMIDataset(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        target_labels=args.target_labels,
        transform=None,
        split="train",
    )
    # Collect all unique labels - sample more to get all labels
    all_labels = set()
    sample_size = min(50000, len(temp_dataset))  # Sample more to get all labels
    print(f"  Sampling {sample_size} samples to collect labels...")
    for i in range(sample_size):
        try:
            _, _, label = temp_dataset[i]
            if label is not None:
                if isinstance(label, dict):
                    # If label is dict, use the first value
                    label = list(label.values())[0] if label else None
                if label is not None:
                    all_labels.add(str(label))
        except Exception as e:
            # Skip problematic samples
            continue
    
    # Create label to index mapping
    unique_labels = sorted(list(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Found {len(unique_labels)} unique labels")
    print(f"Labels: {unique_labels[:10]}..." if len(unique_labels) > 10 else f"Labels: {unique_labels}")
    
    # Update num_classes if not set correctly
    if args.num_classes != len(unique_labels):
        print(f"Updating num_classes from {args.num_classes} to {len(unique_labels)}")
        args.num_classes = len(unique_labels)
        # Recreate model with correct num_classes
        model = BoCViT(
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            aggregator_mode=args.aggregator_mode,
            head_mode=args.head_mode,
            num_classes=args.num_classes,
            metric_embed_dim=args.metric_embed_dim,
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Recreate loss function
        if args.head_mode == "proxynca":
            criterion = ProxyNCA(
                embed_dim=args.metric_embed_dim,
                num_classes=args.num_classes,
                temperature=args.temperature,
            ).to(device)
        
        # Recreate optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create data loaders
    print("Creating data loaders...")
    
    # Training: use dataset-ordered iterator (randomly interleaved batches)
    train_iterator = create_dataset_ordered_training_iterator(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        shuffle=True,
        target_labels=args.target_labels,
        split="train",
        resize_to=args.img_size,
        augment=True,
        normalize=True,
        num_workers=args.num_workers,
        shuffle_dataset_order=True,
    )
    
    # Validation: use grouped loaders
    val_loaders = create_grouped_chammi_dataloaders(
        csv_file=args.csv_file,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        shuffle=False,
        target_labels=args.target_labels,
        split="test",  # Use test split for validation
        resize_to=args.img_size,
        augment=False,
        normalize=True,
        num_workers=args.num_workers,
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    if not args.eval_only:
        print("Starting training...")
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Create new iterator for this epoch (random interleaving)
            # This creates fresh dataloaders each epoch for proper shuffling
            print("Creating training iterator for this epoch...")
            train_iterator = create_dataset_ordered_training_iterator(
                csv_file=args.csv_file,
                root_dir=args.root_dir,
                batch_size=args.batch_size,
                shuffle=True,
                target_labels=args.target_labels,
                split="train",
                resize_to=args.img_size,
                augment=True,
                normalize=True,
                num_workers=args.num_workers,
                shuffle_dataset_order=True,
            )
            print("Iterator created, starting training...")
            
            # Train
            train_loss = train_epoch(
                model=model,
                iterator=train_iterator,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                use_amp=args.use_amp,
                grad_accum_steps=args.grad_accum_steps,
                scaler=scaler,
                label_to_idx=label_to_idx,
            )
            
            # Validate (on first channel group for simplicity)
            val_loss, val_acc = validate(
                model=model,
                dataloader=val_loaders[3],  # Use 3-channel loader
                criterion=criterion,
                device=device,
                use_amp=args.use_amp,
                label_to_idx=label_to_idx,
            )
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}")
            
            # Save checkpoint
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, output_dir / "checkpoint_latest.pth")
    
    # Final evaluation
    print("\nFinal evaluation...")
    for channel_count, val_loader in val_loaders.items():
        val_loss, val_acc = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=args.use_amp,
            label_to_idx=label_to_idx,
        )
        print(f"Channel {channel_count}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


if __name__ == "__main__":
    main()

