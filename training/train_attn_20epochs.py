"""
Train BoC-ViT with attention pooling for 20 epochs.
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
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.boc_vit import BoCViT
from losses.metric_losses import ProxyNCA
from channel_adaptive_pipeline.chammi_grouped_dataloader import (
    create_dataset_specific_dataloaders,
    create_random_dataset_interleaved_iterator,
)


def main():
    parser = argparse.ArgumentParser(description="Train BoC-ViT with attention pooling")
    
    # Data
    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-labels", type=str, default="Label")
    
    # Model
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--metric-embed-dim", type=int, default=256)
    
    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.05)
    
    # Other
    parser.add_argument("--output-dir", type=str, default="./checkpoints/boc_attn_proxynca")
    parser.add_argument("--num-workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create label encoder - get ALL unique labels across ALL datasets
    print("\nCreating label encoder...")
    import pandas as pd
    
    # Get all unique labels from enriched metadata files directly
    all_labels = set()
    for dataset in ['Allen', 'HPA', 'CP']:
        meta_file = os.path.join(args.root_dir, dataset, 'enriched_meta.csv')
        if os.path.exists(meta_file):
            try:
                df = pd.read_csv(meta_file, low_memory=False)
                if args.target_labels in df.columns:
                    labels = df[args.target_labels].dropna().unique()
                    all_labels.update([str(l) for l in labels])
                    print(f"  {dataset}: {len(labels)} labels")
            except Exception as e:
                print(f"  {dataset}: Error reading - {e}")
    
    unique_labels = sorted(list(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"\nTotal unique labels across all datasets: {num_classes}")
    
    if args.num_classes != num_classes:
        print(f"Updating num_classes from {args.num_classes} to {num_classes}")
        args.num_classes = num_classes
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_path = output_dir / "checkpoint_latest.pth"
    start_epoch = 0
    
    # Create model with ATTENTION pooling
    print(f"\nCreating model with ATTENTION pooling...")
    model = BoCViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        aggregator_mode="attn",  # ATTENTION POOLING
        head_mode="proxynca",
        num_classes=args.num_classes,
        metric_embed_dim=args.metric_embed_dim,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss
    criterion = ProxyNCA(
        embed_dim=args.metric_embed_dim,
        num_classes=args.num_classes,
        temperature=args.temperature,
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # LR scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + torch.cos(torch.tensor((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * 3.14159)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Load checkpoint if exists
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        if 'label_to_idx' in checkpoint:
            # Update label_to_idx if saved in checkpoint
            label_to_idx = checkpoint['label_to_idx']
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("\nNo checkpoint found, starting from scratch")
    
    # Create DataLoaders - ONE PER DATASET
    print(f"\nCreating DataLoaders (one per dataset)...")
    dataloaders = create_dataset_specific_dataloaders(
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
    )
    
    dataset_names = list(dataloaders.keys())
    print(f"Created DataLoaders for: {dataset_names}")
    for name, dl in dataloaders.items():
        print(f"  {name}: {len(dl)} batches")
    
    # Training loop
    print(f"\n{'='*70}")
    if start_epoch > 0:
        print(f"RESUMING TRAINING - ATTENTION POOLING (from epoch {start_epoch})")
    else:
        print("STARTING TRAINING - ATTENTION POOLING")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Create random interleaved iterator for this epoch
        iterator = create_random_dataset_interleaved_iterator(
            dataloaders,
            random_seed=None,
        )
        
        # Count total batches
        total_batches = sum(len(dl) for dl in dataloaders.values())
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}")
        
        # Iterate over randomly interleaved batches
        for images, metadatas, labels, dataset_source in iterator:
            # Move to device
            images = images.to(device)
            
            # Encode labels
            if isinstance(labels, list):
                encoded_labels = []
                for label in labels:
                    if label is None:
                        encoded_labels.append(0)
                    elif isinstance(label, dict):
                        label = list(label.values())[0] if label else None
                        encoded_labels.append(label_to_idx.get(str(label), 0))
                    else:
                        label_str = str(label)
                        if label_str in label_to_idx:
                            encoded_labels.append(label_to_idx[label_str])
                        else:
                            encoded_labels.append(0)
                labels_tensor = torch.tensor(encoded_labels, dtype=torch.long, device=device)
            else:
                labels_tensor = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.update(1)
            if num_batches > 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dataset': dataset_source,
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
        
        pbar.close()
        
        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"  Average loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_loss,
            "label_to_idx": label_to_idx,  # Save label mapping
        }
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, output_dir / "checkpoint_latest.pth")
        print(f"  Saved checkpoint")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE - ATTENTION POOLING")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

