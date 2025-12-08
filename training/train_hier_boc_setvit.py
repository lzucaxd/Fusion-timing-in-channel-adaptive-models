"""
Training script for HierBoCSetViT: One DataLoader per dataset, randomly sample batches.
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

from models.hier_boc_setvit import HierBoCSetViT
from losses.metric_losses import ProxyNCA
from channel_adaptive_pipeline.chammi_grouped_dataloader import (
    create_dataset_specific_dataloaders,
    create_random_dataset_interleaved_iterator,
)


def main():
    parser = argparse.ArgumentParser(description="Train HierBoCSetViT on CHAMMI")
    
    # Data
    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-labels", type=str, default="Label")
    
    # Model
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=None, nargs='?', const=None, help="Auto-detected from timm ViT if None")
    parser.add_argument("--encoder-type", type=str, default="tiny", choices=["tiny", "small"], help="ViT encoder type: tiny (192 dim) or small (384 dim)")
    parser.add_argument("--encoder-pretrained", action="store_true", default=True, help="Use pretrained timm ViT")
    parser.add_argument("--aggregator-depth", type=int, default=2, help="Depth of Set Transformer aggregator")
    parser.add_argument("--aggregator-num-heads", type=int, default=3, help="Number of heads in Set Transformer")
    parser.add_argument("--head-mode", type=str, default="proxynca", choices=["ce", "proxynca"])
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--metric-embed-dim", type=int, default=96, help="Embedding dim for ProxyNCA (default: embed_dim // 2)")
    parser.add_argument("--channel-dropout-p", type=float, default=0.3, help="Channel dropout probability during training")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.05)
    
    # Other
    parser.add_argument("--output-dir", type=str, default="./checkpoints/hier_boc_setvit")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create label encoder - get ALL unique labels across ALL datasets
    print("\nCreating label encoder...")
    from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset
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
    print(f"Labels: {unique_labels[:10]}..." if len(unique_labels) > 10 else f"Labels: {unique_labels}")
    
    if args.num_classes != num_classes:
        print(f"Updating num_classes from {args.num_classes} to {num_classes}")
        args.num_classes = num_classes
    
    # Create model
    print(f"\nCreating HierBoCSetViT model...")
    print(f"  Encoder type: {args.encoder_type}")
    print(f"  Image size: {args.img_size}x{args.img_size}")
    model = HierBoCSetViT(
        img_size=args.img_size,
        embed_dim=args.embed_dim,  # Will auto-detect from timm if None
        encoder_type=args.encoder_type,
        encoder_pretrained=args.encoder_pretrained,
        aggregator_depth=args.aggregator_depth,
        aggregator_num_heads=args.aggregator_num_heads,
        head_mode=args.head_mode,
        num_classes=args.num_classes,
        metric_embed_dim=args.metric_embed_dim,  # Will default to embed_dim // 2 if None
        channel_dropout_p=args.channel_dropout_p,
    ).to(device)
    
    # Get actual embed_dim from model (might be auto-detected from timm)
    actual_embed_dim = model.embed_dim
    
    # Get actual metric_embed_dim from model head if proxynca
    if args.head_mode == "proxynca":
        if hasattr(model, 'metric_embed_dim'):
            actual_metric_embed_dim = model.metric_embed_dim
        elif hasattr(model.head, '1'):  # Sequential with LayerNorm
            actual_metric_embed_dim = model.head[0].out_features
        else:
            actual_metric_embed_dim = args.metric_embed_dim or (actual_embed_dim // 2)
        args.metric_embed_dim = actual_metric_embed_dim
    
    print(f"Model embed_dim: {actual_embed_dim}")
    if args.head_mode == "proxynca":
        print(f"Model metric_embed_dim: {args.metric_embed_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss
    if args.head_mode == "proxynca":
        criterion = ProxyNCA(
            embed_dim=args.metric_embed_dim,
            num_classes=args.num_classes,
            temperature=args.temperature,
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # LR scheduler
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 0.5 * (1 + torch.cos(torch.tensor((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * 3.14159)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch}")
    
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Create random interleaved iterator for this epoch
        # This will randomly sample batches from all datasets
        iterator = create_random_dataset_interleaved_iterator(
            dataloaders,
            random_seed=None,  # Different random order each epoch
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
            
            # Encode labels - convert string labels to integer indices
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
                            encoded_labels.append(0)  # Default to 0
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
            "num_classes": args.num_classes,
            "embed_dim": actual_embed_dim,
            "metric_embed_dim": args.metric_embed_dim if args.head_mode == "proxynca" else None,
        }
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, output_dir / "checkpoint_latest.pth")
        print(f"  Saved checkpoint to {output_dir}")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

