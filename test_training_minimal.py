"""
Minimal training test - prints everything immediately.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.boc_vit import BoCViT
from losses.metric_losses import ProxyNCA
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_dataset_ordered_training_iterator

def main():
    print("=" * 70)
    print("MINIMAL TRAINING TEST")
    print("=" * 70)
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Create model
    print("\nCreating model...")
    model = BoCViT(
        img_size=128,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        aggregator_mode="mean",
        head_mode="proxynca",
        num_classes=20,  # Will be updated
        metric_embed_dim=256,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create label encoder
    print("\nCreating label encoder...")
    from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset
    temp_dataset = CHAMMIDataset(
        csv_file="/Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv",
        root_dir="/Users/zamfiraluca/Downloads/CHAMMI",
        target_labels='Label',
        transform=None,
        split="train",
    )
    all_labels = set()
    for i in range(min(1000, len(temp_dataset))):
        _, _, label = temp_dataset[i]
        if label is not None:
            if isinstance(label, dict):
                label = list(label.values())[0] if label else None
            if label is not None:
                all_labels.add(str(label))
    
    unique_labels = sorted(list(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Found {num_classes} unique labels: {unique_labels[:5]}...")
    
    # Update model
    if num_classes != 20:
        print(f"Updating model num_classes to {num_classes}")
        model = BoCViT(
            img_size=128,
            patch_size=16,
            embed_dim=192,
            depth=6,
            num_heads=3,
            aggregator_mode="mean",
            head_mode="proxynca",
            num_classes=num_classes,
            metric_embed_dim=256,
        ).to(device)
    
    # Create loss
    print("\nCreating loss...")
    criterion = ProxyNCA(
        embed_dim=256,
        num_classes=num_classes,
        temperature=0.05,
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create iterator
    print("\nCreating training iterator...")
    iterator = create_dataset_ordered_training_iterator(
        csv_file="/Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv",
        root_dir="/Users/zamfiraluca/Downloads/CHAMMI",
        batch_size=4,
        shuffle=True,
        target_labels='Label',
        split='train',
        resize_to=128,
        augment=False,
        normalize=True,
        num_workers=0,
        shuffle_dataset_order=True,
    )
    print("Iterator created")
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    print("\nIterating over batches...")
    for batch_idx, batch_data in enumerate(iterator):
        print(f"\nBatch {batch_idx + 1}:")
        
        # Unpack batch
        if len(batch_data) == 4:
            images, metadatas, labels, dataset_source = batch_data
        else:
            print(f"Unexpected batch format: {len(batch_data)} elements")
            break
        
        print(f"  Images: {images.shape}, Dataset: {dataset_source}")
        print(f"  Labels: {labels[:2]}...")
        
        # Move to device
        images = images.to(device)
        
        # Encode labels
        encoded_labels = []
        for label in labels:
            if label is None:
                encoded_labels.append(0)
            elif isinstance(label, dict):
                label = list(label.values())[0] if label else None
                encoded_labels.append(label_to_idx.get(str(label), 0))
            else:
                encoded_labels.append(label_to_idx.get(str(label), 0))
        labels_tensor = torch.tensor(encoded_labels, dtype=torch.long, device=device)
        print(f"  Encoded labels: {labels_tensor.tolist()}")
        
        # Forward pass
        print("  Running forward pass...")
        outputs = model(images)
        print(f"  Outputs: {outputs.shape}")
        
        # Loss
        print("  Computing loss...")
        loss = criterion(outputs, labels_tensor)
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward
        print("  Running backward pass...")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("  ✓ Step complete")
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx >= 4:  # Test 5 batches
            break
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"\n" + "=" * 70)
    print(f"Training test complete!")
    print(f"Processed {num_batches} batches")
    print(f"Average loss: {avg_loss:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()

