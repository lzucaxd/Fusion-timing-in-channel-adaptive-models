"""
Test forward pass of BoC-ViT with actual CHAMMI data.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.boc_vit import BoCViT
from losses.metric_losses import ProxyNCA
from channel_adaptive_pipeline.chammi_grouped_dataloader import create_dataset_ordered_training_iterator

def test_forward_pass():
    """Test forward pass with actual CHAMMI data."""
    print("=" * 70)
    print("Testing Forward Pass with CHAMMI Data")
    print("=" * 70)
    
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("\n1. Creating model...")
    model = BoCViT(
        img_size=128,
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        aggregator_mode="mean",
        head_mode="proxynca",
        num_classes=2,  # Will update after seeing data
        metric_embed_dim=256,
    ).to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create iterator
    print("\n2. Creating data iterator...")
    csv_file = "/Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv"
    root_dir = "/Users/zamfiraluca/Downloads/CHAMMI"
    
    iterator = create_dataset_ordered_training_iterator(
        csv_file=csv_file,
        root_dir=root_dir,
        batch_size=4,
        shuffle=True,
        target_labels='Label',
        split='train',
        resize_to=128,
        augment=False,  # No augmentation for testing
        normalize=True,
        num_workers=0,
        shuffle_dataset_order=True,
    )
    print("   Iterator created")
    
    # Get first batch
    print("\n3. Getting first batch...")
    try:
        batch_data = next(iter(iterator))
        print(f"   Got batch! Length: {len(batch_data)}")
        
        if len(batch_data) == 4:
            images, metadatas, labels, dataset_source = batch_data
            print(f"   Images shape: {images.shape}")
            print(f"   Labels type: {type(labels)}, length: {len(labels)}")
            print(f"   Labels: {labels}")
            print(f"   Dataset: {dataset_source}")
        else:
            print(f"   Unexpected batch format: {len(batch_data)} elements")
            return
    except Exception as e:
        print(f"   Error getting batch: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create label encoder
    print("\n4. Creating label encoder...")
    unique_labels = sorted(set(str(l) if l is not None else "None" for l in labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"   Unique labels: {unique_labels}")
    print(f"   Label mapping: {label_to_idx}")
    
    # Encode labels
    print("\n5. Encoding labels...")
    encoded_labels = []
    for label in labels:
        if label is None:
            encoded_labels.append(0)
        else:
            encoded_labels.append(label_to_idx.get(str(label), 0))
    labels_tensor = torch.tensor(encoded_labels, dtype=torch.long, device=device)
    print(f"   Encoded labels: {labels_tensor}")
    
    # Move images to device
    print("\n6. Moving images to device...")
    images = images.to(device)
    print(f"   Images on device: {images.device}")
    
    # Forward pass
    print("\n7. Running forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(images)
            print(f"   ✓ Forward pass successful!")
            print(f"   Output shape: {outputs.shape}")
            print(f"   Output dtype: {outputs.dtype}")
            print(f"   Output min/max: {outputs.min().item():.4f} / {outputs.max().item():.4f}")
            
            # Check normalization
            norms = torch.norm(outputs, dim=1)
            print(f"   Embedding norms: {norms}")
            print(f"   Are normalized (close to 1.0): {torch.allclose(norms, torch.ones_like(norms), atol=1e-3)}")
        except Exception as e:
            print(f"   ✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Test loss
    print("\n8. Testing loss computation...")
    try:
        # Update model with correct num_classes
        num_classes = len(unique_labels)
        print(f"   Creating loss with {num_classes} classes...")
        
        # Recreate model with correct num_classes if needed
        if num_classes != 2:
            print(f"   Updating model num_classes from 2 to {num_classes}")
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
            outputs = model(images)
        
        criterion = ProxyNCA(
            embed_dim=256,
            num_classes=num_classes,
            temperature=0.05,
        ).to(device)
        
        loss = criterion(outputs, labels_tensor)
        print(f"   ✓ Loss computation successful!")
        print(f"   Loss value: {loss.item():.4f}")
    except Exception as e:
        print(f"   ✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test training mode
    print("\n9. Testing training mode forward/backward...")
    try:
        model.train()
        outputs = model(images)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        print(f"   ✓ Backward pass successful!")
        print(f"   Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ✗ Training mode failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED - Forward pass works!")
    print("=" * 70)

if __name__ == "__main__":
    test_forward_pass()

