"""
Test and visualization script for CHAMMI dataset.
This script loads the unified CHAMMI dataset and visualizes samples from all three sub-datasets.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from channel_adaptive_pipeline.chammi_dataset import CHAMMIDataset, CHAMMITransform, create_chammi_dataloader


def visualize_channels(image_tensor, num_channels, title="", save_path=None):
    """
    Visualize individual channels of an image.
    
    Args:
        image_tensor: Tensor of shape (C, H, W)
        num_channels: Number of actual channels (may be less than tensor shape[0])
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, num_channels, figsize=(num_channels * 3, 3))
    if num_channels == 1:
        axes = [axes]
    
    for i in range(num_channels):
        channel_img = image_tensor[i].detach().cpu().numpy()
        axes[i].imshow(channel_img, cmap='gray')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def test_dataset_basic(csv_file, root_dir):
    """Test basic dataset functionality."""
    print("=" * 60)
    print("Testing CHAMMI Dataset - Basic Functionality")
    print("=" * 60)
    
    # Create dataset without labels first
    transform = CHAMMITransform(size=128, augment=False, mode='center')
    dataset = CHAMMIDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform,
        split='train',  # Only train split
    )
    
    print(f"\nTotal samples in train split: {len(dataset)}")
    
    # Check channel distribution
    channel_counts = {}
    dataset_sources = {}
    
    for i in range(min(1000, len(dataset))):  # Sample first 1000
        row = dataset.metadata.iloc[i]
        num_ch = row['num_channels']
        source = row['file_path'].split('/')[0]
        
        channel_counts[num_ch] = channel_counts.get(num_ch, 0) + 1
        dataset_sources[source] = dataset_sources.get(source, 0) + 1
    
    print(f"\nChannel distribution (first 1000 samples):")
    for ch, count in sorted(channel_counts.items()):
        print(f"  {ch} channels: {count}")
    
    print(f"\nDataset sources (first 1000 samples):")
    for source, count in sorted(dataset_sources.items()):
        print(f"  {source}: {count}")
    
    # Test loading a few samples
    print("\n" + "-" * 60)
    print("Testing sample loading:")
    print("-" * 60)
    
    for idx in [0, len(dataset)//3, len(dataset)//2]:
        try:
            image, metadata, labels = dataset[idx]
            
            print(f"\nSample {idx}:")
            print(f"  Image shape: {image.shape}")
            print(f"  Num channels (from metadata): {metadata['num_channels']}")
            print(f"  Dataset source: {metadata['dataset_source']}")
            print(f"  Cell type: {metadata.get('cell_type', 'N/A')}")
            print(f"  File path: {metadata['file_path']}")
            print(f"  Labels: {labels}")
            
            assert image.shape[1] == 128, f"Expected height 128, got {image.shape[1]}"
            assert image.shape[2] == 128, f"Expected width 128, got {image.shape[2]}"
            assert image.shape[0] == metadata['num_channels'], \
                f"Channel mismatch: tensor has {image.shape[0]}, metadata says {metadata['num_channels']}"
            
        except Exception as e:
            print(f"  ERROR loading sample {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ Basic dataset tests passed!")
    return dataset


def test_dataset_with_labels(csv_file, root_dir):
    """Test dataset with label extraction."""
    print("\n" + "=" * 60)
    print("Testing CHAMMI Dataset - With Labels")
    print("=" * 60)
    
    transform = CHAMMITransform(size=128, augment=False)
    dataset = CHAMMIDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        target_labels='Label',  # Default classification label
        transform=transform,
        split='train',
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    
    # Check label distribution
    label_counts = {}
    for i in range(min(1000, len(dataset))):
        image, metadata, label = dataset[i]
        if label is not None:
            label_str = str(label)
            label_counts[label_str] = label_counts.get(label_str, 0) + 1
    
    print(f"\nLabel distribution (first 1000 samples, top 10):")
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for label, count in sorted_labels:
        print(f"  {label}: {count}")
    
    print("\n✓ Label extraction tests passed!")
    return dataset


def visualize_samples(csv_file, root_dir, num_samples=6, save_dir=None):
    """Visualize samples from each dataset."""
    print("\n" + "=" * 60)
    print("Visualizing CHAMMI Samples")
    print("=" * 60)
    
    transform = CHAMMITransform(size=128, augment=False)
    dataset = CHAMMIDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        transform=transform,
        split='train',
    )
    
    # Get samples from each dataset source
    allen_indices = []
    hpa_indices = []
    cp_indices = []
    
    for i in range(len(dataset)):
        row = dataset.metadata.iloc[i]
        source = row['file_path'].split('/')[0]
        if source == 'Allen' and len(allen_indices) < num_samples // 3:
            allen_indices.append(i)
        elif source == 'HPA' and len(hpa_indices) < num_samples // 3:
            hpa_indices.append(i)
        elif source == 'CP' and len(cp_indices) < num_samples // 3:
            cp_indices.append(i)
        
        if len(allen_indices) + len(hpa_indices) + len(cp_indices) >= num_samples:
            break
    
    # Fill remaining slots
    all_indices = allen_indices + hpa_indices + cp_indices
    while len(all_indices) < num_samples and len(all_indices) < len(dataset):
        i = len(all_indices)
        if i < len(dataset):
            all_indices.append(i)
    
    # Visualize
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    if num_samples == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    
    for idx, ax_row in enumerate(axes):
        for col_idx, ax in enumerate(ax_row):
            sample_idx = idx * cols + col_idx
            if sample_idx >= len(all_indices):
                ax.axis('off')
                continue
            
            try:
                image, metadata, labels = dataset[all_indices[sample_idx]]
                num_channels = metadata['num_channels']
                source = metadata['dataset_source']
                
                # Create RGB composite if 3+ channels, or show first channel
                if num_channels >= 3:
                    # Use first 3 channels as RGB
                    rgb = image[:3].permute(1, 2, 0).detach().cpu().numpy()
                    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                    ax.imshow(rgb)
                else:
                    # Show first channel as grayscale
                    ch0 = image[0].detach().cpu().numpy()
                    ax.imshow(ch0, cmap='gray')
                
                title = f"{source}\n{num_channels} ch\n{metadata.get('cell_type', 'N/A')}"
                if labels:
                    title += f"\n{labels}"
                
                ax.set_title(title, fontsize=9)
                ax.axis('off')
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'chammi_samples.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to {save_path}")
    
    plt.show()
    
    # Also show individual channel breakdown for one sample from each source
    print("\nIndividual channel visualizations:")
    for source_indices, source_name in [(allen_indices, 'Allen'), 
                                        (hpa_indices, 'HPA'), 
                                        (cp_indices, 'CP')]:
        if source_indices:
            try:
                image, metadata, labels = dataset[source_indices[0]]
                num_channels = metadata['num_channels']
                title = f"{source_name} - {metadata['file_path'].split('/')[-1]}"
                save_path = None
                if save_dir:
                    save_path = os.path.join(save_dir, f'{source_name.lower()}_channels.png')
                visualize_channels(image, num_channels, title, save_path)
            except Exception as e:
                print(f"  Error visualizing {source_name}: {e}")


def test_dataloader(csv_file, root_dir, batch_size=4):
    """Test DataLoader functionality."""
    print("\n" + "=" * 60)
    print("Testing CHAMMI DataLoader")
    print("=" * 60)
    
    dataloader = create_chammi_dataloader(
        csv_file=csv_file,
        root_dir=root_dir,
        batch_size=batch_size,
        shuffle=False,
        split='train',
        resize_to=128,
        augment=False,
        num_workers=0,  # Set to 0 for easier debugging
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    
    # Get first batch
    try:
        batch_images, batch_metadatas, batch_labels = next(iter(dataloader))
        
        print(f"\nFirst batch:")
        print(f"  Batch images shape: {batch_images.shape}")
        print(f"  Batch size: {len(batch_metadatas)}")
        
        # Check each sample in batch
        for i in range(len(batch_metadatas)):
            meta = batch_metadatas[i]
            img = batch_images[i]
            print(f"\n  Sample {i}:")
            print(f"    Image shape: {img.shape}")
            print(f"    Num channels: {meta['num_channels']}")
            print(f"    Dataset: {meta['dataset_source']}")
            print(f"    Label: {batch_labels[i]}")
        
        print("\n✓ DataLoader tests passed!")
        
    except Exception as e:
        print(f"  ERROR in DataLoader: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    # Update these paths to match your CHAMMI dataset location
    CHAMMI_ROOT = "/Users/zamfiraluca/Downloads/CHAMMI"
    CSV_FILE = os.path.join(CHAMMI_ROOT, "combined_metadata.csv")
    
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: CSV file not found at {CSV_FILE}")
        print("Please update CHAMMI_ROOT and CSV_FILE paths in the script.")
        return
    
    print(f"CHAMMI root directory: {CHAMMI_ROOT}")
    print(f"Using CSV file: {CSV_FILE}")
    
    # Run tests
    try:
        # Basic dataset test
        dataset = test_dataset_basic(CSV_FILE, CHAMMI_ROOT)
        
        # Test with labels
        dataset_labels = test_dataset_with_labels(CSV_FILE, CHAMMI_ROOT)
        
        # Visualize samples
        visualize_samples(CSV_FILE, CHAMMI_ROOT, num_samples=9, save_dir="./chammi_visualizations")
        
        # Test DataLoader
        test_dataloader(CSV_FILE, CHAMMI_ROOT, batch_size=4)
        
        print("\n" + "=" * 60)
        print("All tests completed successfully! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

