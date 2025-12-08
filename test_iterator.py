"""
Test if the iterator actually yields batches.
"""

import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from channel_adaptive_pipeline.chammi_grouped_dataloader import create_dataset_ordered_training_iterator

def test_iterator():
    """Test if iterator yields batches."""
    print("=" * 70)
    print("Testing Iterator Batch Yielding")
    print("=" * 70)
    
    csv_file = "/Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv"
    root_dir = "/Users/zamfiraluca/Downloads/CHAMMI"
    
    print("\n1. Creating iterator...")
    iterator = create_dataset_ordered_training_iterator(
        csv_file=csv_file,
        root_dir=root_dir,
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
    print("   Iterator created")
    
    print("\n2. Testing iteration...")
    batch_count = 0
    start_time = time.time()
    
    try:
        for i, batch_data in enumerate(iterator):
            batch_count += 1
            if i == 0:
                elapsed = time.time() - start_time
                print(f"   First batch took {elapsed:.2f} seconds")
                images, metadatas, labels, dataset_source = batch_data
                print(f"   Batch {i+1}: images={images.shape}, dataset={dataset_source}")
            
            if i < 5:
                images, metadatas, labels, dataset_source = batch_data
                print(f"   Batch {i+1}: images={images.shape}, dataset={dataset_source}, labels={labels[:2]}...")
            
            if i >= 9:  # Test 10 batches
                break
        
        elapsed = time.time() - start_time
        print(f"\n   Processed {batch_count} batches in {elapsed:.2f} seconds")
        print(f"   Average time per batch: {elapsed/batch_count:.2f} seconds")
        print("   ✓ Iterator works!")
        
    except Exception as e:
        print(f"   ✗ Iterator failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("✓ Iterator test passed!")
    print("=" * 70)

if __name__ == "__main__":
    test_iterator()

