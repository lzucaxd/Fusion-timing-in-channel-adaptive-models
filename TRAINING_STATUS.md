# Training Status and Evaluation Plan

## Current Status

### Mean Pooling Model
- **Location**: `checkpoints/boc_mean_proxynca/`
- **Current Epoch**: 9 (as of latest checkpoint)
- **Target**: 20 epochs
- **Status**: Training should continue to reach 20 epochs

### Attention Pooling Model  
- **Location**: `checkpoints/boc_attn_proxynca/`
- **Current Status**: Training in progress (Epoch 1)
- **Target**: 20 epochs
- **Training Script**: `training/train_attn_20epochs.py`

## Evaluation Script

Once both models reach 20 epochs, run:

```bash
python training/evaluate_latest_checkpoints.py \
    --csv-file /Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv \
    --root-dir /Users/zamfiraluca/Downloads/CHAMMI \
    --batch-size 32 \
    --num-workers 4 \
    --img-size 128 \
    --patch-size 16 \
    --embed-dim 192 \
    --depth 6 \
    --num-heads 3 \
    --num-classes 20 \
    --metric-embed-dim 256 \
    --mean-checkpoint ./checkpoints/boc_mean_proxynca/checkpoint_latest.pth \
    --attn-checkpoint ./checkpoints/boc_attn_proxynca/checkpoint_latest.pth
```

This will:
1. Load both model checkpoints
2. Extract embeddings from train/test sets
3. Evaluate with 1-NN classification
4. Compare results between mean and attention pooling

