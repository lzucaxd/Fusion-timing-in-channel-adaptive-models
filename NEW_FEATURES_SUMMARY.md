# Summary of New Features and Improvements

This document summarizes all the new features, improvements, and changes added to the HierBoCSetViT model and training pipeline.

---

## 1. Model Architecture Enhancements (`models/hier_boc_setvit.py`)

### 1.1 Patch-Token Pooling for Per-Channel Embedding

**Feature**: Three modes for aggregating patch tokens into per-channel embeddings.

**Implementation**:
- **New parameter**: `channel_embed_mode: Literal["cls", "mean_patches", "attn_pool"]`
  - Default: `"attn_pool"` (attention pooling)
  - Replaces deprecated `use_cls_token` boolean (backward compatible)

**Modes**:
1. **`"cls"`**: Uses CLS token (original behavior with `use_cls_token=True`)
2. **`"mean_patches"`**: Mean pooling over patch tokens, excluding CLS (original behavior with `use_cls_token=False`)
3. **`"attn_pool"`**: **NEW** - Attention pooling over patch tokens with learnable query parameter
   - Learnable query: `self.patch_pool_query` with shape `(1, 1, D)`
   - Attention computed over patch tokens only (excludes CLS token)
   - Formula: `z = softmax(QK^T / √D) * V` where Q is learnable query, K/V are patch tokens

**Applied to**: `PerChannelEncoderTiny` and `PerChannelEncoderSmall`

---

### 1.2 Channel Gating (Optional)

**Feature**: Learnable gating mechanism applied to channel embeddings before aggregation.

**Parameters**:
- `use_channel_gating: bool = False` (default: disabled)
- `channel_gate_hidden_ratio: float = 0.25` (default)

**Architecture**:
```
Channel Gate MLP:
  LayerNorm(D) 
  → Linear(D → hidden_dim) 
  → GELU 
  → Linear(hidden_dim → 1) 
  → Sigmoid
```

**Initialization**:
- Final Linear bias: `+2.0` (sigmoid ~0.88, keeps most channels active initially)
- Final Linear weights: Normal with `std=0.01` (small initialization)

**Forward pass**:
```python
gates = torch.sigmoid(self.channel_gate(z_channels))  # (B, C, 1)
z_channels = z_channels * gates  # Element-wise gating
```

**Return values**: If `return_attn=True` and gating enabled, returns `(output, attn_weights, gates)`

---

### 1.3 Multi-Seed PMA (K Bag Queries)

**Feature**: Multiple learnable bag queries in Pooling-by-Multihead-Attention for richer aggregation.

**Parameter**: `pma_num_seeds: int = 1` (default: 1, matches previous single-query behavior)

**Implementation**:
- Replaced `self.bag_query` (shape: `(1, 1, D)`) with `self.bag_queries` (shape: `(1, K, D)`)
- Forward pass:
  ```python
  q = self.bag_queries.expand(B, K, D)  # (B, K, D)
  scores = q @ x.transpose(1, 2) / √D  # (B, K, C)
  attn = softmax(scores, dim=-1)  # (B, K, C)
  pooled = attn @ x  # (B, K, D)
  z_bag = pooled.mean(dim=1)  # (B, D) - average over K queries
  ```
- Attention weights (if `return_attn=True`): `attn.mean(dim=1)` → `(B, C)`

**Applied to**: `ChannelSetTransformer`

---

### 1.4 Performance Fix: Vectorized Channel Dropout

**Feature**: Replaced Python for-loop with vectorized masking for channel dropout.

**Location**: `HierBoCSetViT._apply_channel_permutation_and_dropout()`

**Old implementation**: Python loop iterating over batch samples

**New implementation**:
```python
mask = torch.ones(B, C, device=x.device, dtype=x.dtype)
drop_sample = (torch.rand(B, device=x.device) < self.channel_dropout_p)
drop_idx = torch.randint(0, C, (B,), device=x.device)
mask[torch.arange(B, device=x.device)[drop_sample], drop_idx[drop_sample]] = 0.0
x = x * mask[:, :, None, None]
```

**Benefit**: Significantly faster channel dropout operation, especially for large batches.

---

## 2. Training Loop Improvements (`training/train_hier_boc_setvit.py`)

### 2.1 ProxyNCA Proxy Optimization

**Feature**: ProxyNCA proxies are now optimized as a separate parameter group.

**Implementation**:
- Added proxy parameters (`criterion.parameters()`) to optimizer as third parameter group
- Proxy learning rate: Same as main model (`args.lr`)
- Proxy weight decay: `0.0` (proxies don't use weight decay)
- Logging: Shows proxy parameter count (must be >0 for ProxyNCA)

---

### 2.2 Two-Tier Learning Rates

**Feature**: Separate learning rates for encoder vs. non-encoder parameters.

**Parameter**: `encoder_lr_mult: float = 1.0` (default: 1.0, can be set to 0.2 for lower encoder LR)

**Implementation**:
- **Parameter Group 1**: Encoder parameters
  - Learning rate: `args.lr * args.encoder_lr_mult`
  - Weight decay: `args.weight_decay`
- **Parameter Group 2**: Non-encoder model parameters
  - Learning rate: `args.lr`
  - Weight decay: `args.weight_decay`
- **Parameter Group 3**: ProxyNCA proxies (if enabled)
  - Learning rate: `args.lr`
  - Weight decay: `0.0`

**Rationale**: Lower learning rate for pretrained encoder prevents catastrophic forgetting while allowing fine-tuning.

---

### 2.3 Encoder Freeze/Unfreeze Schedule

**Feature**: Option to freeze encoder for initial epochs, then unfreeze.

**Parameter**: `freeze_encoder_epochs: int = 0` (default: 0, no freezing)

**Behavior**:
- Epochs 1 to `freeze_encoder_epochs`: Encoder parameters `requires_grad = False`
- Epoch `freeze_encoder_epochs + 1` and onwards: Encoder parameters `requires_grad = True`
- Logging: Shows when encoder is frozen/unfrozen

---

### 2.4 Hard Label Encoding with Invalid Label Filtering

**Feature**: Robust label encoding that filters out invalid labels (None, unknown labels).

**Implementation**:
- Validates each label before encoding
- Handles multiple label formats:
  - Direct string/numeric labels
  - Dictionary labels (extracts first value)
  - None labels (filtered out)
- Creates valid mask and filters batch accordingly
- Skips batches if all labels are invalid
- Logging: Counts of `num_none_labels` and `num_unknown_labels` per epoch

**Benefit**: Prevents training crashes from invalid labels, improves robustness.

---

### 2.5 Dataset-Balanced Sampling (Random Sampling with Full Coverage)

**Feature**: Randomly samples from available datasets while ensuring all batches are processed each epoch.

**Implementation**:
- Tracks batches processed per dataset
- Randomly selects from datasets that still have unprocessed batches
- Continues until all batches from all datasets are processed
- Progress bar shows current dataset and batch progress

**Key difference from previous**:
- **Previous**: Fixed `steps_per_epoch` with random dataset selection (could miss batches)
- **New**: Processes ALL batches from ALL datasets each epoch (no data loss)

**Benefits**:
- Ensures full data coverage each epoch
- Maintains label diversity through random sampling
- Balances dataset contributions naturally

---

### 2.6 Gradient Clipping

**Feature**: Gradient clipping to prevent exploding gradients.

**Parameter**: `grad_clip_norm: float = 0.0` (default: 0.0, disabled; set to 1.0 to enable)

**Implementation**:
- Clips gradients using `torch.nn.utils.clip_grad_norm_()`
- Applied to model parameters
- Also applied to ProxyNCA proxy parameters if ProxyNCA is used
- Only active when `grad_clip_norm > 0`

---

### 2.7 Improved LR Scheduler

**Feature**: Pure Python implementation of cosine decay LR scheduler (no torch tensor dependencies).

**Parameters**:
- `warmup_epochs: int = 0` (default: 0, no warmup)
- `epochs: int` (total training epochs)

**Schedule**:
- **Warmup phase** (epochs 0 to `warmup_epochs`): Linear warmup `lr = base_lr * (epoch + 1) / warmup_epochs`
- **Cosine decay phase** (epochs `warmup_epochs` to `epochs`): 
  ```
  progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
  lr = base_lr * 0.5 * (1 + cos(π * progress))
  ```

**Implementation**: Uses Python `math.cos()` and `math.pi` (no torch tensors in lambda function).

---

### 2.8 Reproducibility

**Feature**: Random seed control for reproducible training.

**Parameter**: `seed: int = None` (optional)

**Implementation**:
- Sets `random.seed()` for Python random
- Sets `torch.manual_seed()` for PyTorch
- Sets `np.random.seed()` if NumPy available
- Sets `torch.cuda.manual_seed_all()` if CUDA available
- Logs seed value at training start

---

### 2.9 Enhanced Logging

**New logging includes**:
- Model architecture details (encoder type, embed_dim, channel_embed_mode, pma_num_seeds)
- Parameter group details (LRs, weight decays)
- Criterion parameter counts (for ProxyNCA)
- Invalid label counts per epoch
- Current dataset in progress bar
- Per-parameter-group learning rates at end of each epoch

---

## 3. Evaluation Script Improvements (`training/evaluate_hier_boc.py`)

### 3.1 Auto-Detection of Model Configuration

**Feature**: Automatically detects model configuration from checkpoint to prevent size mismatches.

**Detected parameters**:
- `embed_dim`: From encoder state dict keys
- `encoder_type`: From encoder architecture (tiny vs. small)
- `metric_embed_dim`: From head layer weights
- `channel_embed_mode`: From checkpoint metadata (if available)

**Benefit**: Eliminates manual parameter specification errors during evaluation.

---

### 3.2 Support for New Model Parameters

**Added CLI arguments**:
- `--channel-embed-mode`: Specify channel embedding mode (default: `"attn_pool"`)
- `--pma-num-seeds`: Specify number of PMA seeds (default: 1)

**Model initialization**: Now includes `channel_embed_mode` and `pma_num_seeds` in model creation.

---

### 3.3 Enhanced Visualizations

**Feature**: Visualizations now show actual class names instead of generic "Class X" labels.

**Implementation**: Uses `label_to_name_map` to decode integer labels back to original string names.

**Applied to**: UMAP plots, t-SNE plots, attention heatmaps, attention distributions, attention examples.

---

## 4. New Training Scripts

### 4.1 `start_training_hier_boc_non_balanced.sh`

**Purpose**: Training script with random dataset sampling that processes all batches each epoch.

**Key configuration**:
- Random sampling from datasets
- Full data coverage per epoch
- All improvements from improved training loop
- Output: `checkpoints/hier_boc_setvit_tiny_non_balanced/`

---

## 5. Default Values Summary

### Model Defaults (HierBoCSetViT):
- `channel_embed_mode`: `"attn_pool"` (changed from `"cls"`)
- `pma_num_seeds`: `1` (can be set to 4 for multi-seed)
- `use_channel_gating`: `False` (disabled by default)
- `channel_gate_hidden_ratio`: `0.25`

### Training Defaults:
- `encoder_lr_mult`: `1.0` (can be set to 0.2 for pretrained encoder)
- `freeze_encoder_epochs`: `0` (no freezing)
- `grad_clip_norm`: `0.0` (disabled, can be set to 1.0)
- `warmup_epochs`: `0` (no warmup, can be set to 3)
- `seed`: `None` (random, can be set to 42 for reproducibility)

---

## 6. Backward Compatibility

**All changes maintain backward compatibility**:
- Legacy `use_cls_token` parameter still accepted and mapped to `channel_embed_mode`
- Default behavior matches previous implementation when not using new features
- Existing checkpoints can be loaded (with auto-detection for new parameters)
- Old training scripts still work (with default values for new parameters)

---

## 7. Key Improvements Summary

### Model Enhancements:
1. ✅ Attention pooling for patch tokens (learnable query)
2. ✅ Optional channel gating mechanism
3. ✅ Multi-seed PMA (K bag queries)
4. ✅ Vectorized channel dropout (performance)

### Training Improvements:
1. ✅ ProxyNCA proxy optimization
2. ✅ Two-tier learning rates (encoder vs. rest)
3. ✅ Encoder freeze/unfreeze schedule
4. ✅ Hard label encoding with filtering
5. ✅ Random sampling with full data coverage
6. ✅ Gradient clipping
7. ✅ Improved LR scheduler (pure Python)
8. ✅ Reproducibility (random seeds)
9. ✅ Enhanced logging

### Evaluation Improvements:
1. ✅ Auto-detection of model configuration
2. ✅ Support for new model parameters
3. ✅ Enhanced visualizations with class names

---

## 8. Recommended Training Configuration

**For best results with new features**:

```bash
--channel-embed-mode attn_pool    # Attention pooling (default)
--pma-num-seeds 4                 # Multi-seed PMA
--encoder-lr-mult 0.2             # Lower LR for pretrained encoder
--freeze-encoder-epochs 0         # Or 2-3 for gradual unfreezing
--warmup-epochs 3                 # Linear warmup
--grad-clip-norm 1.0              # Gradient clipping
--seed 42                         # Reproducibility
```

---

**Last Updated**: December 2024


