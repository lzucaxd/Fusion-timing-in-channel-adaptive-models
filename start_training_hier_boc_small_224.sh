#!/bin/bash
# Training script for HierBoCSetViT with ViT-Small and 224x224 images

python training/train_hier_boc_setvit.py \
    --csv-file /Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv \
    --root-dir /Users/zamfiraluca/Downloads/CHAMMI \
    --batch-size 16 \
    --epochs 20 \
    --img-size 224 \
    --encoder-type small \
    --embed-dim None \
    --head-mode proxynca \
    --metric-embed-dim 192 \
    --aggregator-depth 2 \
    --aggregator-num-heads 6 \
    --channel-dropout-p 0.3 \
    --output-dir ./checkpoints/hier_boc_setvit_small_224x224 \
    --num-workers 4 \
    --lr 1e-4 \
    --weight-decay 0.01 \
    --temperature 0.05 \
    2>&1 | tee training_hier_boc_small_224_log.txt

