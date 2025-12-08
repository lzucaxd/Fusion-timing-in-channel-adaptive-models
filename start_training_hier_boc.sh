#!/bin/bash
# Quick start script for HierBoCSetViT training

python training/train_hier_boc_setvit.py \
    --csv-file /Users/zamfiraluca/Downloads/CHAMMI/combined_metadata.csv \
    --root-dir /Users/zamfiraluca/Downloads/CHAMMI \
    --batch-size 32 \
    --epochs 50 \
    --head-mode proxynca \
    --output-dir ./checkpoints/hier_boc_setvit_proxynca \
    --num-workers 4
