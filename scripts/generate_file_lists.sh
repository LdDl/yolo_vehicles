#!/bin/bash
# Generate train/val file lists for Darknet

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATASET_DIR="${1:-$PROJECT_DIR/aic_hcmc2020}"

echo "Dataset: $DATASET_DIR"

find "$DATASET_DIR/images/train" -name "*.jpg" | sort > "$PROJECT_DIR/train_aic_hcmc.txt"
find "$DATASET_DIR/images/val" -name "*.jpg" | sort > "$PROJECT_DIR/val_aic_hcmc.txt"

echo "Generated: $(wc -l < "$PROJECT_DIR/train_aic_hcmc.txt") train, $(wc -l < "$PROJECT_DIR/val_aic_hcmc.txt") val"
