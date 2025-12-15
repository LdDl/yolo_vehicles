#!/bin/bash
# Train YOLO v3-tiny or v4-tiny using Darknet (AlexeyAB fork)
#
# Prerequisites:
#   - Darknet compiled with GPU=1 CUDNN=1 OPENCV=1
#   - Dataset prepared with prepare_dataset.py
#
# Usage:
#   ./train_darknet.sh v3-tiny    # Train YOLOv3-tiny
#   ./train_darknet.sh v4-tiny    # Train YOLOv4-tiny

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
DATA_FILE="$PROJECT_DIR/data/vehicles.data"
WEIGHTS_DIR="$PROJECT_DIR/weights"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <model>"
    echo "  model: v3-tiny or v4-tiny"
    exit 1
fi

MODEL="$1"

case "$MODEL" in
    v3-tiny|yolov3-tiny)
        CFG_FILE="$PROJECT_DIR/configs/yolov3-tiny-vehicles.cfg"
        echo "Training YOLOv3-tiny..."
        ;;
    v4-tiny|yolov4-tiny)
        CFG_FILE="$PROJECT_DIR/configs/yolov4-tiny-vehicles.cfg"
        echo "Training YOLOv4-tiny..."
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Supported models: v3-tiny, v4-tiny"
        exit 1
        ;;
esac

# Check if darknet is available
if ! command -v darknet &> /dev/null; then
    echo "Error: 'darknet' command not found."
    echo "Please ensure Darknet is compiled and in your PATH."
    echo "See: https://github.com/AlexeyAB/darknet"
    exit 1
fi

# Check if config exists
if [ ! -f "$CFG_FILE" ]; then
    echo "Error: Config file not found: $CFG_FILE"
    exit 1
fi

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    echo "Run prepare_dataset.py first."
    exit 1
fi

# Create weights directory
mkdir -p "$WEIGHTS_DIR"

echo ""
echo "Configuration:"
echo "  Data file: $DATA_FILE"
echo "  Config file: $CFG_FILE"
echo "  Weights output: $WEIGHTS_DIR"
echo ""

# Start training
# -map flag enables mAP calculation during training
# -dont_show disables chart window (useful for headless servers)
# Add -gpus 0,1 for multi-GPU training

cd "$PROJECT_DIR"
darknet detector train "$DATA_FILE" "$CFG_FILE" -map -dont_show

echo ""
echo "Training complete!"
echo "Best weights: $WEIGHTS_DIR/yolov*_best.weights"
