#!/usr/bin/env python3
"""
Train YOLOv5 for vehicles detection.

This script uses the original YOLOv5 (not yolov5u/yolov5nu from the main ultralytics
package, which has YOLOv8-style output format).

Prerequisites:
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5 && pip install -r requirements.txt

Usage:
    # YOLOv5n (default)
    python scripts/train_yolov5.py --epochs 100

    # YOLOv5s
    python scripts/train_yolov5.py --model v5s --epochs 100

    # Train from scratch
    python scripts/train_yolov5.py --model v5n --epochs 100 --scratch

    # Resume training
    python scripts/train_yolov5.py --resume weights/yolov5n-vehicles/weights/last.pt

Supported models (lightweight for edge deployment):
    v5n  - YOLOv5 nano  (~1.9M params)
    v5s  - YOLOv5 small (~7.2M params)
    v5m  - YOLOv5 medium (~21M params)
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import yaml

# Model name mappings: our short names -> YOLOv5 pretrained weights / architecture configs
MODEL_PRETRAINED = {
    'v5n': 'yolov5n.pt',   # YOLOv5 nano (~1.9M params)
    'v5s': 'yolov5s.pt',   # YOLOv5 small (~7.2M params)
    'v5m': 'yolov5m.pt',   # YOLOv5 medium (~21M params)
}

MODEL_YAML = {
    'v5n': 'yolov5n.yaml',
    'v5s': 'yolov5s.yaml',
    'v5m': 'yolov5m.yaml',
}


def get_model_display_name(model_key: str) -> str:
    """Convert model key to display name (e.g., 'v5n' -> 'YOLOv5n')."""
    return f"YOLOv5{model_key[2:]}"


def create_absolute_data_yaml(data_yaml: Path, project_dir: Path) -> Path:
    """Create a temporary data yaml with absolute paths for YOLOv5 compatibility."""
    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    # Resolve relative path to absolute
    dataset_path = config.get('path', '.')
    if not Path(dataset_path).is_absolute():
        dataset_path = str((project_dir / dataset_path).resolve())
    config['path'] = dataset_path

    # Write to temp file
    temp_yaml = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False, prefix='yolov5_data_'
    )
    yaml.dump(config, temp_yaml, default_flow_style=False)
    temp_yaml.close()

    return Path(temp_yaml.name)


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv5 for vehicles detection (not yolov5u)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/train_yolov5.py --model v5n --epochs 100
    python scripts/train_yolov5.py --model v5s --epochs 100 --scratch
    python scripts/train_yolov5.py --model v5n --epochs 100 --batch 16
        """
    )
    parser.add_argument('--model', type=str, default='v5n',
                        choices=list(MODEL_PRETRAINED.keys()),
                        help='Model variant (default: v5n)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size (single int for training)')
    parser.add_argument('--device', type=str, default='0', help='CUDA device (0, 1, cpu)')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch (no pretrained weights)')
    parser.add_argument('--yolov5-dir', type=str, default=None, help='Path to YOLOv5 repo')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    # Find YOLOv5 repository
    if args.yolov5_dir:
        yolov5_dir = Path(args.yolov5_dir)
    else:
        yolov5_dir = project_dir / 'yolov5'

    if not yolov5_dir.exists() or not (yolov5_dir / 'train.py').exists():
        print(f"Error: YOLOv5 repository not found at {yolov5_dir}")
        print("Clone it first:")
        print("  git clone https://github.com/ultralytics/yolov5.git")
        print("  cd yolov5 && pip install -r requirements.txt")
        sys.exit(1)

    # Add YOLOv5 to path
    sys.path.insert(0, str(yolov5_dir))
    from train import run as train_run
    from export import run as export_run

    data_yaml_orig = project_dir / 'data' / 'vehicles.yaml'
    weights_dir = project_dir / 'weights'

    weights_dir.mkdir(exist_ok=True)

    # Create temp yaml with absolute paths (YOLOv5 resolves relative paths from its own dir)
    data_yaml = create_absolute_data_yaml(data_yaml_orig, project_dir)

    model_name = get_model_display_name(args.model)
    project_name = f"{model_name.lower()}-vehicles"

    print("=" * 60)
    print(f"{model_name} Training for Vehicles Detection")
    print("=" * 60)
    print(f"YOLOv5 repo: {yolov5_dir}")
    print(f"Model: {args.model} ({MODEL_PRETRAINED[args.model]})")
    print(f"Data config: {data_yaml_orig}")
    print(f"Image size: {args.imgsz}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print()

    train_kwargs = {
        'data': str(data_yaml),
        'epochs': args.epochs,
        'batch_size': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': str(weights_dir),
        'name': project_name,
        'exist_ok': True,
        'rect': True,
    }

    if args.resume:
        print(f"Resuming from: {args.resume}")
        train_kwargs = {'data': str(data_yaml), 'resume': args.resume}
    elif args.scratch:
        model_yaml = MODEL_YAML[args.model]
        print(f"Training from scratch ({model_yaml})")
        train_kwargs['cfg'] = model_yaml
    else:
        pretrained_name = MODEL_PRETRAINED[args.model]
        print(f"Using COCO pretrained weights ({pretrained_name})")
        train_kwargs['weights'] = pretrained_name

    train_run(**train_kwargs)

    # Cleanup temp yaml
    os.unlink(data_yaml)

    # Export to ONNX for deployment
    print()
    print("Exporting to ONNX...")
    best_weights = weights_dir / project_name / 'weights' / 'best.pt'

    if best_weights.exists():
        export_run(
            weights=str(best_weights),
            imgsz=[256, 416],  # height=256, width=416 to match Darknet 416x256
            include=['onnx'],
            opset=12,
            simplify=True,
        )
        print(f"ONNX exported: {best_weights.with_suffix('.onnx')}")

    print()
    print("Training complete!")
    print(f"Best weights: {best_weights}")


if __name__ == "__main__":
    main()
