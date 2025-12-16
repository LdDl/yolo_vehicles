#!/usr/bin/env python3
"""
Train YOLO models (v8, v9, v11) using Ultralytics.

Prerequisites:
    pip install ultralytics

Usage:
    # YOLOv8n (default)
    python train_ultralytics.py --epochs 100

    # YOLOv9t
    python train_ultralytics.py --model v9t --epochs 100

    # YOLOv11n
    python train_ultralytics.py --model v11n --epochs 100

    # Resume training
    python train_ultralytics.py --resume weights/yolov9t-vehicles/weights/last.pt

Supported models (lightweight for edge deployment):
    v8n  - YOLOv8 nano  (~3.2M params)
    v9t  - YOLOv9 tiny  (~2.0M params)
    v11n - YOLOv11 nano (~2.6M params)
"""

import argparse
from pathlib import Path

# Model name mappings: our short names -> Ultralytics pretrained weights
# Focus on lightweight models for edge deployment (Jetson Nano)
MODEL_MAP = {
    'v8n': 'yolov8n.pt',   # YOLOv8 nano (~3.2M params)
    'v9t': 'yolov9t.pt',   # YOLOv9 tiny (~2.0M params)
    'v11n': 'yolo11n.pt',  # YOLOv11 nano (~2.6M params)
}


def get_model_display_name(model_key: str) -> str:
    """Convert model key to display name (e.g., 'v9t' -> 'YOLOv9t')."""
    if model_key.startswith('v8'):
        return f"YOLOv8{model_key[2:]}"
    elif model_key.startswith('v9'):
        return f"YOLOv9{model_key[2:]}"
    elif model_key.startswith('v11'):
        return f"YOLOv11{model_key[3:]}"
    return model_key


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO models for vehicles detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_ultralytics.py --model v8n --epochs 100
    python train_ultralytics.py --model v9t --epochs 100 --pretrained
    python train_ultralytics.py --model v11n --epochs 100 --batch 16
        """
    )
    parser.add_argument('--model', type=str, default='v8n',
                        choices=list(MODEL_MAP.keys()),
                        help='Model variant (default: v8n)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=-1, help='Batch size (-1 for auto)')
    parser.add_argument('--imgsz', type=int, default=416, help='Image size (single int for training)')
    parser.add_argument('--device', type=str, default='0', help='CUDA device (0, 1, cpu)')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--pretrained', action='store_true', help='Use COCO pretrained weights (recommended)')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch (v8n only, uses custom config)')
    args = parser.parse_args()

    # Import here to avoid slow import if just checking --help
    from ultralytics import YOLO

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_yaml = project_dir / 'data' / 'vehicles.yaml'
    weights_dir = project_dir / 'weights'

    weights_dir.mkdir(exist_ok=True)

    model_name = get_model_display_name(args.model)
    project_name = f"{model_name.lower()}-vehicles"

    print("=" * 60)
    print(f"{model_name} Training for Vehicles Detection")
    print("=" * 60)
    print(f"Model: {args.model} ({MODEL_MAP[args.model]})")
    print(f"Data config: {data_yaml}")
    print(f"Image size: {args.imgsz}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print()

    if args.resume:
        # Resume from checkpoint
        print(f"Resuming from: {args.resume}")
        model = YOLO(args.resume)
    elif args.scratch and args.model == 'v8n':
        # Train from scratch using custom config (v8n only)
        print("Training from scratch (custom config)")
        model_yaml = project_dir / 'configs' / 'yolov8n-vehicles.yaml'
        model = YOLO(str(model_yaml))
    else:
        # Use pretrained weights (default for v9/v11, optional for v8)
        pretrained_name = MODEL_MAP[args.model]
        print(f"Using COCO pretrained weights ({pretrained_name})")
        model = YOLO(pretrained_name)

    # Train
    # rect=True enables rectangular batching (adapts to each batch's aspect ratio)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(weights_dir),
        name=project_name,
        exist_ok=True,
        pretrained=args.pretrained or not args.scratch,
        verbose=True,
        rect=True,
    )

    # Export to ONNX for deployment
    print()
    print("Exporting to ONNX...")
    best_weights = weights_dir / project_name / 'weights' / 'best.pt'

    if best_weights.exists():
        export_model = YOLO(str(best_weights))
        # imgsz=[height, width] in Ultralytics - use [256, 416] for 416x256 (width x height)
        export_model.export(
            format='onnx',
            imgsz=[256, 416],  # height=256, width=416 to match Darknet 416x256
            opset=12,
            simplify=True,
        )
        print(f"ONNX exported: {best_weights.with_suffix('.onnx')}")

    print()
    print("Training complete!")
    print(f"Best weights: {best_weights}")


if __name__ == "__main__":
    main()
