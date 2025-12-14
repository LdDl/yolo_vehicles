#!/usr/bin/env python3
"""
Train YOLOv8n using Ultralytics.

Prerequisites:
    pip install ultralytics

Usage:
    python train_ultralytics.py
    python train_ultralytics.py --epochs 100 --batch 16
    python train_ultralytics.py --resume weights/yolov8n-vehicles-last.pt
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8n for vehicles detection')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[416, 256], help='Image size [width, height]')
    parser.add_argument('--device', type=str, default='0', help='CUDA device (0, 1, cpu)')
    parser.add_argument('--workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--pretrained', action='store_true', help='Use COCO pretrained weights')
    args = parser.parse_args()

    # Import here to avoid slow import if just checking --help
    from ultralytics import YOLO

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_yaml = project_dir / 'data' / 'vehicles.yaml'
    weights_dir = project_dir / 'weights'

    weights_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("YOLOv8n Training for Vehicles Detection")
    print("=" * 60)
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
    elif args.pretrained:
        # Start from COCO pretrained weights
        print("Using COCO pretrained weights (yolov8n.pt)")
        model = YOLO('yolov8n.pt')
    else:
        # Train from scratch using our config
        print("Training from scratch")
        model_yaml = project_dir / 'configs' / 'yolov8n-vehicles.yaml'
        model = YOLO(str(model_yaml))

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(weights_dir),
        name='yolov8n-vehicles',
        exist_ok=True,
        pretrained=args.pretrained,
        verbose=True,
    )

    # Export to ONNX for deployment
    print()
    print("Exporting to ONNX...")
    best_weights = weights_dir / 'yolov8n-vehicles' / 'weights' / 'best.pt'

    if best_weights.exists():
        export_model = YOLO(str(best_weights))
        export_model.export(
            format='onnx',
            imgsz=args.imgsz,
            opset=12,
            simplify=True,
        )
        print(f"ONNX exported: {best_weights.with_suffix('.onnx')}")

    print()
    print("Training complete!")
    print(f"Best weights: {best_weights}")


if __name__ == "__main__":
    main()
