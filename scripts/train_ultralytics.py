#!/usr/bin/env python3
"""Train and export YOLOv5u, YOLOv8n, YOLOv9t and YOLO11n on Junction + MIO-TCD."""

import argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODELS = {
    "v5nu": "yolov5nu",
    "v5su": "yolov5su",
    "v5mu": "yolov5mu",
    "v8n": "yolov8n",
    "v9t": "yolov9t",
    "v11n": "yolo11n",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="v8n")
    parser.add_argument("--data", type=Path, default=PROJECT / "data/generated/vehicles.yaml")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT / "weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--scratch", action="store_true")
    initialization.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.weights and (args.scratch or args.resume):
        parser.error("--weights cannot be combined with --scratch or --resume")
    name = MODELS[args.model]
    destination = args.output.expanduser().resolve() / f"{name}-vehicles"
    if not args.resume and destination.exists():
        parser.error(f"Run already exists: {destination}; use --resume or a new --output")
    if not args.resume and not args.data.is_file():
        parser.error(f"Missing dataset YAML: {args.data}; run prepare_dataset.py configure")
    weights = (args.weights or PROJECT / "weights/pretrained" / f"{name}.pt").expanduser().resolve()
    if not args.scratch and not args.resume and not weights.is_file():
        parser.error(f"Missing weights: {weights}; run scripts/download_pretrained.py --model {args.model}")

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(str(args.resume.expanduser().resolve()))
        model.train(resume=True)
    else:
        # Ultralytics YOLOv5u YAML names omit the 'u' suffix used by its checkpoints.
        architecture = f"{name.removesuffix('u')}.yaml"
        model = YOLO(architecture if args.scratch else str(weights))
        model.train(
            data=str(args.data.expanduser().resolve()), epochs=args.epochs,
            patience=args.patience, batch=args.batch, imgsz=args.imgsz,
            device=args.device, workers=args.workers, seed=42,
            project=str(destination.parent), name=destination.name,
            exist_ok=False, rect=True, pretrained=not args.scratch,
        )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce best.pt: {best}")
    YOLO(str(best)).export(
        format="onnx", imgsz=[256, 416], batch=1,
        dynamic=False, half=False, nms=None, opset=12, simplify=True,
    )
    print(f"Best weights: {best}\nONNX: {best.with_suffix('.onnx')}")


if __name__ == "__main__":
    main()
