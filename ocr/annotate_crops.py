#!/usr/bin/env python3
"""Create candidate character labels with the 22-class ocr_plates.onnx teacher.

Run in the existing OCR virtual environment on the annotation machine. Defaults
target ~/dima/ocr-prep. Only direct image files are read, never nested folders.
Requires numpy, Pillow and onnxruntime-gpu. No packages are installed by this file.
The teacher uses RGB stretched to 416x416, independently of student input sizes.
Labels need review. Images with no detections do not receive empty label files.
Use --resume to continue the same run. Original images are never modified.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time


CLASSES = list("0123456789ABCEHKMOPTXY")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover_journal(path, filenames):
    """Recover complete records; discard only an interrupted final write."""
    records = []
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                break
            record = json.loads(line)
            index = len(records)
            if index >= len(filenames) or record["image"] != filenames[index]:
                raise ValueError("Journal does not match the input image order")
            records.append(record)
    return records


def label_text(characters, width, height):
    lines = []
    for item in characters:
        x1, y1, x2, y2 = item["bbox"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid box: {item['bbox']}")
        box = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
               (x2 - x1) / width, (y2 - y1) / height)
        lines.append(f"{item['class_id']} " + " ".join(f"{v:.10f}" for v in box))
    return "\n".join(lines) + ("\n" if lines else "")


def decode(output, width, height, confidence, nms_iou):
    import numpy as np

    if output.ndim != 3 or output.shape[:2] != (1, 26):
        raise ValueError(f"Expected output [1,26,N], received {output.shape}")
    if output.dtype != np.float32 or not np.isfinite(output).all():
        raise ValueError("Teacher output must be finite float32")
    rows = output[0].T
    class_ids = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), class_ids + 4]
    valid = (scores >= confidence) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
    rows, class_ids, scores = rows[valid], class_ids[valid], scores[valid]
    boxes = np.column_stack((rows[:, 0] - rows[:, 2] / 2,
                             rows[:, 1] - rows[:, 3] / 2,
                             rows[:, 0] + rows[:, 2] / 2,
                             rows[:, 1] + rows[:, 3] / 2))
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * width / 416, 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * height / 416, 0, height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, class_ids, scores = boxes[valid], class_ids[valid], scores[valid]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept = []
    for category in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == category)
        order = indices[np.argsort(-scores[indices], kind="stable")]
        while order.size:
            current = int(order[0])
            kept.append(current)
            rest = order[1:]
            intersection = np.maximum(0, np.minimum(boxes[current, 2:], boxes[rest, 2:])
                                      - np.maximum(boxes[current, :2], boxes[rest, :2]))
            intersection = intersection[:, 0] * intersection[:, 1]
            overlap = intersection / (areas[current] + areas[rest] - intersection)
            order = rest[overlap <= nms_iou]
    characters = [{"class_id": int(class_ids[i]), "char": CLASSES[int(class_ids[i])],
                   "confidence": float(scores[i]), "bbox": boxes[i].tolist()} for i in kept]
    return sorted(characters, key=lambda item: (item["bbox"][0], item["bbox"][1]))


def preview(image, characters, path):
    from PIL import Image, ImageDraw

    scale = min(6, 1600 / image.width, 800 / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas = Image.new("RGB", (max(320, size[0]), size[1] + 28), "white")
    canvas.paste(image.resize(size, Image.Resampling.NEAREST), (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), "CANDIDATE LABELS: REVIEW" if characters else "NO DETECTIONS: REVIEW", fill="black")
    for item in characters:
        x1, y1, x2, y2 = item["bbox"]
        box = (x1 * scale, y1 * scale + 28, x2 * scale, y2 * scale + 28)
        draw.rectangle(box, outline="red", width=1)
        draw.text((box[0], box[1]), item["char"], fill="yellow", stroke_width=1, stroke_fill="black")
    temporary = path.with_name(path.name + ".tmp")
    canvas.save(temporary, format="JPEG", quality=90)
    temporary.replace(path)


def summary(records, total):
    counts = Counter(item["char"] for record in records for item in record["characters"])
    statuses = Counter(record["status"] for record in records)
    return {"status": "candidate_annotations_require_review", "input_images": total,
            "processed": len(records), "complete": len(records) == total,
            "images_with_detections": statuses["needs_review"],
            "images_without_detections": statuses["no_detections"],
            "unreadable_images": statuses["image_error"], "classes": CLASSES,
            "class_objects": {name: counts[name] for name in CLASSES},
            "inference_seconds": sum(record.get("inference_seconds", 0) for record in records)}


def main():
    root = Path.home() / "dima" / "ocr-prep"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=root / "images-50k")
    parser.add_argument("--output", type=Path, default=root / "ocr-onnx-50k")
    parser.add_argument("--model", type=Path, default=root / "license_plate_recognition/cmd/data/ocr_plates.onnx")
    parser.add_argument("--names", type=Path, default=root / "license_plate_recognition/cmd/data/ocr_plates.names")
    parser.add_argument("--expected-images", type=int, default=50000, help="Require this count; 0 accepts any nonempty input")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--nms-iou", type=float, default=0.4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=500, help="Evenly sample at most this many previews; 0 disables")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for field in ("input", "output", "model", "names"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (args.confidence, args.nms_iou)):
        parser.error("Thresholds must be finite values between 0 and 1")
    if min(args.expected_images, args.preview_count, args.device) < 0:
        parser.error("Counts and device ID must be nonnegative")
    if args.output == args.input or args.input in args.output.parents or args.output in args.input.parents:
        parser.error("Input and output must be separate directories")
    names = args.names.read_text(encoding="utf-8-sig").split()
    if names != CLASSES:
        parser.error("Teacher names must be exactly: " + " ".join(CLASSES))
    images = sorted(path for path in args.input.iterdir()
                    if not path.is_symlink() and path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not images or (args.expected_images and len(images) != args.expected_images):
        parser.error(f"Found {len(images)} direct images; expected {args.expected_images}")
    if len({path.stem for path in images}) != len(images):
        parser.error("Image stems collide; labels would overwrite each other")
    fingerprint = hashlib.sha256()
    for path in images:
        stat = path.stat()
        fingerprint.update(json.dumps([path.name, stat.st_size, stat.st_mtime_ns]).encode())
    config = {"format_version": 1, "input": str(args.input), "model": str(args.model),
              "model_sha256": sha256(args.model), "image_fingerprint": fingerprint.hexdigest(),
              "image_count": len(images), "classes": CLASSES, "confidence": args.confidence,
              "nms_iou": args.nms_iou, "nms_class_aware": True, "preview_count": args.preview_count,
              "preprocessing": "RGB, stretch 416x416 (Pillow bilinear), float32 /255, NCHW",
              "bbox_units": "original image pixels, xyxy", "device": args.device}

    import fcntl

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock inode so simultaneous processes cannot lock different files.
    with args.output.with_name(args.output.name + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another annotation process is using this output")
        if args.resume:
            saved = json.loads((args.output / "run.json").read_text())
            if saved != config:
                parser.error("Run settings, model or source files changed; use a new output directory")
            records = recover_journal(args.output / "predictions.jsonl", [p.name for p in images])
            for record in records:
                if record.get("label") and not (args.output / record["label"]).is_file():
                    parser.error(f"Missing completed label: {record['label']}")
        else:
            if args.output.exists():
                parser.error("Output exists; use --resume or choose a new --output")
            records = []
        if len(records) == len(images):
            write_json(args.output / "summary.json", summary(records, len(images)))
            print("All images already processed.")
            return

        import numpy as np
        import onnxruntime as ort
        from PIL import Image

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA provider unavailable; use the existing onnxruntime-gpu environment")
        session = ort.InferenceSession(str(args.model), providers=[("CUDAExecutionProvider", {"device_id": args.device})])
        session.disable_fallback()
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("CUDA provider failed to initialize; CPU fallback is forbidden")
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 3, 416, 416]:
            raise ValueError("Teacher must have one float32 input [1,3,416,416]")
        if len(outputs) != 1 or outputs[0].type != "tensor(float)" or len(outputs[0].shape) != 3 or outputs[0].shape[:2] != [1, 26]:
            raise ValueError("Teacher must have one float32 output [1,26,N]")
        if not args.resume:
            args.output.mkdir()
            (args.output / "labels").mkdir()
            (args.output / "previews").mkdir()
            write_json(args.output / "run.json", config)
            atomic_text(args.output / "classes.names", "\n".join(CLASSES) + "\n")
            (args.output / "predictions.jsonl").touch()
        preview_count = min(args.preview_count, len(images))
        preview_indices = {i * len(images) // preview_count for i in range(preview_count)}
        print(f"Runtime: {ort.__version__}; providers: {session.get_providers()}", flush=True)
        print(f"Images: {len(images)}; completed: {len(records)}; output: {args.output}", flush=True)
        started, initial = time.monotonic(), len(records)
        with (args.output / "predictions.jsonl").open("a", encoding="utf-8") as journal:
            try:
                for index in range(initial, len(images)):
                    path = images[index]
                    record = {"image": path.name, "status": "image_error", "characters": [], "label": None}
                    try:
                        with Image.open(path) as source:
                            image = source.convert("RGB")
                    except (OSError, ValueError, Image.DecompressionBombError) as error:
                        record["error"] = str(error)
                    else:
                        tensor = np.asarray(image.resize((416, 416), Image.Resampling.BILINEAR), dtype=np.float32) / 255
                        tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
                        before = time.perf_counter()
                        output = session.run([outputs[0].name], {inputs[0].name: tensor})[0]
                        record["inference_seconds"] = time.perf_counter() - before
                        characters = decode(output, image.width, image.height, args.confidence, args.nms_iou)
                        record.update(size_wh=list(image.size), characters=characters,
                                      status="needs_review" if characters else "no_detections")
                        if characters:
                            record["label"] = f"labels/{path.stem}.txt"
                            atomic_text(args.output / record["label"], label_text(characters, *image.size))
                        if index in preview_indices:
                            record["preview"] = f"previews/{path.stem}.jpg"
                            preview(image, characters, args.output / record["preview"])
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                    journal.flush()
                    records.append(record)
                    if len(records) % 100 == 0 or len(records) == len(images):
                        os.fsync(journal.fileno())
                        write_json(args.output / "summary.json", summary(records, len(images)))
                        elapsed = time.monotonic() - started
                        speed = (len(records) - initial) / elapsed
                        eta = (len(images) - len(records)) / speed / 60
                        print(f"{len(records)}/{len(images)}; {speed:.1f} images/s; ETA {eta:.1f} min", flush=True)
            finally:
                journal.flush()
                os.fsync(journal.fileno())
                write_json(args.output / "summary.json", summary(records, len(images)))
        print(f"Done. Review candidates; summary: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped. Continue with the same arguments and --resume.")
