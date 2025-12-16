#!/usr/bin/env python3
"""
Distillation: Use YOLOv8-large to generate pseudo-labels for training smaller models.

This script:
1. Takes images directly OR extracts frames from videos
2. Runs YOLOv8l (or custom teacher model) to detect vehicles
3. Filters detections by confidence threshold
4. Saves annotations in YOLO format

Usage:
    # From images (recommended if you already have images)
    python scripts/distill_annotations.py --image-dir path/to/images/

    # From videos
    python scripts/distill_annotations.py --video path/to/video.mp4
    python scripts/distill_annotations.py --video-dir path/to/videos/

    # With custom teacher model
    python scripts/distill_annotations.py --image-dir imgs/ --teacher weights/yolov8l-vehicles.pt --no-coco-mapping
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# COCO class IDs that map to our vehicle classes
COCO_TO_VEHICLES = {
    2: 0,   # car -> car
    3: 1,   # motorcycle -> motorbike
    5: 2,   # bus -> bus
    7: 3,   # truck -> truck
}

VEHICLE_NAMES = ['car', 'motorbike', 'bus', 'truck']


def extract_frames(video_path: Path, output_dir: Path, frame_step: int = 10) -> list[Path]:
    """Extract every Nth frame from video."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return []

    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_output = total_frames // frame_step

    frames = []
    frame_idx = 0
    saved_idx = 0
    video_name = video_path.stem

    # Setup progress bar
    pbar = tqdm(total=expected_output, desc=f"Extracting {video_name}", unit="frame") if tqdm else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            frame_path = output_dir / f"{video_name}_{saved_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append(frame_path)
            saved_idx += 1
            if pbar:
                pbar.update(1)

        frame_idx += 1

    if pbar:
        pbar.close()
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path.name}")
    return frames


def annotate_with_teacher(
    model: YOLO,
    image_paths: list[Path],
    output_dir: Path,
    confidence: float = 0.5,
    use_coco_mapping: bool = True,
) -> int:
    """Run teacher model and save YOLO format annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total_annotations = 0

    # Use tqdm if available, otherwise plain iterator
    iterator = tqdm(image_paths, desc="Annotating", unit="img") if tqdm else image_paths

    for img_path in iterator:
        results = model(img_path, verbose=False)[0]

        # Get image dimensions
        img_h, img_w = results.orig_shape

        annotations = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < confidence:
                continue

            cls_id = int(box.cls[0])

            # Map COCO classes to vehicle classes if using pretrained model
            if use_coco_mapping:
                if cls_id not in COCO_TO_VEHICLES:
                    continue
                cls_id = COCO_TO_VEHICLES[cls_id]
            else:
                # Custom model already has correct classes
                if cls_id >= len(VEHICLE_NAMES):
                    continue

            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Convert to YOLO format (center_x, center_y, width, height) normalized
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            annotations.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Save annotation file (always create, even if empty)
        label_path = output_dir / f"{img_path.stem}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
        total_annotations += len(annotations)

    return total_annotations


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-labels using teacher model')
    parser.add_argument('--video', type=Path, help='Single video file')
    parser.add_argument('--video-dir', type=Path, help='Directory with video files')
    parser.add_argument('--image-dir', type=Path, help='Directory with images (jpg/png)')
    parser.add_argument('--output', type=Path, default=Path('distilled_data'),
        help='Output directory (default: distilled_data)')
    parser.add_argument('--teacher', type=str, default='yolov8l.pt',
        help='Teacher model (default: yolov8l.pt pretrained)')
    parser.add_argument('--confidence', type=float, default=0.5,
        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--frame-step', type=int, default=10,
        help='Extract every Nth frame from videos (default: 10)')
    parser.add_argument('--no-coco-mapping', action='store_true',
        help='Disable COCO->vehicle class mapping (for custom teacher)')
    parser.add_argument('--copy-images', action='store_true',
        help='Copy images to output dir (for --image-dir mode)')
    args = parser.parse_args()

    # Determine input mode
    mode = None
    if args.image_dir:
        mode = 'images'
        images = list(args.image_dir.glob('*.jpg')) + list(args.image_dir.glob('*.png'))
        if not images:
            print(f"No images found in {args.image_dir}")
            return
        print(f"Found {len(images)} image(s)")
    elif args.video or args.video_dir:
        mode = 'videos'
        videos = []
        if args.video:
            videos = [args.video]
        else:
            videos = list(args.video_dir.glob('*.mp4')) + list(args.video_dir.glob('*.avi'))
        if not videos:
            print("No video files found")
            return
        print(f"Found {len(videos)} video(s)")
        print(f"Frame step: every {args.frame_step} frames")
    else:
        parser.error('Provide --image-dir, --video, or --video-dir')

    print(f"Teacher model: {args.teacher}")
    print(f"Confidence threshold: {args.confidence}")
    print()

    # Load teacher model
    model = YOLO(args.teacher)
    use_coco = not args.no_coco_mapping and 'yolov8' in args.teacher and 'vehicles' not in args.teacher

    if use_coco:
        print("Using COCO class mapping (car, motorcycle, bus, truck)")
    else:
        print("Using direct class IDs (custom teacher model)")
    print()

    # Setup output dirs
    images_dir = args.output / 'images'
    labels_dir = args.output / 'labels'

    total_images = 0
    total_annotations = 0

    if mode == 'images':
        # Process images directly
        print(f"Processing {len(images)} images...")

        # Optionally copy images to output
        if args.copy_images:
            import shutil
            images_dir.mkdir(parents=True, exist_ok=True)
            for img in images:
                shutil.copy(img, images_dir / img.name)
            print(f"Copied images to {images_dir}")

        # Generate annotations
        n_annotations = annotate_with_teacher(
            model, images, labels_dir,
            confidence=args.confidence,
            use_coco_mapping=use_coco
        )
        total_images = len(images)
        total_annotations = n_annotations
        print(f"Generated {n_annotations} annotations")

    else:
        # Process videos
        for video in videos:
            print(f"Processing {video.name}...")

            # Extract frames
            frames = extract_frames(video, images_dir, args.frame_step)
            total_images += len(frames)

            # Generate annotations
            n_annotations = annotate_with_teacher(
                model, frames, labels_dir,
                confidence=args.confidence,
                use_coco_mapping=use_coco
            )
            total_annotations += n_annotations
            print(f"  Generated {n_annotations} annotations")
            print()

    print("=" * 50)
    print(f"Total images: {total_images}")
    print(f"Total annotations generated: {total_annotations}")
    print(f"Labels saved to: {labels_dir}")
    print()
    print("To use this data for training:")
    if mode == 'images' and not args.copy_images:
        print(f"  1. Labels are in {labels_dir}, images stay in {args.image_dir}")
    else:
        print(f"  1. Images and labels are in {args.output}")
    print(f"  2. Review/clean annotations if needed")
    print(f"  3. Add to your training set and update file lists")


if __name__ == '__main__':
    main()
