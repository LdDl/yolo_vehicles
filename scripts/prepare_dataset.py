#!/usr/bin/env python3
"""
Prepare AIC HCMC 2020 dataset for YOLO training.

This script:
1. Modifies class labels if needed (swaps class IDs)
2. Copies label files to images directory
3. Generates train/val txt files with absolute paths (for Darknet)
4. Generates train/val txt files with relative paths (for Ultralytics)
"""

import argparse
import shutil
from pathlib import Path


# Class ID mapping: AIC HCMC 2020 -> our custom order
# Modify this if you need different class ordering
CLASS_MAPPING = {
    0: 1,  # AIC motorbike -> motorbike (id=1)
    1: 0,  # AIC car -> car (id=0)
    2: 2,  # AIC bus -> bus (id=2)
    3: 3,  # AIC truck -> truck (id=3)
}


def remap_class_id(class_id: int) -> int:
    """Remap class ID according to CLASS_MAPPING."""
    return CLASS_MAPPING.get(class_id, class_id)


def modify_label_file(src_file: Path, dst_file: Path, remap: bool = True) -> None:
    """Read label file, optionally remap classes, write to destination."""
    with open(src_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        if remap:
            class_id = remap_class_id(class_id)
        parts[0] = str(class_id)
        new_lines.append(' '.join(parts))

    with open(dst_file, 'w') as f:
        f.write('\n'.join(new_lines))


def process_labels(labels_dir: Path, output_dir: Path, remap: bool = True) -> None:
    """Process all label files in directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for label_file in labels_dir.glob('*.txt'):
        dst_file = output_dir / label_file.name
        modify_label_file(label_file, dst_file, remap)

    print(f"Processed labels: {labels_dir} -> {output_dir}")


def copy_labels_to_images(images_dir: Path, labels_dir: Path) -> None:
    """Copy label files to images directory (YOLO expects them side by side)."""
    for label_file in labels_dir.glob('*.txt'):
        shutil.copy(label_file, images_dir / label_file.name)

    print(f"Copied labels to images directory: {images_dir}")


def generate_file_list(
    images_dir: Path,
    source_file: Path,
    output_file: Path,
    absolute: bool = True
) -> None:
    """Generate train/val file list with absolute or relative paths."""
    with open(source_file, 'r') as f:
        lines = f.readlines()

    output_lines = []
    for line in lines:
        fname = line.strip().split()[0]
        if absolute:
            path = str((images_dir / fname).resolve())
        else:
            path = fname
        output_lines.append(path)

    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Generated file list: {output_file} ({'absolute' if absolute else 'relative'})")


def main():
    parser = argparse.ArgumentParser(description='Prepare AIC HCMC 2020 dataset for YOLO')
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('./aic_hcmc2020/aic_hcmc2020'),
        help='Path to extracted dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Output directory for generated files'
    )
    parser.add_argument(
        '--no-remap',
        action='store_true',
        help='Skip class ID remapping'
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    remap = not args.no_remap

    images_dir = dataset_dir / 'images'
    labels_dir = dataset_dir / 'labels'
    labels_custom_dir = dataset_dir / 'labels_custom'

    print("=" * 60)
    print("AIC HCMC 2020 Dataset Preparation")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Class remapping: {'enabled' if remap else 'disabled'}")
    print()

    # Step 1: Process labels
    process_labels(labels_dir, labels_custom_dir, remap=remap)

    # Step 2: Copy labels to images directory
    copy_labels_to_images(images_dir, labels_custom_dir)

    # Step 3: Generate file lists for Darknet (absolute paths)
    train_relative = output_dir / 'train_aic_hcmc_relative.txt'
    val_relative = output_dir / 'val_aic_hcmc_relative.txt'

    if train_relative.exists():
        generate_file_list(
            images_dir,
            train_relative,
            output_dir / 'train_aic_hcmc.txt',
            absolute=True
        )

    if val_relative.exists():
        generate_file_list(
            images_dir,
            val_relative,
            output_dir / 'val_aic_hcmc.txt',
            absolute=True
        )

    # Step 4: Generate file lists for Ultralytics (relative paths)
    # Ultralytics expects paths relative to dataset root
    if train_relative.exists():
        generate_file_list(
            images_dir,
            train_relative,
            dataset_dir / 'train.txt',
            absolute=False
        )

    if val_relative.exists():
        generate_file_list(
            images_dir,
            val_relative,
            dataset_dir / 'val.txt',
            absolute=False
        )

    print()
    print("Done! Dataset is ready for training.")
    print()
    print("For Darknet (v3-tiny, v4-tiny):")
    print(f"  Train: {output_dir / 'train_aic_hcmc.txt'}")
    print(f"  Val:   {output_dir / 'val_aic_hcmc.txt'}")
    print()
    print("For Ultralytics (v8n):")
    print(f"  Config: data/vehicles.yaml")


if __name__ == "__main__":
    main()
