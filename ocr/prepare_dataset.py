#!/usr/bin/env python3
"""Download the published OCR dataset and generate Darknet/Ultralytics inputs."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile

from PIL import Image
import yaml


PROJECT = Path(__file__).resolve().parents[1]
SLUG = "dimahkiin/russian-license-plate-characters-23-classes"
NAMES = list("0123456789ABCEHKMOPTXYD")
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_manifest(dataset, image_info):
    path = dataset / "manifest.csv"
    if not path.exists():
        return
    seen, groups = set(), {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        require({"image", "label", "group", "sha256"} <= set(reader.fieldnames or []), "Unexpected manifest columns")
        for row in reader:
            relative = row["image"]
            require(relative in image_info and relative not in seen, f"Unknown or repeated manifest image: {relative}")
            seen.add(relative)
            split, label, digest = image_info[relative]
            require(row["label"] == label, f"Manifest label mismatch: {relative}")
            require(row["sha256"] == digest, f"Image checksum mismatch: {relative}")
            require(bool(row["group"]), f"Missing group ID: {relative}")
            previous = groups.setdefault(row["group"], split)
            require(previous == split, f"Manifest group crosses splits: {row['group']}")
    require(seen == set(image_info), "Manifest does not cover every image")


def validate(dataset):
    config = yaml.safe_load((dataset / "data.yaml").read_text())
    require(isinstance(config, dict), "Expected a dataset YAML mapping")
    require(config.get("names") in (NAMES, dict(enumerate(NAMES))), "Expected 23 classes in the published order, including D at ID 22")
    require(config.get("nc", len(NAMES)) == len(NAMES), "Expected nc=23")
    for split in SPLITS:
        require(config.get(split) == f"images/{split}", f"Unexpected {split} path in dataset YAML")
    names_file = dataset / "classes.names"
    if names_file.exists():
        require(names_file.read_text().split() == NAMES, "classes.names does not match the OCR alphabet")
    pairs, summary, hashes, image_info = {}, {}, {}, {}
    for split in SPLITS:
        images = sorted(p for p in (dataset / "images" / split).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        labels = {p.stem: p for p in (dataset / "labels" / split).glob("*.txt")}
        require(images and len({p.stem for p in images}) == len(images) and set(labels) == {p.stem for p in images},
                f"Missing pairs, empty split or colliding image stems: {split}")
        counts, empty = [0] * len(NAMES), 0
        pairs[split] = []
        for index, image in enumerate(images, 1):
            require(not image.is_symlink(), f"Image symlinks are not supported: {image}")
            with Image.open(image) as header:
                require(min(header.size) > 0, f"Invalid image dimensions: {image}")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            previous = hashes.setdefault(digest, split)
            require(previous == split, f"Exact image duplicate across splits: {image}")
            label = labels[image.stem]
            require(label.is_file() and not label.is_symlink(), f"Invalid label file: {label}")
            rows = label.read_text().splitlines()
            empty += int(not rows)
            for number, row in enumerate(rows, 1):
                fields = row.split()
                require(len(fields) == 5, f"{label}:{number}: expected class xc yc w h")
                category = int(fields[0])
                x, y, w, h = map(float, fields[1:])
                require(0 <= category < len(NAMES) and all(math.isfinite(v) for v in (x, y, w, h)) and min(w, h) > 0,
                        f"{label}:{number}: invalid class or box")
                # Published coordinates are rounded; accept the rounding tolerance.
                require(min(x - w / 2, y - h / 2) >= -1e-6 and max(x + w / 2, y + h / 2) <= 1 + 1e-6,
                        f"{label}:{number}: box outside image")
                counts[category] += 1
            adjacent = image.with_suffix(".txt")
            if adjacent.exists():
                require(adjacent.is_file() and not adjacent.is_symlink() and adjacent.read_bytes() == label.read_bytes(),
                        f"Conflicting adjacent annotation: {adjacent}")
            pairs[split].append((image, label))
            image_info[image.relative_to(dataset).as_posix()] = (split, label.relative_to(dataset).as_posix(), digest)
            if index % 10000 == 0:
                print(f"{split}: checked {index}/{len(images)}", flush=True)
        # Keep absent classes, including D, in the output taxonomy and summary.
        summary[split] = {"images": len(images), "empty_images": empty, "class_objects": dict(zip(NAMES, counts))}
    validate_manifest(dataset, image_info)
    return pairs, summary


def unpack(archive, destination):
    with zipfile.ZipFile(archive) as source:
        files = set()
        for entry in source.infolist():
            path = PurePosixPath(entry.filename)
            mode = (entry.external_attr >> 16) & 0o170000
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in entry.filename and mode in (0, 0o100000, 0o040000),
                    f"Unsafe archive member: {entry.filename}")
            if not entry.is_dir():
                require(str(path) not in files, f"Repeated archive member: {entry.filename}")
                files.add(str(path))
        source.extractall(destination)


def dataset_roots(directory):
    roots = []
    for folder, subfolders, files in os.walk(directory):
        path = Path(folder)
        if "data.yaml" in files and (path / "images/train").is_dir() and (path / "labels/train").is_dir():
            roots.append(path)
            subfolders[:] = []
        else:
            subfolders[:] = [name for name in subfolders if name not in ("images", "labels")]
    return roots


def extract(archive, destination):
    require(not destination.exists(), f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ocr-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        unpack(archive, staging / "outer")
        roots = dataset_roots(staging / "outer")
        if not roots:
            # Kaggle downloads can wrap the uploaded v1 ZIP in another ZIP.
            nested = list((staging / "outer").rglob("*.zip"))
            require(len(nested) == 1, "Expected one dataset root or one nested dataset ZIP")
            unpack(nested[0], staging / "inner")
            roots = dataset_roots(staging / "inner")
        require(len(roots) == 1, "Expected exactly one OCR dataset root")
        validate(roots[0])
        require(not destination.exists(), f"Destination appeared during extraction: {destination}")
        roots[0].rename(destination)


def download(archive):
    if archive.exists():
        require(archive.is_file() and zipfile.is_zipfile(archive), f"Invalid existing ZIP: {archive}")
        print(f"Using existing archive: {archive}", flush=True)
        return
    if shutil.which("kaggle") is None:
        raise ValueError("Kaggle CLI not found. Activate your venv and run: python3 -m pip install -r ocr/requirements.txt")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kaggle-ocr-", dir=archive.parent) as temporary:
        subprocess.run(["kaggle", "datasets", "download", "-d", SLUG, "-p", temporary], check=True)
        archives = list(Path(temporary).glob("*.zip"))
        require(len(archives) == 1 and zipfile.is_zipfile(archives[0]), "Kaggle did not return one ZIP archive")
        require(not archive.exists(), f"Archive appeared during download: {archive}")
        archives[0].rename(archive)


def configure(dataset, output, backup):
    require(dataset != output and dataset not in output.parents and output not in dataset.parents,
            "Generated training files must be outside the dataset directory")
    pairs, summary = validate(dataset)
    output.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)
    for split, entries in pairs.items():
        for image, label in entries:
            adjacent = image.with_suffix(".txt")
            if not adjacent.exists():
                shutil.copy2(label, adjacent)
        (output / f"ocr-{split}.txt").write_text("".join(f"{image}\n" for image, _ in entries))
    (output / "ocr.names").write_text("\n".join(NAMES) + "\n")
    for suffix, split in (("", "val"), ("-test", "test")):
        (output / f"ocr{suffix}.data").write_text(
            f"classes = {len(NAMES)}\ntrain = {output}/ocr-train.txt\nvalid = {output}/ocr-{split}.txt\n"
            f"names = {output}/ocr.names\nbackup = {backup}\n"
        )
    config = {"path": str(dataset), "train": "images/train", "val": "images/val", "test": "images/test",
              "nc": len(NAMES), "names": dict(enumerate(NAMES))}
    (output / "ocr.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "ocr-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Training config: {output / 'ocr.yaml'}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "datasets/ocr")
    parser.add_argument("--archive", type=Path, default=PROJECT / "datasets/raw/russian-license-plate-characters-23-classes.zip")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/generated")
    args = parser.parse_args(argv)
    dataset, archive, output = [p.expanduser().resolve() for p in (args.dataset, args.archive, args.output)]
    try:
        if not dataset.exists():
            download(archive)
            extract(archive, dataset)
        configure(dataset, output, PROJECT / "weights")
    except (OSError, ValueError, KeyError, csv.Error, zipfile.BadZipFile, subprocess.CalledProcessError, yaml.YAMLError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
