#!/usr/bin/env python3
"""Download the published plate dataset and generate local training files."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile

import yaml
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
SLUG = "dimahkiin/russian-license-plates-5-class-detection"
NAMES = ["civilian", "taxi", "military", "police", "diplomatic"]
SPLITS = ("train", "val", "test")


def validate(dataset):
    config = yaml.safe_load((dataset / "data_multiclass.yaml").read_text())
    names = config.get("names")
    if names != NAMES and names != dict(enumerate(NAMES)):
        raise ValueError("Expected five classes in the published order")
    pairs, summary, hashes = {}, {}, {}
    for split in SPLITS:
        images = sorted(p for p in (dataset / "images" / split).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        labels = {p.stem: p for p in (dataset / "labels" / split).glob("*.txt")}
        if not images or len({p.stem for p in images}) != len(images) or set(labels) != {p.stem for p in images}:
            raise ValueError(f"Missing pairs or duplicate image stems: {split}")
        counts = [0] * len(NAMES)
        empty = 0
        pairs[split] = []
        for image in images:
            with Image.open(image) as header:
                if min(header.size) <= 0:
                    raise ValueError(f"Invalid image size: {image}")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError(f"Exact image duplicate across splits: {image}")
            hashes[digest] = split
            label = labels[image.stem]
            rows = label.read_text().splitlines()
            empty += int(not rows)
            for number, row in enumerate(rows, 1):
                fields = row.split()
                if len(fields) != 5:
                    raise ValueError(f"{label}:{number}: expected class xc yc w h")
                category = int(fields[0])
                x, y, w, h = map(float, fields[1:])
                if not 0 <= category < len(NAMES) or not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
                    raise ValueError(f"{label}:{number}: invalid class or box")
                # Published YOLO coordinates are rounded to six decimal places.
                if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                    raise ValueError(f"{label}:{number}: box outside image")
                counts[category] += 1
            adjacent = image.with_suffix(".txt")
            if adjacent.exists() and adjacent.read_bytes() != label.read_bytes():
                raise ValueError(f"Conflicting adjacent annotation: {adjacent}")
            pairs[split].append((image, label))
        summary[split] = {"images": len(images), "empty_images": empty, "class_objects": dict(zip(NAMES, counts))}
    return pairs, summary


def extract(archive, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".plates-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Unsafe archive member: {entry.filename}")
            source.extractall(staging)
        roots = [p.parent for p in staging.rglob("data_multiclass.yaml") if (p.parent / "images/train").is_dir()]
        if len(roots) != 1:
            raise ValueError("Expected one dataset root containing data_multiclass.yaml and images/train")
        validate(roots[0])
        if destination.exists():
            raise ValueError(f"Destination already exists: {destination}")
        roots[0].rename(destination)


def download(archive):
    if archive.exists():
        print(f"Using existing archive: {archive}")
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kaggle-", dir=archive.parent) as temporary:
        subprocess.run(["kaggle", "datasets", "download", "-d", SLUG, "-p", temporary], check=True)
        archives = list(Path(temporary).glob("*.zip"))
        if len(archives) != 1 or not zipfile.is_zipfile(archives[0]):
            raise ValueError("Kaggle did not return one ZIP archive")
        archives[0].rename(archive)


def configure(dataset, output, backup):
    pairs, summary = validate(dataset)
    output.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)
    for split, entries in pairs.items():
        for image, label in entries:
            adjacent = image.with_suffix(".txt")
            if not adjacent.exists():
                shutil.copy2(label, adjacent)
        (output / f"plates-{split}.txt").write_text("".join(f"{image}\n" for image, _ in entries))
    (output / "plates.names").write_text("\n".join(NAMES) + "\n")
    for suffix, split in (("", "val"), ("-test", "test")):
        (output / f"plates{suffix}.data").write_text(
            f"classes = 5\ntrain = {output}/plates-train.txt\nvalid = {output}/plates-{split}.txt\n"
            f"names = {output}/plates.names\nbackup = {backup}\n"
        )
    config = {"path": str(dataset), "train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(NAMES))}
    (output / "plates.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "plates-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Training config: {output / 'plates.yaml'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "datasets/plates")
    parser.add_argument("--archive", type=Path, default=PROJECT / "datasets/raw/russian-license-plates-5-class-detection.zip")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/generated")
    args = parser.parse_args(argv)
    dataset, archive, output = [p.expanduser().resolve() for p in (args.dataset, args.archive, args.output)]
    try:
        if not dataset.exists():
            download(archive)
            extract(archive, dataset)
        configure(dataset, output, PROJECT / "weights")
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.CalledProcessError, yaml.YAMLError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
