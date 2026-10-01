#!/usr/bin/env python3
"""Download, convert and merge Junction and MIO-TCD localization datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tarfile
import tempfile
import time
import urllib.request
import zipfile


# These are our four classes, not the original datasets' class IDs.
# ID 1 is the same category as COCO's "motorcycle".
CLASS_NAMES = ["car", "motorbike", "bus", "truck"]

# None removes a non-target annotation, but keeps the other objects in the image.
# Our taxonomy treats vans as cars, minibuses as buses and pickups as trucks.
JUNCTION_CLASS_MAP = {
    "car": 0,
    "taksi": 0,
    "van": 0,
    "motorcycle": 1,
    "bus": 2,
    "minibus": 2,
    "light_truck": 3,
    "heavy_truck": 3,
    "person": None,
    "bicycle": None,
}

# An ambiguous motor vehicle cannot safely become an unlabelled background object.
# Images containing this value are excluded entirely during conversion.
AMBIGUOUS = "exclude_image"
MIO_CLASS_MAP = {
    "car": 0,
    "work_van": 0,
    "motorcycle": 1,
    "bus": 2,
    "pickup_truck": 3,
    "single_unit_truck": 3,
    "articulated_truck": 3,
    "pedestrian": None,
    "bicycle": None,
    "non-motorized_vehicle": None,
    "motorized_vehicle": AMBIGUOUS,
}

DATASETS = {
    "junction": {
        "folder": "Junction-based Vehicle Detection Dataset",
        "archive": "Junction-based Vehicle Detection Dataset.zip",
        "url": "https://data.mendeley.com/public-api/zip/vwjg6b7kpt/download/1",
        "required": ["sample_labels.csv", "sampled_images"],
    },
    "mio": {
        "folder": "MIO-TCD-Localization",
        "archive": "MIO-TCD-Localization.tar",
        "url": "https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar",
        "required": ["gt_train.csv", "train"],
    },
}
SPLITS = ("train", "val", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def output_directory(destination, dry_run=False):
    """Publish only complete outputs; never overwrite an existing dataset."""
    if dry_run:
        yield None
        return
    require(not destination.exists(), f"Output already exists: {destination}. Choose a new output path.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as name:
        staging = Path(name) / "dataset"
        staging.mkdir()
        yield staging
        require(not destination.exists(), f"Output appeared while processing: {destination}")
        staging.rename(destination)


def validate_source(root, spec):
    for relative in spec["required"]:
        require((root / relative).exists(), f"Incomplete dataset: missing {root / relative}")


def extract_archive(archive, destination, spec):
    """Reject links and paths outside the archive's expected dataset directory."""
    def check(name):
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, f"Unsafe archive path: {name}")
        require(path.parts and path.parts[0] == spec["folder"], f"Unexpected archive root: {name}")

    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination) as name:
        staging = Path(name)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as source:
                for entry in source.infolist():
                    check(entry.filename)
                    require((entry.external_attr >> 16) & 0o170000 != 0o120000, "Archive contains a symlink")
                source.extractall(staging)
        else:
            with tarfile.open(archive) as source:
                for entry in source.getmembers():
                    check(entry.name)
                    require(entry.isfile() or entry.isdir(), f"Unsupported archive entry: {entry.name}")
                # Paths and entry types have been checked for Python versions without filter=.
                source.extractall(staging)
        root = staging / spec["folder"]
        validate_source(root, spec)
        require(not (destination / spec["folder"]).exists(), "Dataset folder appeared during extraction")
        root.rename(destination / spec["folder"])


def download_dataset(directory, dataset):
    spec = DATASETS[dataset]
    directory.mkdir(parents=True, exist_ok=True)
    root, archive = directory / spec["folder"], directory / spec["archive"]
    if root.exists():
        print(f"{dataset}: folder exists, no download: {root}", flush=True)
        validate_source(root, spec)
        return
    if archive.exists():
        print(f"{dataset}: archive exists, no download: {archive}", flush=True)
    else:
        partial = archive.with_name(archive.name + ".part")
        for attempt in range(3):
            try:
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "vehicles-yolo-dataset-preparation/1.0"})
                print(f"{dataset}: downloading {spec['url']} (attempt {attempt + 1}/3)", flush=True)
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                    copied = 0
                    previous = time.monotonic()
                    while chunk := response.read(4 * 1024 * 1024):
                        stream.write(chunk)
                        copied += len(chunk)
                        if time.monotonic() - previous > 10:
                            print(f"{dataset}: downloaded {copied / 1024**2:.0f} MiB", flush=True)
                            previous = time.monotonic()
                    length = response.headers.get("Content-Length")
                    require(copied > 0 and (length is None or copied == int(length)), "Incomplete download")
                require(zipfile.is_zipfile(partial) if archive.suffix == ".zip" else tarfile.is_tarfile(partial), "Response is not the expected archive")
                partial.rename(archive)
                break
            except (OSError, ValueError) as error:
                if attempt == 2:
                    raise ValueError(f"Download failed: {spec['url']}: {error}. Partial file is not a completed archive.") from error
                time.sleep(2)
    print(f"{dataset}: extracting {archive}", flush=True)
    extract_archive(archive, directory, spec)


def image_size(path):
    try:
        from PIL import Image
    except ImportError as error:
        raise ValueError("Image headers require Pillow: python3 -m pip install Pillow") from error
    # Read the actual image header without resizing, rotating or decoding pixels.
    with Image.open(path) as image:
        return image.size


def safe_filename(name):
    require(name and name not in (".", "..") and "/" not in name and "\\" not in name, f"Unsafe image filename: {name}")
    return name


def junction_group(filename):
    """Conservative source groups inferred from this archive's filename conventions."""
    stem = filename.removeprefix("aug_")
    match = re.match(r"(1112-\d+)(?:_|\.)", stem)
    if match:
        return match[1]
    if stem.startswith("fuatOld"):
        return "fuatOld"
    raise ValueError(f"Unrecognized Junction source group: {filename}. Extend junction_group() explicitly.")


def normalize_box(box, width, height):
    x1, y1, x2, y2 = box
    require(all(math.isfinite(v) for v in box), f"Non-finite box: {box}")
    require(x2 > x1 and y2 > y1, f"Non-positive box: {box}")
    clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    x1, y1, x2, y2 = clipped
    require(x2 > x1 and y2 > y1, f"Box is entirely outside {width}x{height}: {box}")
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height), clipped != box


def transfer_image(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def convert_records(dataset, image_root, annotations, csv_sizes, mapping, output, mode, dry_run, unlabelled_test=0):
    actual_names = {p.name for p in image_root.iterdir() if p.suffix.lower() == ".jpg"}
    require(actual_names == set(annotations), f"{dataset}: CSV/image mismatch; missing JPG: {sorted(set(annotations) - actual_names)[:5]}, no CSV rows: {sorted(actual_names - set(annotations))[:5]}")
    require(len({Path(name).stem for name in annotations}) == len(annotations), f"{dataset}: image stems collide in the labels directory")
    unknown = {label for rows in annotations.values() for label, _ in rows} - mapping.keys()
    require(not unknown, f"{dataset}: unknown source classes: {sorted(unknown)}")
    stats = {
        "dataset": dataset,
        "source_images": len(annotations),
        "source_objects": sum(map(len, annotations.values())),
        "images": 0,
        "empty_images": 0,
        "excluded_ambiguous_images": 0,
        "excluded_invalid_box_images": 0,
        "excluded_unlabelled_official_test_images": unlabelled_test,
        "csv_size_mismatch_images": 0,
        "clipped_boxes": 0,
        "duplicate_annotations_removed": 0,
        "class_objects": {name: 0 for name in CLASS_NAMES},
        "source_class_objects": dict(Counter(label for rows in annotations.values() for label, _ in rows)),
    }
    excluded = []
    with output_directory(output, dry_run) as staging:
        if staging:
            (staging / "images").mkdir()
            (staging / "labels").mkdir()
        records = []
        for index, (filename, rows) in enumerate(sorted(annotations.items()), 1):
            if any(mapping[label] == AMBIGUOUS for label, _ in rows):
                stats["excluded_ambiguous_images"] += 1
                excluded.append({"source_image": str(image_root / filename), "reason": "ambiguous_motorized_vehicle"})
                continue
            source = image_root / filename
            width, height = image_size(source)
            if filename in csv_sizes and csv_sizes[filename] != (width, height):
                stats["csv_size_mismatch_images"] += 1
            lines = []
            image_clipped = 0
            invalid = None
            for label, box in rows:
                target = mapping[label]
                if target is None:
                    continue
                try:
                    normalized, clipped = normalize_box(box, width, height)
                except ValueError as error:
                    invalid = str(error)
                    break
                image_clipped += int(clipped)
                lines.append(f"{target} " + " ".join(f"{v:.10f}" for v in normalized))
            if invalid:
                # Keep none of this image, so invalid target annotations do not become background.
                stats["excluded_invalid_box_images"] += 1
                excluded.append({"source_image": str(source), "reason": "invalid_target_box", "detail": invalid})
                continue
            stats["clipped_boxes"] += image_clipped
            stats["duplicate_annotations_removed"] += len(lines) - len(set(lines))
            lines = sorted(set(lines))
            for line in lines:
                stats["class_objects"][CLASS_NAMES[int(line.split()[0])]] += 1
            stats["images"] += 1
            stats["empty_images"] += int(not lines)
            record = {
                "image": f"images/{filename}",
                "label": f"labels/{Path(filename).stem}.txt",
                "source_image": str(source),
                "source_dataset": dataset,
                "group": junction_group(filename) if dataset == "junction" else Path(filename).stem,
                "augmented": dataset == "junction" and filename.startswith("aug_"),
                "width": width,
                "height": height,
            }
            if staging:
                record["sha256"] = file_hash(source)
                transfer_image(source, staging / record["image"], mode)
                (staging / record["label"]).write_text("\n".join(lines) + ("\n" if lines else ""))
                records.append(record)
            if index % 10000 == 0:
                print(f"{dataset}: checked {index}/{len(annotations)} images", flush=True)
        if staging:
            write_jsonl(staging / "manifest.jsonl", records)
            write_jsonl(staging / "excluded.jsonl", excluded)
            write_json(staging / "metadata.json", {
                "format_version": 1,
                "dataset": dataset,
                "names": CLASS_NAMES,
                "class_mapping": mapping,
                "image_storage": mode,
                "grouping": "source prefix heuristic; fuatOld kept together" if dataset == "junction" else "individual images; camera/video IDs unavailable",
            })
            write_json(staging / "summary.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return stats


def convert_junction(source, output, mode="copy", dry_run=False):
    """Convert absolute pixel boxes; CSV width/height are wrong in the public sample."""
    annotations, sizes = defaultdict(list), {}
    validate_source(source, DATASETS["junction"])
    with (source / "sample_labels.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}
        require(required <= set(reader.fieldnames or []), "Unexpected Junction CSV header")
        for row in reader:
            name = safe_filename(row["filename"])
            size = (int(row["width"]), int(row["height"]))
            require(name not in sizes or sizes[name] == size, f"Inconsistent CSV sizes for {name}")
            sizes[name] = size
            annotations[name].append((row["class"], tuple(float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return convert_records("junction", source / "sampled_images", annotations, sizes, JUNCTION_CLASS_MAP, output, mode, dry_run)


def convert_mio(source, output, mode="copy", dry_run=False):
    """Use only gt_train.csv; your_result_*.csv are not ground truth."""
    annotations = defaultdict(list)
    validate_source(source, DATASETS["mio"])
    with (source / "gt_train.csv").open(newline="", encoding="utf-8-sig") as stream:
        for number, row in enumerate(csv.reader(stream), 1):
            require(len(row) == 6, f"Unexpected MIO CSV row {number}: {row}")
            name = safe_filename(row[0] + ".jpg")
            annotations[name].append((row[1], tuple(map(float, row[2:]))))
    test = source / "test"
    unlabelled = sum(p.suffix.lower() == ".jpg" for p in test.iterdir()) if test.is_dir() else 0
    return convert_records("mio", source / "train", annotations, {}, MIO_CLASS_MAP, output, mode, dry_run, unlabelled)


def checked_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), f"Missing file or path outside dataset: {relative}")
    return path


def read_labels(path, clip_boxes=False, fixes=None):
    lines, counts = set(), [0] * len(CLASS_NAMES)
    for number, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        require(len(parts) == 5, f"{path}:{number}: expected class xc yc w h")
        category = int(parts[0])
        require(0 <= category < len(CLASS_NAMES), f"{path}:{number}: invalid class ID")
        x, y, w, h = map(float, parts[1:])
        require(all(math.isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0, f"{path}:{number}: invalid box")
        tolerance = 1e-9
        inside = min(x - w / 2, y - h / 2) >= -tolerance and max(x + w / 2, y + h / 2) <= 1 + tolerance
        if not inside and clip_boxes:
            try:
                (x, y, w, h), _ = normalize_box((x - w / 2, y - h / 2, x + w / 2, y + h / 2), 1, 1)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            if fixes is not None:
                fixes["clipped_boxes"] += 1
        else:
            require(inside, f"{path}:{number}: box outside image")
        lines.add(f"{category} " + " ".join(f"{v:.10f}" for v in (x, y, w, h)))
    for line in lines:
        counts[int(line.split()[0])] += 1
    return "\n".join(sorted(lines)) + ("\n" if lines else ""), counts


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def assign_splits(records, ratios, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["split_group"]].append(record)
    totals = Counter(r["input_id"] for r in records)
    assigned = {s: Counter() for s in SPLITS}
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda key: -len(grouped[key]))
    assignments = {}
    for key in keys:
        counts = Counter(r["input_id"] for r in grouped[key])
        scores = {}
        for split, ratio in zip(SPLITS, ratios):
            if ratio == 0:
                continue
            # Minimize the increase in squared deviation from each source's target size.
            scores[split] = sum((2 * n * (assigned[split][source] - totals[source] * ratio) + n * n) / (totals[source] * ratio) for source, n in counts.items())
        chosen = min(scores, key=scores.get)
        assignments[key] = chosen
        assigned[chosen].update(counts)
    return assignments


def validate_ratios(ratios):
    require(len(ratios) == len(SPLITS), "Expected three split ratios: train val test")
    require(all(math.isfinite(r) and r >= 0 for r in ratios) and math.isclose(sum(ratios), 1), "Split ratios must be non-negative and sum to 1")
    require(ratios[0] > 0, "Train ratio must be positive")


def save_split_dataset(records, output, metadata, summary, excluded, duplicates, mode, dry_run, required_splits):
    stats = {s: {"images": 0, "empty_images": 0, "class_objects": dict.fromkeys(CLASS_NAMES, 0), "source_images": {}} for s in SPLITS}
    paths = set()
    for record in records:
        split = record["split"]
        require(split in SPLITS, f"Invalid split: {split}")
        for kind, key in (("images", "image"), ("labels", "label")):
            relative = PurePosixPath(record[key])
            require(not relative.is_absolute() and ".." not in relative.parts and relative.parts[:2] == (kind, split), f"Invalid output path: {relative}")
            require(str(relative) not in paths, f"Output name collision: {relative}")
            paths.add(str(relative))
        count = stats[split]
        count["images"] += 1
        count["empty_images"] += int(not sum(record["counts"]))
        origin = record["source_dataset"]
        count["source_images"][origin] = count["source_images"].get(origin, 0) + 1
        for name, n in zip(CLASS_NAMES, record["counts"]):
            count["class_objects"][name] += n
    for split in required_splits:
        require(stats[split]["images"] > 0, f"Empty {split} split: need more independent groups or different ratios")
    summary = {**summary, "splits": stats, "removed_exact_duplicates": len(duplicates), "excluded_reasons": dict(Counter(r["reason"] for r in excluded))}
    with output_directory(output, dry_run) as staging:
        if staging:
            for split in SPLITS:
                (staging / "images" / split).mkdir(parents=True)
                (staging / "labels" / split).mkdir(parents=True)
            manifest = []
            for index, record in enumerate(records, 1):
                transfer_image(record["image_path"], staging / record["image"], mode)
                label = staging / record["label"]
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text(record["label_text"])
                manifest.append({k: v for k, v in record.items() if k not in ("image_path", "label_path", "label_text", "counts", "original_group")})
                if index % 10000 == 0:
                    print(f"write: {index}/{len(records)} images", flush=True)
            write_jsonl(staging / "manifest.jsonl", manifest)
            write_jsonl(staging / "excluded.jsonl", excluded)
            write_jsonl(staging / "duplicates.jsonl", duplicates)
            write_json(staging / "summary.json", summary)
            write_json(staging / "metadata.json", metadata)
            # No absolute path: moving the dataset preserves paths relative to this YAML.
            yaml = "train: images/train\nval: images/val\n"
            if "test" in required_splits or stats["test"]["images"]:
                yaml += "test: images/test\n"
            yaml += "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
            (staging / "data.yaml").write_text(yaml)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def merge_datasets(inputs, output, ratios=(0.8, 0.1, 0.1), seed=42, mode="copy", dry_run=False):
    validate_ratios(ratios)
    require(len(set(inputs)) == len(inputs), "Repeated input dataset")
    records, excluded, duplicates = [], [], []
    groups = Groups()
    by_hash = defaultdict(list)
    metadata = []
    for index, root in enumerate(inputs):
        require(not output.is_relative_to(root) and not root.is_relative_to(output), "Input and output datasets must be separate directories")
        meta = json.loads((root / "metadata.json").read_text())
        require(meta["names"] == CLASS_NAMES and meta["format_version"] == 1, f"Incompatible class IDs or manifest version: {root}")
        metadata.append({"path": str(root), "metadata": meta})
        with (root / "manifest.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                record["input_id"] = index
                record["image_path"] = checked_path(root, record["image"])
                record["label_path"] = checked_path(root, record["label"])
                record["label_text"], record["counts"] = read_labels(record["label_path"])
                digest = file_hash(record["image_path"])
                require(digest == record["sha256"], f"Image changed since conversion: {record['image_path']}")
                record["original_group"] = f"{index}:{record['group']}"
                if by_hash[digest]:
                    groups.union(record["original_group"], by_hash[digest][0]["original_group"])
                by_hash[digest].append(record)
        print(f"merge: checked {root}", flush=True)
    for digest, copies in by_hash.items():
        if len({r["label_text"] for r in copies}) != 1:
            excluded.extend({"source_image": r["source_image"], "reason": "identical_image_conflicting_labels", "sha256": digest} for r in copies)
            continue
        selected = min(copies, key=lambda r: (r["augmented"], r["input_id"], r["image"]))
        selected["split_group"] = groups.find(selected["original_group"])
        records.append(selected)
        duplicates.extend({"source_image": r["source_image"], "kept_source_image": selected["source_image"], "sha256": digest} for r in copies if r is not selected)
    assignments = assign_splits(records, ratios, seed)
    selected = []
    for record in sorted(records, key=lambda r: (r["input_id"], r["image"])):
        split = assignments[record["split_group"]]
        if split != "train" and record["augmented"]:
            excluded.append({"source_image": record["source_image"], "reason": "augmented_image_in_held_out_group", "split": split})
            continue
        stem = f"{record['input_id']:02d}_{record['image_path'].stem}"
        record.update(image=f"images/{split}/{stem}{record['image_path'].suffix}", label=f"labels/{split}/{stem}.txt", split=split)
        selected.append(record)
    return save_split_dataset(
        selected, output,
        {"format_version": 1, "names": CLASS_NAMES, "sources": metadata, "image_storage": mode},
        {"seed": seed, "requested_ratios": dict(zip(SPLITS, ratios))},
        excluded, duplicates, mode, dry_run, [s for s, r in zip(SPLITS, ratios) if r > 0],
    )


def prepare_all(directory, output, ratios=(0.8, 0.1, 0.1), seed=42, mode="copy"):
    """Run all stages, keeping converted datasets in prepared/ and the result in merged/."""
    validate_ratios(ratios)
    converted = [output / "prepared" / dataset for dataset in DATASETS]
    merged = output / "merged"
    for destination in [*converted, merged]:
        require(not destination.exists(), f"Output already exists: {destination}. Choose a new output path or run individual stages.")
    for dataset in DATASETS:
        download_dataset(directory, dataset)
    for dataset, destination in zip(DATASETS, converted):
        converter = convert_junction if dataset == "junction" else convert_mio
        converter(directory / DATASETS[dataset]["folder"], destination, mode=mode)
    result = merge_datasets(converted, merged, ratios=ratios, seed=seed, mode=mode)
    print(f"Ready: {merged / 'data.yaml'}", flush=True)
    return result


def configure_training(dataset, output, backup, darknet_labels=False):
    """Generate machine-local training paths from an already merged public dataset."""
    metadata = json.loads((dataset / "metadata.json").read_text())
    require(metadata["names"] == CLASS_NAMES, "Unexpected class order in merged metadata")
    lists, pairs, seen = {}, [], set()
    for split in SPLITS:
        images = sorted(p for p in (dataset / "images" / split).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        require(images, f"Empty split: {split}")
        labels = {p.stem: p for p in (dataset / "labels" / split).glob("*.txt")}
        require(len({p.stem for p in images}) == len(images), f"Colliding image stems in {split}")
        require(set(labels) == {p.stem for p in images}, f"Image/label mismatch in {split}")
        for image in images:
            require(image.resolve() not in seen, f"Image path appears in multiple splits: {image}")
            seen.add(image.resolve())
            label = labels[image.stem]
            read_labels(label)
            adjacent = image.with_suffix(".txt")
            if darknet_labels and adjacent.exists():
                require(adjacent.read_bytes() == label.read_bytes(), f"Conflicting adjacent label: {adjacent}")
            pairs.append((label, adjacent))
        lists[split] = "".join(f"{p.resolve()}\n" for p in images)
    output.mkdir(parents=True, exist_ok=True)
    backup.mkdir(parents=True, exist_ok=True)
    if darknet_labels:
        for label, adjacent in pairs:
            if not adjacent.exists():
                shutil.copy2(label, adjacent)
    for split, contents in lists.items():
        (output / f"vehicles-{split}.txt").write_text(contents)
    (output / "vehicles.names").write_text("\n".join(CLASS_NAMES) + "\n")
    for suffix, split in (("", "val"), ("-test", "test")):
        (output / f"vehicles{suffix}.data").write_text(
            f"classes = {len(CLASS_NAMES)}\ntrain = {output}/vehicles-train.txt\n"
            f"valid = {output}/vehicles-{split}.txt\nnames = {output}/vehicles.names\nbackup = {backup}\n"
        )
    yaml = f"path: {json.dumps(str(dataset))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    (output / "vehicles.yaml").write_text(yaml)
    print(f"Training config: {output / 'vehicles.yaml'}", flush=True)
    print("Images: " + ", ".join(f"{s}={len(v.splitlines())}" for s, v in lists.items()), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download missing archives and extract them")
    convert = commands.add_parser("convert", help="Convert original annotations to our four YOLO classes")
    merge = commands.add_parser("merge", help="Merge converted datasets and create train/val/test")
    all_stages = commands.add_parser("all", help="Download both datasets, convert and merge in one run")
    configure = commands.add_parser("configure", help="Write training paths for an existing merged dataset")
    configure.add_argument("--dataset", type=Path, required=True, help="Merged root with images, labels and metadata.json")
    configure.add_argument("--output", type=Path, default=Path("data/generated"), help="Generated training files (default: data/generated)")
    configure.add_argument("--backup", type=Path, default=Path("weights"))
    configure.add_argument("--darknet-labels", action="store_true", help="Also copy TXT beside JPG for this Darknet fork")
    all_stages.add_argument("--training-dir", type=Path, help="Also generate training configs and adjacent Darknet labels; use data/generated for the training scripts")
    all_stages.add_argument("--backup", type=Path, default=Path("weights"))
    for command in (download, convert, all_stages):
        command.add_argument("--directory", type=Path, required=True, help="Directory containing source dataset folders/archives")
    for command in (download, convert):
        command.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    for command in (convert, merge):
        command.add_argument("--output", type=Path, required=True, help="New output directory; existing data are never overwritten")
        command.add_argument("--dry-run", action="store_true", help="Read and validate inputs, report counts, write nothing")
    all_stages.add_argument("--output", type=Path, required=True, help="Working directory for prepared/junction, prepared/mio and merged")
    for command in (convert, merge, all_stages):
        command.add_argument("--mode", choices=["copy", "hardlink"], default="copy", help="copy is independent; hardlink requires same filesystem")
    merge.add_argument("--inputs", nargs="+", type=Path, required=True, help="Converted roots containing images, labels and manifest.jsonl")
    for command in (merge, all_stages):
        command.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
        command.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command == "all":
            prepare_all(args.directory.expanduser().resolve(), args.output.expanduser().resolve(), args.ratios, args.seed, args.mode)
            if args.training_dir:
                configure_training(args.output.expanduser().resolve() / "merged", args.training_dir.expanduser().resolve(), args.backup.expanduser().resolve(), True)
        elif args.command == "configure":
            configure_training(args.dataset.expanduser().resolve(), args.output.expanduser().resolve(), args.backup.expanduser().resolve(), args.darknet_labels)
        elif args.command == "merge":
            merge_datasets([p.expanduser().resolve() for p in args.inputs], args.output.expanduser().resolve(), args.ratios, args.seed, args.mode, args.dry_run)
        else:
            directory = args.directory.expanduser().resolve()
            for dataset in DATASETS if args.dataset == "all" else [args.dataset]:
                if args.command == "download":
                    download_dataset(directory, dataset)
                else:
                    converter = convert_junction if dataset == "junction" else convert_mio
                    converter(directory / DATASETS[dataset]["folder"], args.output.expanduser().resolve() / dataset, args.mode, args.dry_run)
    except (OSError, ValueError, KeyError, csv.Error, zipfile.BadZipFile, tarfile.TarError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
