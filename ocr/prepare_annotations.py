#!/usr/bin/env python3
"""Validate automatic character annotations, group crops and package YOLO splits.

Requires Pillow. This script runs no models and changes no source files.
Background labels require explicit confirmation with --confirmed-backgrounds.
The output is a new directory; existing output directories are never overwritten.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import zipfile

from PIL import Image

from annotate_crops import CLASSES as ANNOTATOR_CLASSES, EXTENSIONS, label_text, write_json


SPLITS = ("train", "val", "test")
# Reserve D as ID 22 even though the current annotations have no examples of it.
CLASSES = [*ANNOTATOR_CLASSES, "D"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_label(path):
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        require(len(fields) == 5, f"{path}:{number}: expected class xc yc w h")
        category = int(fields[0])
        box = tuple(map(float, fields[1:]))
        x, y, w, h = box
        require(0 <= category < len(CLASSES), f"{path}:{number}: invalid class")
        require(all(math.isfinite(v) for v in box) and w > 0 and h > 0,
                f"{path}:{number}: invalid coordinates")
        require(min(x - w / 2, y - h / 2) >= -1e-9 and max(x + w / 2, y + h / 2) <= 1 + 1e-9,
                f"{path}:{number}: box outside image")
        rows.append(f"{category} " + " ".join(f"{v:.10f}" for v in box))
    require(len(rows) == len(set(rows)), f"{path}: repeated identical annotations")
    return "\n".join(sorted(rows)) + ("\n" if rows else "")


def features(image):
    gray = image.convert("L")
    small = gray.resize((9, 8), Image.Resampling.BILINEAR).tobytes()
    fingerprint = 0
    for y in range(8):
        for x in range(8):
            fingerprint = (fingerprint << 1) | (small[y * 9 + x] > small[y * 9 + x + 1])
    pixels = gray.resize((16, 16), Image.Resampling.BILINEAR).tobytes()
    return fingerprint, pixels


class Groups:
    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        self.parent[max(left, right)] = min(left, right)


def group_records(records):
    groups = Groups(len(records))
    texts, pixels = {}, {}
    # Five disjoint bands ensure hashes at Hamming distance <= 4 share a band.
    bands = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))
    buckets = defaultdict(set)
    hashes = defaultdict(list)
    text_links, visual_links = 0, 0
    for index, record in enumerate(records):
        pixel_hash = record["pixel_sha256"]
        if pixel_hash in pixels:
            groups.union(index, pixels[pixel_hash])
        else:
            pixels[pixel_hash] = index
        text = record["predicted_text"]
        if 6 <= len(text) <= 12:
            if text in texts:
                groups.union(index, texts[text])
                text_links += 1
            else:
                texts[text] = index
        fingerprint = record["dhash"]
        candidates = set()
        for band, (offset, bits) in enumerate(bands):
            key = (band, (fingerprint >> offset) & ((1 << bits) - 1))
            candidates.update(buckets[key])
        for other_hash in sorted(candidates):
            if (fingerprint ^ other_hash).bit_count() > 4:
                continue
            for other in hashes[other_hash]:
                previous = records[other]
                aspect_ratio = record["aspect"] / previous["aspect"]
                if not 0.9 <= aspect_ratio <= 1 / 0.9:
                    continue
                difference = sum(abs(a - b) for a, b in zip(record["thumbnail"], previous["thumbnail"])) / 256
                if difference <= 12:
                    groups.union(index, other)
                    visual_links += 1
        # Every image remains indexed, so later matches can connect entire groups.
        hashes[fingerprint].append(index)
        for band, (offset, bits) in enumerate(bands):
            buckets[(band, (fingerprint >> offset) & ((1 << bits) - 1))].add(fingerprint)
        if (index + 1) % 10000 == 0:
            print(f"Grouped {index + 1}/{len(records)}", flush=True)
    for index, record in enumerate(records):
        record["group"] = groups.find(index)
    return {"same_text_links": text_links, "near_image_links": visual_links}


def split_records(records, val_size, test_size, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    targets = (len(records) - val_size - test_size, val_size, test_size)
    require(min(targets) > 0, "Not enough images for the requested splits")
    totals = [sum(record["counts"][i] for record in records) for i in range(len(CLASSES) + 1)]
    assigned_images = [0, 0, 0]
    assigned_objects = [[0] * len(totals) for _ in SPLITS]
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda key: -len(grouped[key]))
    for key in keys:
        batch = grouped[key]
        n = len(batch)
        counts = [sum(record["counts"][i] for record in batch) for i in range(len(totals))]
        scores = []
        for split, target in enumerate(targets):
            score = (2 * n * (assigned_images[split] - target) + n * n) / target
            for i, count in enumerate(counts):
                expected = totals[i] * target / len(records)
                if expected:
                    score += 0.3 / sum(total > 0 for total in totals) * (2 * count * (assigned_objects[split][i] - expected) + count * count) / expected
            scores.append(score)
        # Prefer groups that fit while preserving the requested image counts.
        choices = [i for i, target in enumerate(targets) if assigned_images[i] + n <= target]
        chosen = min(choices or range(3), key=lambda i: scores[i])
        assigned_images[chosen] += n
        for i, count in enumerate(counts):
            assigned_objects[chosen][i] += count
        for record in batch:
            record["split"] = SPLITS[chosen]
    for i, split in enumerate(SPLITS):
        require(assigned_images[i] > 0 and all(assigned_objects[i][j] > 0 for j, total in enumerate(totals[:-1]) if total),
                f"{split}: empty split or a missing represented character class")
    return {"groups": len(grouped), "largest_group": max(map(len, grouped.values()))}


def prepare(args):
    images_root, annotations, output = args.images.resolve(), args.annotations.resolve(), args.output.resolve()
    require(not output.exists(), f"Output already exists: {output}")
    for source in (images_root, annotations):
        require(output != source and source not in output.parents and output not in source.parents,
                "Source and output must be separate directories")
    require((annotations / "classes.names").read_text().split() == ANNOTATOR_CLASSES,
            "Expected the 22 source annotation classes; output additionally reserves D")
    source_summary = json.loads((annotations / "summary.json").read_text())
    require(source_summary["complete"] is True, "Annotation run is incomplete")
    images = {p.name: p for p in images_root.iterdir()
              if p.is_file() and not p.is_symlink() and p.suffix.lower() in EXTENSIONS}
    require(len({p.stem for p in images.values()}) == len(images), "Image stems collide")
    predictions = {}
    with (annotations / "predictions.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            name = record["image"]
            require(name in images and name not in predictions, f"Unknown or repeated image: {name}")
            predictions[name] = record
    require(set(predictions) == set(images), "Image/journal mismatch")
    require(len(images) == source_summary["processed"], "Summary/journal count mismatch")
    expected_labels = {Path(name).stem + ".txt" for name, record in predictions.items() if record["status"] == "needs_review"}
    require({p.name for p in (annotations / "labels").iterdir()} == expected_labels, "Unexpected or missing annotation files")
    records, exact = [], defaultdict(list)
    for index, name in enumerate(sorted(images), 1):
        record, path = predictions[name], images[name]
        characters = record["characters"]
        require(record["status"] in ("needs_review", "no_detections"), f"Unusable image: {name}")
        with Image.open(path) as original:
            image = original.convert("RGB")
        require(list(image.size) == record["size_wh"], f"Image size changed: {name}")
        if record["status"] == "no_detections":
            require(args.confirmed_backgrounds and not characters,
                    "Background candidates need visual confirmation and --confirmed-backgrounds")
            text = ""
        else:
            require(bool(characters), f"Missing detections: {name}")
            for item in characters:
                require(type(item["class_id"]) is int and 0 <= item["class_id"] < len(ANNOTATOR_CLASSES), f"Invalid source class: {name}")
                require(item["char"] == CLASSES[item["class_id"]], f"Character/class mismatch: {name}")
                require(math.isfinite(item["confidence"]) and 0 <= item["confidence"] <= 1, f"Invalid confidence: {name}")
            text = read_label(annotations / "labels" / (path.stem + ".txt"))
            expected = "\n".join(sorted(label_text(characters, *image.size).splitlines())) + "\n"
            require(text == expected, f"Label/journal mismatch: {name}")
        counts = Counter(item["class_id"] for item in characters)
        fingerprint, thumbnail = features(image)
        pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        flags = []
        if characters and not 6 <= len(characters) <= 10:
            flags.append("unusual_character_count")
        if characters and min(item["confidence"] for item in characters) < 0.8:
            flags.append("confidence_below_0.8")
        entry = {"source": path, "name": name, "label": text, "pixel_sha256": pixel_hash,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "width": image.width, "height": image.height,
                 "dhash": fingerprint, "thumbnail": thumbnail, "aspect": image.width / image.height,
                 "predicted_text": "".join(item["char"] for item in sorted(characters, key=lambda item: item["bbox"][0])),
                 "counts": [counts[i] for i in range(len(CLASSES))] + [int(not characters)], "review_flags": flags}
        exact[pixel_hash].append(entry)
        if index % 10000 == 0:
            print(f"Validated {index}/{len(images)}", flush=True)
    conflicts, duplicates = [], []
    for copies in exact.values():
        if len({record["label"] for record in copies}) != 1:
            conflicts.extend({"image": r["name"], "reason": "identical_pixels_conflicting_labels"} for r in copies)
            for record in copies:
                record["review_flags"].append("identical_pixels_conflicting_labels")
        records.extend(copies)
        duplicates.extend({"image": r["name"], "same_pixels_as": copies[0]["name"]} for r in copies[1:])
    records.sort(key=lambda record: record["name"])
    grouping = group_records(records)
    grouping.update(split_records(records, args.val_size, args.test_size, args.seed))
    stats = {}
    for split in SPLITS:
        selected = [r for r in records if r["split"] == split]
        stats[split] = {"images": len(selected), "empty_images": sum(r["counts"][-1] for r in selected),
                        "class_objects": {name: sum(r["counts"][i] for r in selected) for i, name in enumerate(CLASSES)},
                        "flagged_for_review": sum(bool(r["review_flags"]) for r in selected)}
    summary = {"source_images": len(images), "images": len(records), "classes": CLASSES, "seed": args.seed,
               "annotation_method": "automatic, with visual spot checks; not fully manually verified",
               "backgrounds_confirmed_by_owner": args.confirmed_backgrounds,
               "removed_images": 0, "exact_duplicate_images_retained": len(duplicates),
               "conflicting_images_retained": len(conflicts),
               "grouping": grouping, "splits": stats}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ocr-prepare-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for split in SPLITS:
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        with (staging / "manifest.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "label", "group", "width", "height", "sha256", "review_flags"])
            for index, record in enumerate(records, 1):
                split = record["split"]
                image = f"images/{split}/{record['name']}"
                label = f"labels/{split}/{Path(record['name']).stem}.txt"
                shutil.copy2(record["source"], staging / image)
                (staging / label).write_text(record["label"])
                writer.writerow([image, label, record["group"], record["width"], record["height"], record["sha256"], ";".join(record["review_flags"])])
                if index % 10000 == 0:
                    print(f"Written {index}/{len(records)}", flush=True)
        write_json(staging / "summary.json", summary)
        write_json(staging / "duplicates.json", duplicates)
        write_json(staging / "conflicts.json", conflicts)
        (staging / "classes.names").write_text("\n".join(CLASSES) + "\n")
        (staging / "data.yaml").write_text(f"train: images/train\nval: images/val\ntest: images/test\nnc: {len(CLASSES)}\nnames:\n" + "".join(f"  {i}: '{name}'\n" for i, name in enumerate(CLASSES)))
        if args.readme:
            shutil.copy2(args.readme, staging / "README.md")
        require(not output.exists(), f"Output appeared during preparation: {output}")
        staging.rename(output)
    print(json.dumps(summary, indent=2), flush=True)
    return output


def description(summary):
    rows = "\n".join(f"| {split} | {stats['images']:,} | {stats['empty_images']} | {sum(stats['class_objects'].values()):,} |" for split, stats in summary["splits"].items())
    return f"""# Russian license plate characters: 23 classes

Cropped Russian license plates with individual character bounding boxes in YOLO detection format. The dataset contains {summary['images']:,} images and uses original image dimensions. This is character detection data, not a collection of full-frame vehicle images or plate-level transcription labels.

## Classes

IDs 0 through 22 follow this order: `0 1 2 3 4 5 6 7 8 9 A B C E H K M O P T X Y D`. Letters use Latin lookalikes. Diplomatic `D` is reserved as class 22, with zero examples in this version. When I have more time I will try to add examples of diplomatic plate numbers, but not sure when it will be possible

## Annotations

Annotations were created using [yolo-ann](https://github.com/LdDl/yolo-ann) and a [pretrained model](https://github.com/LdDl/license_plate_recognition/releases/tag/v1.4.0). They were also visually spot-checked, but there could be some mistakes because I am just a meatbag. They have not been exhaustively corrected by hand. Incorrect classes, missed characters and imperfect boxes may remain. Validation and test annotations use the same automatic process; scores against them measure agreement with these labels and should not be presented as accuracy against fully human-verified ground truth.

Background candidates were inspected by the dataset owner before empty annotations were added. Every image has a TXT file, including zero-byte files for backgrounds. Each nonempty row contains `class_id x_center y_center width height`, with coordinates normalized to the original image dimensions. Confidence scores are not included in training labels. Images with unusual detection counts or low-confidence predictions remain in the dataset and are marked in `manifest.csv` for review.

## Splits

| Split | Images | Backgrounds | Character boxes |
| --- | ---: | ---: | ---: |
{rows}

The split uses seed {summary['seed']}. Images sharing the same automatically predicted left-to-right character sequence of 6 to 12 symbols are grouped. Visually close crops are also grouped using a 64-bit difference hash (Hamming distance <= 4), aspect ratios within roughly 10% and grayscale thumbnail mean absolute difference <= 12. Entire connected groups stay within one split. Class counts and image counts guide group assignment.

Grouping is a heuristic, not verified vehicle identity or camera metadata. Incorrect OCR and substantial viewpoint changes can leave related images in separate groups; unrelated images may also be grouped together. The split reduces obvious leakage but cannot guarantee independence by vehicle, recording or camera. Predictions used for grouping are not verified plate transcriptions, especially for multi-row plates.

## Files

- `images/train`, `images/val`, `images/test`: original crops.
- `labels/train`, `labels/val`, `labels/test`: matching YOLO TXT files.
- `data.yaml`: relative split paths and the class mapping. Set an absolute dataset root with `path` in a local copy if required by your training tool.
- `classes.names`: class names in ID order.
- `summary.json`: split statistics and preparation details.
- `manifest.csv`: image/label paths, group IDs, dimensions, file hashes and review flags.
- `conflicts.json`: conflicting annotation records, if any.
"""


def package(output, archive):
    require(not archive.exists(), f"Archive already exists: {archive}")
    require(output != archive and output not in archive.parents, "Archive must be outside the dataset")
    with tempfile.TemporaryDirectory(prefix=".ocr-zip-", dir=archive.parent) as temporary:
        packed = Path(temporary) / "dataset.zip"
        with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    compression = zipfile.ZIP_STORED if path.suffix.lower() in EXTENSIONS else zipfile.ZIP_DEFLATED
                    stream.write(path, path.relative_to(output), compress_type=compression)
        with zipfile.ZipFile(packed) as stream:
            require(stream.testzip() is None, "Archive integrity check failed")
        require(not archive.exists(), f"Archive appeared during preparation: {archive}")
        packed.rename(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(f"Archive: {archive}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--readme", type=Path, help="Include this explicitly approved README; omitted by default")
    parser.add_argument("--confirmed-backgrounds", action="store_true")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    require(min(args.val_size, args.test_size) > 0, "Validation and test sizes must be positive")
    if args.readme:
        require(args.readme.is_file(), f"README not found: {args.readme}")
    if args.archive:
        require(not args.archive.exists(), f"Archive already exists: {args.archive}")
    output = prepare(args)
    if args.archive:
        package(output, args.archive.resolve())


if __name__ == "__main__":
    main()
