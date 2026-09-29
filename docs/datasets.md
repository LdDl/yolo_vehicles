# Preparing Junction and MIO-TCD

[prepare_dataset.py](../scripts/prepare_dataset.py) runs the full workflow with `all`, or individual stages with `download`, `convert`, and `merge`. Two conversion functions, `convert_junction()` and `convert_mio()`, read the source CSV files and create YOLO TXT annotations for my four classes. Images keep their original dimensions and contents. The output uses separate `images/` and `labels/` directories.

This is the project's main dataset preparation workflow. `configure` generates local paths for Darknet and Ultralytics; `--darknet-labels` also places TXT copies next to JPG files while keeping the original `labels/` directory.

## Table of contents

- [Sources](#sources)
- [Setup](#setup)
- [All stages in one run](#all-stages-in-one-run)
- [Manual stages](#manual-stages)
  - [1. Download and extract](#1-download-and-extract)
  - [2. Convert annotations](#2-convert-annotations)
  - [3. Merge and create train/val/test splits](#3-merge-and-create-trainvaltest-splits)
  - [4. Configure training files](#4-configure-training-files)
- [What you can delete after merging](#what-you-can-delete-after-merging)
- [My four classes](#my-four-classes)
- [Annotation details](#annotation-details)
  - [Junction](#junction)
  - [MIO-TCD localization](#mio-tcd-localization)
- [Splitting and duplicates](#splitting-and-duplicates)
- [Local archive check results](#local-archive-check-results)

## Sources

| Dataset | Page | Archive |
| :--- | :--- | :--- |
| Junction-based Vehicle Detection Dataset | [Mendeley, version 1](https://data.mendeley.com/datasets/vwjg6b7kpt/1) | [ZIP](https://data.mendeley.com/public-api/zip/vwjg6b7kpt/download/1) |
| MIO-TCD Localization | [Official page](https://tcd.miovision.com/challenge/dataset.html) | [TAR](https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar) |

BTW, MIO's original `README.txt` specifies CC BY-NC-SA 4.0, which restricts commercial use. Annotation conversion does not change the source data license. So be aware that the merged dataset is not suitable for commercial use.

## Setup

My setup requires Python 3.10+. You also need Pillow to read image headers. So here is [requirements-data.txt](../requirements-data.txt) for this. Thankfully dataset preparation needs no GPU or OpenCV.

```bash
python3 -m venv .venv-data
source .venv-data/bin/activate
python3 -m pip install -r requirements-data.txt
```

> **Note:** don't forget to deactivate the virtual environment when done.

## All stages in one run

After installing dependencies, run from the project root:

```bash
python3 scripts/prepare_dataset.py all \
  --directory "$PWD/datasets/raw" \
  --output "$PWD/datasets/vehicles" \
  --training-dir "$PWD/data"
```

This basically downloads and extracts both datasets, converts them into `datasets/vehicles/prepared/junction` and `datasets/vehicles/prepared/mio`, then creates `datasets/vehicles/merged/data.yaml` and the matching image and label directories. Existing source archives or folders are reused without downloading again.

For local source files:

```bash
python3 scripts/prepare_dataset.py all \
  --directory datasets/raw \
  --output datasets/vehicles
```

For `all`, `--output` sets the working directory containing `prepared/` and `merged/`. Defaults are an 80/10/10 split, `seed=42`, and copying images. `--ratios`, `--seed`, and `--mode hardlink` work as they do for individual stages.

Before starting the script checks that `prepared/junction`, `prepared/mio`, and `merged` do not already exist.

If some stages are complete, continue with the commands below or choose a new working directory with `--output`. Use `--dry-run` with `convert` or `merge` to check inputs without writing output.

## Manual stages

<details>
<summary>Optional: run stages separately</summary>

### 1. Download and extract

```bash
python3 scripts/prepare_dataset.py download \
  --directory datasets/raw
```

- If the dataset folder exists, the script checks for the required CSV and image directory. No network requests are made.
- If only the archive exists, the script extracts it without downloading again.
- If neither exists, the script downloads to a temporary `.part` file, checks the archive format, and extracts it.
- An incomplete existing folder or damaged existing archive causes an error. The script does not replace it with a new download.

Skip this stage if both datasets are already extracted. To process one dataset, add `--dataset junction` or `--dataset mio`.

Expected sources:

```text
datasets/raw/
  Junction-based Vehicle Detection Dataset/
    sample_labels.csv
    sampled_images/*.jpg
  MIO-TCD-Localization/
    gt_train.csv
    train/*.jpg
    test/*.jpg
    your_result_train.csv
    your_result_test.csv
```

### 2. Convert annotations

```bash
python3 scripts/prepare_dataset.py convert \
  --directory datasets/raw \
  --output datasets/vehicles/prepared
```

Add `--dry-run` to check inputs without writing files. Select one dataset with `--dataset junction` or `--dataset mio`.

```text
datasets/vehicles/prepared/
  junction/
    images/*.jpg
    labels/*.txt
    metadata.json
    manifest.jsonl
    excluded.jsonl
    summary.json
  mio/
    images/*.jpg
    labels/*.txt
    metadata.json
    manifest.jsonl
    excluded.jsonl
    summary.json
```

`metadata.json` records class order and mapping. `manifest.jsonl` records each image's source, source group, dimensions, and SHA-256. `excluded.jsonl` lists excluded images and reasons. `summary.json` contains image and object counts.

### 3. Merge and create train/val/test splits

```bash
python3 scripts/prepare_dataset.py merge \
  --inputs datasets/vehicles/prepared/junction datasets/vehicles/prepared/mio \
  --output datasets/vehicles/merged \
  --ratios 0.8 0.1 0.1 \
  --seed 42
```

`merge` accepts directories with YOLO annotations and manifests generated by `convert`. It checks classes, coordinates, and image hashes, then merges the prepared annotations. `--dry-run` prints merge statistics without writing the result.

```text
datasets/vehicles/merged/
  images/
    train/*.jpg
    val/*.jpg
    test/*.jpg
  labels/
    train/*.txt
    val/*.txt
    test/*.txt
  data.yaml
  metadata.json
  manifest.jsonl
  excluded.jsonl
  duplicates.jsonl
  summary.json
```

Filenames are prefixed with the input dataset index, for example `00_123.jpg` and `01_123.jpg`. Matching TXT files use the same stems. Changing the order of `--inputs` changes the prefixes and may change the split.

Paths in `data.yaml` are relative to its directory. Pass this YAML to Ultralytics and keep the entire `merged/` structure when moving the dataset. This path resolution is supported by the [Ultralytics dataset loader](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py).

By default, images are copied and TXT files are written from scratch. To save space, add `--mode hardlink` to `convert` and `merge` when sources and output are on the same filesystem. The JPG files then share their contents with the sources: editing a JPG through any hard link also changes the original. Use the default `copy` mode if you plan to edit images.

Existing output directories are never overwritten. For a new class mapping or split, use new `--output` paths, such as `datasets/vehicles/prepared-v2` and `datasets/vehicles/merged-v2`.

### 4. Configure training files

```bash
python3 scripts/prepare_dataset.py configure \
  --dataset datasets/vehicles/merged \
  --output data --backup weights --darknet-labels
```

This writes absolute image lists to `data/vehicles-{train,val,test}.txt`, plus `data/vehicles.yaml`, `data/vehicles.data`, `data/vehicles-test.data`, and `data/vehicles.names`. It checks image/TXT pairs, normalized boxes, and that resolved file paths do not overlap between splits. This checks paths; it does not search again for similar images.

`--darknet-labels` places TXT copies next to JPG files for my Darknet fork. If a neighboring TXT already exists with different contents, the command stops. The main annotations remain in `labels/`; Ultralytics and the evaluator read them directly. If the dataset path changes, run `configure` again with the new path.

</details>

## What you can delete after merging

After `merge` or `all` completes successfully, training only needs the ENTIRE `datasets/vehicles/merged/` folder, including `images/`, `labels/`, `data.yaml`, and reports.

| Directory | What you can delete |
| :--- | :--- |
| `datasets/raw/` | Source archives and extracted Junction and MIO-TCD folders. Delete the whole directory only if it contains sources for these two datasets and nothing else. |
| `datasets/vehicles/prepared/` | Intermediate datasets with converted annotations. |

In the default `copy` mode, the final `merged/` contains its own JPG and TXT copies. With `--mode hardlink`, deleting sources and `prepared/` also preserves images in `merged/`: the remaining hard links still reference the file contents. Source paths in JSON reports record where images came from and are not required for training.

If the sources are in a shared Downloads folder, delete only these two datasets' archives and directories. Converting again after deleting sources requires downloading them again; merging again after deleting `prepared/` requires repeating conversion.

## My four classes

`JUNCTION_CLASS_MAP` and `MIO_CLASS_MAP` are at the top of the script. They define how source categories are grouped in this project; adjust them before conversion if needed.

| ID | My class | Junction | MIO-TCD |
| :--- | :--- | :--- | :--- |
| 0 | Car (`car`) | `car`, `taksi`, `van` | `car`, `work_van` |
| 1 | Motorcycle (`motorbike`) | `motorcycle` | `motorcycle` |
| 2 | Bus (`bus`) | `bus`, `minibus` | `bus` |
| 3 | Truck (`truck`) | `light_truck`, `heavy_truck` | `pickup_truck`, `single_unit_truck`, `articulated_truck` |

Here are some notes on the source categories and my mapping:

- `taksi` is the spelling used in the Junction CSV though.
- My class ID 1 corresponds to the COCO `motorcycle` category; I prefer to call it `motorbike`.
- `person`, `pedestrian`, `bicycle`, and `non-motorized_vehicle` annotations are removed. Images with other target objects keep those annotations. If no target objects remain, the converter writes an empty TXT: the image is background for my four classes.
- MIO's `motorized_vehicle` category means a vehicle whose specific category could not be determined. It maps to `AMBIGUOUS`, so any image containing it is excluded entirely. This avoids training on visible vehicles whose target annotations would otherwise be removed. Unknown source categories cause an error.

## Annotation details

### Junction

The downloaded public archive contains 3952 images and 9685 annotation rows. This is the `sampled_images` / `sample_labels.csv` subset, not the full dataset of tens of thousands of images described in the paper. A total of 3 370 filenames start with `aug_`.

The CSV header is `filename,width,height,class,xmin,ymin,xmax,ymax`. Boxes use absolute pixel coordinates. In this archive version, CSV `width` and `height` are incorrect for every image: they report 1280x720 or 720x480 instead of the actual 800x450 or 800x533. On inspected examples, boxes match the actual JPG files. The converter normalizes coordinates using dimensions from JPG headers, without first resizing images or coordinates. Using the CSV dimensions would produce incorrect boxes.

Geometry checks do not assess class correctness, annotation completeness, or the effect of black masks on individual objects.

### MIO-TCD localization

`gt_train.csv` has no header: `image_id,class,xmin,ymin,xmax,ymax`. The converter appends `.jpg` to the ID, preserving leading zeros. Dimensions come from JPG headers.

The training subset contains 110000 images and 351549 objects. The official `test` folder contains another 27743 images without ground-truth annotations though. According to the source `README.txt`, `your_result_train.csv` and `your_result_test.csv` contain random class assignments to demonstrate the results format. Conversion uses only `gt_train.csv`.

My validation (`val`) and test (`test`) splits come from the annotated training (`train`) subset. They are not the official MIO-TCD challenge test set.

For both datasets, boxes crossing an image edge are clipped to its boundaries. An image with a target box of zero area, non-finite coordinates, or a box entirely outside the image is excluded, with the reason recorded. Duplicate annotations within a TXT file are removed.

## Splitting and duplicates

- Junction images are grouped by `1112-<number>` prefixes. All `fuatOld` variants stay in one conservative group. The `aug_` prefix and extra suffixes do not create separate source groups.
- Each group goes entirely into one split. Images starting with `aug_` are additionally excluded from groups assigned to `val` or `test`; they are not moved back into `train`.
- MIO filenames contain no confirmed camera or video IDs. Splitting uses individual images and a fixed `seed`, so evaluation on unseen cameras is not guaranteed.
- Byte-identical JPG files are detected using SHA-256 before splitting. If annotations match, one copy remains. If annotations differ, all copies are excluded and their source groups are linked so related frames cannot end up in different splits.
- Conflict handling is conservative: even small coordinate differences count as conflicts. In inspected MIO groups, both box coordinates and sometimes the number of annotated objects differ. The script does not combine or choose between these annotations automatically.
- SHA-256 detects identical files. It does not detect similar frames or the same image with different JPEG compression.
- The algorithm approximates the requested ratios separately for each input dataset while keeping groups together. Group sizes and augmentation exclusions can shift the final ratio away from 80/10/10. Classes are not automatically balanced.

Evaluation on unseen cameras requires an additional independent dataset. Junction has few images in the current validation and test splits because it contains many augmentations and few original source groups.

## Local archive check results

Checked on September 15, 2026 with the mappings above. Counts after conversion, before image deduplication and splitting:

| Dataset | Images | Empty TXT | Cars | Motorcycles | Buses | Trucks |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Junction | 3 952 | 276 | 3 520 | 802 | 1 864 | 1 695 |
| MIO-TCD | 89 697 | 562 | 184 354 | 1 557 | 8 222 | 46 944 |

MIO: 20299 images containing `motorized_vehicle` and four images with invalid target boxes were excluded. Junction: one duplicate annotation was removed.

Merging with input order `junction mio`, ratios `0.8 0.1 0.1`, and `--seed 42` produces 77503 images:

| Split | Images | From Junction | From MIO | Cars | Motorcycles | Buses | Trucks |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training (`train`) | 62 587 | 3 087 | 59 500 | 133 375 | 1 369 | 7 542 | 27 285 |
| Validation (`val`) | 7 456 | 18 | 7 438 | 16 371 | 89 | 776 | 3 295 |
| Test (`test`) | 7 460 | 23 | 7 437 | 16 454 | 87 | 793 | 3 238 |

The merge removed 12 extra copies with matching annotations, excluded 15 310 files from groups of identical JPGs with conflicting annotations, and excluded 824 augmented images from validation/test groups. Here, 15310 counts file copies, not unique scenes. Each new run saves its details in JSON reports alongside the result.

The full validation run used temporary directories and hard links. It checked JPG/TXT pairs, unique hashes, and that groups did not overlap between splits. Temporary output was removed after checking; the commands above create the permanent `datasets/vehicles/prepared` and `datasets/vehicles/merged` directories.

Tests cover conversion, reusing existing downloads, extraction, annotation conflicts, unique filenames, and reproducible splitting:

```bash
python3 -m unittest discover -s tests -p 'test_prepare_dataset.py' -v
```
