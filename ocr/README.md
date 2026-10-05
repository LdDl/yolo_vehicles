# License plate character detection

The third stage of the vehicle -> plate -> OCR cascade: detect individual characters inside a cropped license plate. The initial comparison uses YOLOv3-tiny, YOLOv4-tiny, YOLOv5nu, YOLOv8n, YOLOv9t and YOLO11n at **224x64**, width x height, with letterbox. Full-number decoding, row ordering and transcription metrics are not implemented yet.

Dataset on Kaggle: my [Russian license plate characters: 23 classes](https://www.kaggle.com/datasets/dimahkiin/russian-license-plate-characters-23-classes). It contains plate crops with individual character boxes, including partial plates, multiple visible plates and backgrounds.

## Table of contents

- [Model settings](#model-settings)
- [Dataset and splits](#dataset-and-splits)
- [Download and prepare](#download-and-prepare)
- [Train models](#train-models)
  - [Darknet](#darknet)
  - [Ultralytics](#ultralytics)
- [Export to ONNX](#export-to-onnx)
- [Benchmark](#benchmark)
- [Benchmark results](#benchmark-results)
- [Output files](#output-files)

## Model settings

The initial alphabet has 23 classes. [classes.names](classes.names) defines the ID order, starting at zero:

```text
0 1 2 3 4 5 6 7 8 9 A B C E H K M O P T X Y D
```

Letters use Latin characters. `D` is class 22, reserved for diplomatic plates. The published v1 dataset has no D examples, but the class stays in the dataset YAML, Darknet heads, exported models and benchmark. No D recognition quality can be measured from this version. A change to the alphabet requires matching changes to the annotations, Darknet classes/filters, training configuration and benchmark class list.

Darknet uses 23 classes and 84 filters before each detection layer: `(23 + 5) * 3`. Both training and inference configs use 224x64. Letterbox preserves each crop's proportions. Flipping, mosaic, crop jitter and random input resizing are disabled. Anchors are provisional character-sized values, not fitted to annotations. Batch 64, subdivisions 4 and 46000 iterations with learning-rate drops at 36800/41400 are starting settings, not measured optimal values.

Ultralytics training uses `imgsz=224` and `rect=True`; batch shapes vary with the image proportions and are not fixed at 224x64. Export fixes the input to `[1,3,64,224]`. Flipping, mosaic, mixup, rotation, shear, perspective and hue/saturation augmentation are disabled for this task.

## Dataset and splits

The published v1 dataset contains 85,455 images:

| Split | Images | Background images | Character boxes |
| --- | ---: | ---: | ---: |
| Train | 75,455 | 123 | 591,323 |
| Val | 5,000 | 8 | 39,169 |
| Test | 5,000 | 8 | 39,173 |

Every image has a matching YOLO TXT file, including empty files for the 139 backgrounds. Each row is `class_id x_center y_center width height`, normalized to the original image dimensions. The published groups and split are preserved during preparation. Related crops were grouped using recognized character sequences and image similarity; this is a heuristic, not a guarantee of independent cameras or vehicles.

Annotations were created using [yolo-ann](https://github.com/LdDl/yolo-ann) and a [pretrained recognition model](https://github.com/LdDl/license_plate_recognition/releases/tag/v1.4.0), followed by visual spot checks. The validation and test labels also contain automatic annotations. Use validation for model selection and reserve test for the final comparison. The preview notebook is available in the dataset's [Code section](https://www.kaggle.com/datasets/dimahkiin/russian-license-plate-characters-23-classes/code).

## Download and prepare

Run commands from the project root. Use Python 3.12 for the training environment. If credentials are requested, follow the [Kaggle CLI authentication instructions](https://github.com/Kaggle/kaggle-cli#authentication).

```bash
python3 -m venv .venv-ocr
source .venv-ocr/bin/activate
python3 -m pip install -r ocr/requirements.txt
python3 ocr/prepare_dataset.py
```

The script downloads the Kaggle dataset into `datasets/raw/`, extracts it into `datasets/ocr/` and generates `data/generated/ocr.*`. Existing data or an existing ZIP are reused. The archive may contain the dataset directly, a wrapper directory or a single nested dataset ZIP. Class IDs, image dimensions and the published train/val/test split stay unchanged.

Preparation checks image/label pairs, image headers, class IDs, box coordinates and exact duplicates across splits. If `manifest.csv` is present, it also checks image hashes and that a group does not cross splits. It accepts zero D annotations and keeps all 23 class names. Missing TXT files are errors, not implicit backgrounds.

For the v1 archive downloaded in a browser:

```bash
python3 ocr/prepare_dataset.py --archive /path/to/russian-license-plate-characters-23-classes_v1.zip
```

For an already extracted dataset:

```bash
python3 ocr/prepare_dataset.py --dataset /path/to/plates-ocr-dataset
```

The dataset root must contain `data.yaml`, `images/{train,val,test}` and `labels/{train,val,test}`. TXT labels are also copied next to images for Darknet; the original `labels/` layout remains available for Ultralytics and the benchmark. Conflicting adjacent labels cause an error rather than being overwritten. Generated paths are absolute and ignored by Git. After moving the dataset, rerun preparation with the new location. To prepare a newer Kaggle version, use new archive and dataset paths.

Preparation generates training inputs only. The already published annotations do not need to be recreated. Vehicle and plate data use separate paths and task names.

## Train models

Download initialization weights if they are not already present:

```bash
python3 scripts/download_pretrained.py --model all
```

Run each model separately. OCR checkpoints use `weights/*-ocr/`; vehicle and plate checkpoints stay in their own task directories. Fresh runs reject an existing output directory.

### Darknet

Install Darknet as described in the [main README](../README.md#1-install-dependencies).

```bash
bash scripts/train_darknet.sh v3-tiny --task ocr
```

```bash
bash scripts/train_darknet.sh v4-tiny --task ocr
```

Darknet uses [yolov3-tiny-ocr.cfg](configs/yolov3-tiny-ocr.cfg) or [yolov4-tiny-ocr.cfg](configs/yolov4-tiny-ocr.cfg), with 23 classes and 84 filters before each detection layer. COCO configs are only used to extract initialization weights for the first 11 layers of v3-tiny or 29 layers of v4-tiny. The OCR-specific detection heads are trained from that initialization. Matching training and inference configs are copied into each run directory.

Resume an interrupted run:

```bash
bash scripts/train_darknet.sh v3-tiny --task ocr --resume weights/yolov3-tiny-ocr/yolov3-tiny-ocr_last.weights
```

### Ultralytics

In the active `.venv-ocr` environment, install the common CUDA 12.4 training dependencies. These pin PyTorch 2.6.0 and torchvision 0.21.0 to CUDA 12.4 builds; see the [main README](../README.md#1-install-dependencies) for the GPU check.

```bash
python3 -m pip install -r requirements-cu124.txt
```

```bash
python3 scripts/train_ultralytics.py --task ocr --model v5nu
```

```bash
python3 scripts/train_ultralytics.py --task ocr --model v8n
```

```bash
python3 scripts/train_ultralytics.py --task ocr --model v9t
```

```bash
python3 scripts/train_ultralytics.py --task ocr --model v11n
```

These commands use `data/generated/ocr.yaml`, COCO pretrained weights, batch 16, 100 epochs and patience 20. YOLOv5nu is the updated `u` variant. Each run exports its best checkpoint to float32 ONNX at 224x64. To use a separate experiment directory, pass `--output weights/experiment-name`.

Resume an interrupted run:

```bash
python3 scripts/train_ultralytics.py --task ocr --model v8n --resume weights/yolov8n-ocr/weights/last.pt
```

## Export to ONNX

Ultralytics training automatically exports `best.pt` to `best.onnx`. Convert Darknet checkpoints separately:

```bash
darknet2onnx --format yolov8 --cfg weights/yolov3-tiny-ocr/yolov3-tiny-ocr-infer.cfg --weights weights/yolov3-tiny-ocr/yolov3-tiny-ocr_best.weights --output weights/yolov3-tiny-ocr/yolov3-tiny-ocr_best.onnx
```

```bash
darknet2onnx --format yolov8 --cfg weights/yolov4-tiny-ocr/yolov4-tiny-ocr-infer.cfg --weights weights/yolov4-tiny-ocr/yolov4-tiny-ocr_best.weights --output weights/yolov4-tiny-ocr/yolov4-tiny-ocr_best.onnx
```

All six exports use static float32 input `[1,3,64,224]`, output `[1,27,N]` and no embedded NMS. The Rust benchmark checks these dimensions when loading the model.

## Benchmark

Build the shared program using the [benchmark instructions](../benchmark/README.md#build). The commands below use CUDA; omit `--cuda` for CPU. If preparation used a custom dataset directory, adjust the image and label paths. Remove model arguments for models that have not finished training.

```bash
mkdir -p benchmark-results/ocr
set -o pipefail
```

Character detection quality:

```bash
benchmark/target/release/benchmark --cuda --task ocr --width 224 --height 64 \
  --val-images datasets/ocr/images/val --val-labels datasets/ocr/labels/val \
  --max-images 0 --detailed \
  --v3-onnx weights/yolov3-tiny-ocr/yolov3-tiny-ocr_best.onnx \
  --v4-onnx weights/yolov4-tiny-ocr/yolov4-tiny-ocr_best.onnx \
  --v5-onnx weights/yolov5nu-ocr/weights/best.onnx \
  --v8-onnx weights/yolov8n-ocr/weights/best.onnx \
  --v9-onnx weights/yolov9t-ocr/weights/best.onnx \
  --v11-onnx weights/yolo11n-ocr/weights/best.onnx \
  2>&1 | tee benchmark-results/ocr/val-cuda.log
```

For a quick check, use `--max-images 100`. Use the whole split for reported metrics. After model selection, switch both `val` paths to `test` and save to `benchmark-results/ocr/test-cuda.log`. Class D stays in the model output, but has no evaluation examples in v1. The benchmark averages AP over classes present in the ground truth; different evaluators may handle absent classes differently.

Speed on one fixed crop:

```bash
sample_image="$(python3 -c 'from pathlib import Path; print(sorted(p for p in Path("datasets/ocr/images/val").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"))[0])')"
printf '%s\n' "$sample_image" > benchmark-results/ocr/speed-image.txt

benchmark/target/release/benchmark --cuda --task ocr --width 224 --height 64 \
  --image "$sample_image" --iterations 1000 --warmup 50 \
  --v3-onnx weights/yolov3-tiny-ocr/yolov3-tiny-ocr_best.onnx \
  --v4-onnx weights/yolov4-tiny-ocr/yolov4-tiny-ocr_best.onnx \
  --v5-onnx weights/yolov5nu-ocr/weights/best.onnx \
  --v8-onnx weights/yolov8n-ocr/weights/best.onnx \
  --v9-onnx weights/yolov9t-ocr/weights/best.onnx \
  --v11-onnx weights/yolo11n-ocr/weights/best.onnx \
  2>&1 | tee benchmark-results/ocr/speed-cuda.log
```

The evaluator measures character AP, precision, recall and detector speed. It does not assemble plate strings or compute character error rate or exact plate accuracy. Speed covers preprocessing, inference and postprocessing, excluding image loading. Use the same image and device for every model. See [AP conventions](../benchmark/README.md#reading-ap-results) before comparing its mAP with training logs. See the [benchmark results](#benchmark-results) below for the measured checkpoints.

## Benchmark results

Validation results on my **Russian license plate character dataset**, measured on 2026-10-05. All six models were evaluated on the same 5,000 validation crops. Speed was measured separately on one fixed crop with warmup. Test-set results are not available yet.

| Model | mAP@0.50 | Mean time (ms) | FPS |
| :--- | ---: | ---: | ---: |
| YOLOv3-tiny | 54.50% | 1.01 | 992.58 |
| YOLOv4-tiny | 65.12% | **0.985** | **1015.35** |
| YOLOv5nu | 85.47% | 1.30 | 769.75 |
| YOLOv8n | 85.88% | 1.31 | 760.46 |
| YOLOv9t | **85.91%** | 2.98 | 335.64 |
| YOLO11n | 85.07% | 1.51 | 663.63 |

> **Note on timing:** Accuracy comes from the full validation run. Timing comes from a separate run with 50 warmup calls and 1,000 measured calls per model. It includes preprocessing, inference and postprocessing, excluding image loading. FPS describes this detector call on the measured GPU and crop, not a complete video stream or Jetson performance.

**Detection metrics at confidence 0.25:**

| Model | Precision (micro) | Recall (micro) | F1 (micro) |
| :--- | ---: | ---: | ---: |
| YOLOv3-tiny | 77.64% | 67.38% | 72.15% |
| YOLOv4-tiny | 81.73% | 75.85% | 78.68% |
| YOLOv5nu | 92.56% | 89.84% | 91.18% |
| YOLOv8n | 92.43% | 90.18% | 91.29% |
| YOLOv9t | 92.52% | 90.28% | 91.39% |
| YOLO11n | 92.68% | 89.80% | 91.22% |

<details>
<summary>Per-character AP@0.50</summary>

| Character | YOLOv3-tiny | YOLOv4-tiny | YOLOv5nu | YOLOv8n | YOLOv9t | YOLO11n |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 58.55% | 67.17% | 81.63% | 81.66% | 81.71% | 81.66% |
| 1 | 67.07% | 79.17% | 90.06% | 90.05% | 90.12% | 90.03% |
| 2 | 58.45% | 68.15% | 90.43% | 81.78% | 90.49% | 81.79% |
| 3 | 59.19% | 68.11% | 81.48% | 81.53% | 81.47% | 81.50% |
| 4 | 57.11% | 64.40% | 81.77% | 90.55% | 90.52% | 81.80% |
| 5 | 60.63% | 68.14% | 90.64% | 90.68% | 90.60% | 90.58% |
| 6 | 59.55% | 69.71% | 81.73% | 81.73% | 81.72% | 81.71% |
| 7 | 68.33% | 78.21% | 90.25% | 90.26% | 90.30% | 90.24% |
| 8 | 58.30% | 67.56% | 90.43% | 90.51% | 90.48% | 90.47% |
| 9 | 58.53% | 67.88% | 81.74% | 81.72% | 81.74% | 81.72% |
| A | 46.55% | 63.36% | 90.52% | 90.43% | 90.45% | 90.37% |
| B | 39.93% | 56.58% | 80.80% | 80.84% | 80.90% | 80.90% |
| C | 45.83% | 63.85% | 90.07% | 90.06% | 90.22% | 90.06% |
| E | 43.35% | 54.38% | 81.00% | 81.23% | 81.02% | 80.93% |
| H | 46.47% | 65.01% | 81.20% | 81.40% | 81.26% | 81.18% |
| K | 48.49% | 59.53% | 81.48% | 89.75% | 81.55% | 81.46% |
| M | 54.62% | 65.84% | 81.55% | 81.60% | 81.60% | 81.60% |
| O | 44.92% | 54.26% | 81.35% | 81.26% | 81.45% | 81.41% |
| P | 66.18% | 68.77% | 90.20% | 90.24% | 90.23% | 90.30% |
| T | 64.34% | 67.17% | 90.72% | 90.69% | 90.72% | 90.74% |
| X | 46.49% | 57.19% | 90.38% | 90.44% | 90.38% | 90.30% |
| Y | 46.14% | 58.10% | 80.95% | 80.97% | 80.99% | 80.91% |
| D | N/A | N/A | N/A | N/A | N/A | N/A |

`D` has no ground-truth examples. The log prints 0.00% for this class, but its AP cannot be evaluated and it is excluded from the reported mAP.

</details>

**Measurement setup:**

- NVIDIA GeForce RTX 5060 Ti, AMD Ryzen 7 7800X3D, ONNX Runtime 1.29.0 / CUDA (system build).
- Float32 ONNX input `[1,3,64,224]`, batch 1, 224x64 with letterbox. Confidence 0.25, NMS IoU 0.45, evaluation IoU 0.50.
- Accuracy: all 5,000 validation crops, 39,169 annotated characters and 8 background crops. The model has 23 output classes; mAP averages the 22 classes represented in the ground truth.
- AP uses 11-point interpolation after confidence filtering. It is not directly interchangeable with Darknet or Ultralytics mAP. Micro precision, recall and F1 aggregate object counts across classes.
- Speed: 50 warmup calls and 1,000 timed calls on `images/val/00112bb2-6418-4bd9-b72e-eb1172742026.jpeg` (137x41). The local dataset root was `/mnt/bigdisk/userspace/plates-ocr-dataset`; the complete image path is recorded in `benchmark-results/ocr/speed-image.txt`.
- Source logs: `benchmark-results/ocr/val-cuda.log` and `benchmark-results/ocr/speed-cuda.log`. System runtime linked through `ORT_LIB_PATH=/usr/lib` and `ORT_PREFER_DYNAMIC_LINK=1`; cuDNN preloaded with `LD_PRELOAD=/usr/lib/libcudnn.so.9`. See [CUDA troubleshooting](../benchmark/README.md#cuda-troubleshooting).

YOLOv9t has the highest measured mAP at 85.91%, only 0.03 percentage points above YOLOv8n. YOLOv8n takes 1.31 ms per crop versus 2.98 ms for YOLOv9t in this speed run, making it a useful starting point for deployment testing. YOLOv5nu is close at 85.47% and 1.30 ms. These small quality and timing differences need repeated measurements before treating them as stable advantages.

YOLOv4-tiny is fastest in this run at 0.985 ms, but its mAP is 20.76 percentage points below YOLOv8n. These results compare the trained checkpoints and current configurations, not the best possible performance of each architecture; the Darknet anchors are still provisional.

The validation annotations were generated automatically and visually spot-checked. Scores measure agreement with those labels, including any remaining annotation errors. They describe individual character detection, not exact plate transcription, character error rate or the complete vehicle -> plate -> OCR cascade. Evaluate the selected model on held-out test data and manually checked examples before drawing conclusions about full-number recognition.

## Output files

- `ocr/configs/`: source Darknet training and inference configs.
- `ocr/classes.names`: character ID order.
- `ocr/prepare_dataset.py`, `ocr/requirements.txt`: published dataset preparation and its dependencies.
- `datasets/raw/`: downloaded archives; safe to remove after successful extraction.
- `datasets/ocr/`: extracted images and annotations used for training and evaluation.
- `data/generated/ocr.yaml`: Ultralytics dataset paths and all 23 classes.
- `data/generated/ocr.data`, `ocr-test.data`, `ocr.names`, `ocr-{train,val,test}.txt`: Darknet inputs.
- `data/generated/ocr-summary.json`: actual split and class counts from validation.
- `weights/*-ocr/`: task-specific checkpoints, exported ONNX and Darknet config copies.
- `benchmark-results/ocr/`: location for saved benchmark logs.

The `-ocr` suffix separates these runs from `-vehicles` and `-plates`. Common pretrained COCO weights are reused only for initialization. Downloaded data, generated training files, checkpoints and benchmark logs are ignored by Git.
