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

The evaluator measures character AP, precision, recall and detector speed. It does not assemble plate strings or compute character error rate or exact plate accuracy. Speed covers preprocessing, inference and postprocessing, excluding image loading. Use the same image and device for every model. See [AP conventions](../benchmark/README.md#reading-ap-results) before comparing its mAP with training logs. No OCR quality or speed results have been measured yet.

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
