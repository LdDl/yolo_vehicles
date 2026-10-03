# Russian license plate detection

Five-class plate detection with YOLOv3-tiny, YOLOv4-tiny, YOLOv5nu, YOLOv8n, YOLOv9t and YOLO11n. Dataset: [Russian license plates: 5-class detection](https://www.kaggle.com/datasets/dimahkiin/russian-license-plates-5-class-detection).

This is the plate detector stage of a planned vehicle -> plate -> OCR cascade. The current dataset contains mixed views, including full vehicles. The commands below train on those published images. Training specifically on vehicle crops will require a separate dataset preparation step that transforms the plate boxes into crop coordinates. OCR needs text annotations and a recognition model; this dataset only provides detection boxes and plate categories.

## Table of contents

- [Classes and splits](#classes-and-splits)
- [Download and prepare](#download-and-prepare)
- [Train models](#train-models)
  - [Darknet](#darknet)
  - [Ultralytics](#ultralytics)
- [Export Darknet to ONNX](#export-darknet-to-onnx)
- [Benchmark](#benchmark)
- [Input size experiments](#input-size-experiments)
- [Output files](#output-files)

## Classes and splits

Keep all five categories in this order:

| ID | Class |
|---|---|
| 0 | civilian |
| 1 | taxi |
| 2 | military |
| 3 | police |
| 4 | diplomatic |

The initial published version contains:

| Split | Images | Background images | Civilian | Taxi | Military | Police | Diplomatic |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2349 | 31 | 1102 | 123 | 503 | 402 | 364 |
| Val | 303 | 0 | 57 | 29 | 105 | 69 | 70 |
| Test | 295 | 4 | 138 | 16 | 63 | 51 | 46 |

Class columns count objects. Images use numeric names, and each image has a matching YOLO TXT file, including an empty file for a background image. Each row is `class_id x_center y_center width height`, normalized to the image dimensions.

The test split was selected from the original training set using seed 42 and similarity groups, while approximately preserving the class proportions. The existing validation split was retained. Similarity grouping is a heuristic, not a guarantee of independent vehicles, cameras or sources. The downloaded dataset includes `split_summary.json` with the split details. Use validation for model selection and reserve test for the final comparison.

The preview Notebook attached to the [Kaggle dataset](https://www.kaggle.com/datasets/dimahkiin/russian-license-plates-5-class-detection/code) displays labels and class counts. It runs without trained models. Image provenance and usage terms are described on the dataset page; the repository's code license does not establish rights to the source images.

## Download and prepare

Run commands from the project root. Use Python 3.11+ and the [Kaggle CLI authentication instructions](https://github.com/Kaggle/kaggle-cli#authentication) if credentials are requested.

```bash
python3 -m venv .venv-plates
source .venv-plates/bin/activate
python3 -m pip install -r plates/requirements.txt
python3 plates/prepare_dataset.py
```

The script downloads the dataset through the official Kaggle CLI into `datasets/raw/`, extracts it to `datasets/plates/` and generates training files in `data/generated/`. It preserves the published class IDs, names and train/val/test split. Existing extracted data or an existing archive are reused. It checks image/label pairs, box coordinates, image headers and exact duplicates across splits before generating training files. Re-running refreshes the generated paths without downloading again.

For an archive downloaded in a browser:

```bash
python3 plates/prepare_dataset.py --archive /path/to/russian-license-plates-5-class-detection.zip
```

For an already extracted dataset:

```bash
python3 plates/prepare_dataset.py --dataset /path/to/plate-dataset
```

The extracted root must contain `data_multiclass.yaml`, `images/{train,val,test}` and `labels/{train,val,test}`. The script also copies TXT labels next to images for Darknet, keeping the standard `labels/` directories for Ultralytics and the benchmark. Generated absolute paths are machine-local and ignored by Git. After moving the dataset, rerun preparation with its new location.

Keep the dataset version and ZIP checksum alongside experiment results. To use a newer Kaggle version, choose a new archive path and a new `--dataset` directory.

## Train models

The initial comparison uses 320x192 letterboxed inference input. This is a starting size for the plate experiment; evaluate small plates in the original images before choosing a deployment size. Training outputs use the `-plates` suffix and do not share runs with vehicle models.

Download the COCO initialization weights once:

```bash
python3 scripts/download_pretrained.py --model all
```

Run the following training commands individually. No model is trained by the dataset preparation script.

### Darknet

Install Darknet as described in the [main README](../README.md#1-install-dependencies).

```bash
bash scripts/train_darknet.sh v3-tiny --task plates
```

```bash
bash scripts/train_darknet.sh v4-tiny --task plates
```

Training uses [yolov3-tiny-plates.cfg](configs/yolov3-tiny-plates.cfg) or [yolov4-tiny-plates.cfg](configs/yolov4-tiny-plates.cfg): five classes, 30 filters before each detection layer, batch 64, subdivisions 4, 10000 iterations and learning-rate drops at 8000/9000. These are starting settings, not measured optimal values. COCO configs are used only to extract the first 11 or 29 layers from pretrained weights. The five-class detection layers are trained separately from that initialization.

Letterbox is enabled, mosaic is disabled, and horizontal flipping and hue/saturation augmentation are disabled to preserve text direction and category colors. Anchors remain the initial templates; adjust them only from training labels when exploring a separate experiment.

Resume an interrupted run:

```bash
bash scripts/train_darknet.sh v3-tiny --task plates --resume weights/yolov3-tiny-plates/yolov3-tiny-plates_last.weights
```

To start from random initialization, add `--scratch` to a fresh run. Existing run directories are rejected for fresh training. Keep previous runs under another name before starting a new experiment.

### Ultralytics

Install the common training dependencies using the [main README](../README.md#1-install-dependencies).

```bash
python3 -m pip install -r requirements.txt
```

```bash
python3 scripts/train_ultralytics.py --task plates --model v5nu
```

```bash
python3 scripts/train_ultralytics.py --task plates --model v8n
```

```bash
python3 scripts/train_ultralytics.py --task plates --model v9t
```

```bash
python3 scripts/train_ultralytics.py --task plates --model v11n
```

These commands use `data/generated/plates.yaml`, COCO pretrained weights, `imgsz=320`, rectangular batching, 100 epochs and patience 20. Horizontal/vertical flips and hue/saturation changes are disabled for this task. YOLOv5nu is the updated `u` variant, not original YOLOv5n. Each command exports its best checkpoint to static float32 ONNX with input `[1,3,192,320]` and no embedded NMS.

Resume an interrupted run:

```bash
python3 scripts/train_ultralytics.py --task plates --model v8n --resume weights/yolov8n-plates/weights/last.pt
```

Resume uses the checkpoint's training settings. For a new experiment, `--output weights/experiment-name` selects a separate run location.

## Export Darknet to ONNX

With [darknet2onnx](https://github.com/LdDl/darknet2onnx) installed:

```bash
darknet2onnx --format yolov8 --cfg weights/yolov3-tiny-plates/yolov3-tiny-plates-infer.cfg --weights weights/yolov3-tiny-plates/yolov3-tiny-plates_best.weights --output weights/yolov3-tiny-plates/yolov3-tiny-plates_best.onnx
```

```bash
darknet2onnx --format yolov8 --cfg weights/yolov4-tiny-plates/yolov4-tiny-plates-infer.cfg --weights weights/yolov4-tiny-plates/yolov4-tiny-plates_best.weights --output weights/yolov4-tiny-plates/yolov4-tiny-plates_best.onnx
```

All six plate models use output `[1,9,N]`: four box coordinates and five class scores. `darknet2onnx --format yolov8` includes objectness in the class scores. The Rust benchmark does not need OpenCV.

## Benchmark

Follow the [benchmark build instructions](../benchmark/README.md#build). The shared executable uses `--task plates` to select the five-class taxonomy; the default remains `vehicles`.

```bash
mkdir -p benchmark-results/plates
set -o pipefail
benchmark/target/release/benchmark \
  --cuda --task plates --width 320 --height 192 \
  --val-images datasets/plates/images/val \
  --val-labels datasets/plates/labels/val \
  --max-images 0 --detailed \
  --v3-onnx weights/yolov3-tiny-plates/yolov3-tiny-plates_best.onnx \
  --v4-onnx weights/yolov4-tiny-plates/yolov4-tiny-plates_best.onnx \
  --v5-onnx weights/yolov5nu-plates/weights/best.onnx \
  --v8-onnx weights/yolov8n-plates/weights/best.onnx \
  --v9-onnx weights/yolov9t-plates/weights/best.onnx \
  --v11-onnx weights/yolo11n-plates/weights/best.onnx \
  2>&1 | tee benchmark-results/plates/val-cuda.log
```

Pass only the model arguments whose weights are ready. For CPU, omit `--cuda`. For a quick check, use `--max-images 20`; use the whole split for reported metrics. For the final evaluation, change both `val` directories to `test` and the log name to `test-cuda.log`.

For a separate speed measurement, replace the two validation directory arguments with `--image path/to/fixed-image.jpg --iterations 1000 --warmup 50` and retain the same model arguments. Use one fixed image and one device for all models. The measured call includes preprocessing, inference and postprocessing, not JPEG loading. The AP estimator and its limitations are described in [Reading AP results](../benchmark/README.md#reading-ap-results).

Do not combine plate results with the vehicle table: these are different tasks and taxonomies. Record dataset version, model checkpoint, dimensions, thresholds, runtime and device with every result. No plate benchmark results are available yet.

## Input size experiments

Compare the initial 320x192 input against other sizes, such as 416x256, using the same validation split and device. Keep the baseline run intact. For Darknet, update both the training and inference configs consistently and re-export; for Ultralytics, use `--imgsz 416 --export-width 416 --export-height 256` with a fresh output directory for a 416x256 experiment. Training uses a single `imgsz` plus rectangular batching; ONNX export fixes the exact dimensions. Set benchmark `--width 416 --height 256` to match that experiment. A benchmark flag does not resize the ONNX graph; mismatched inputs are rejected.

## Output files

- `plates/`: documentation, preparation script, dependencies and source configs.
- `datasets/raw/`: downloaded archives; safe to remove after successful extraction.
- `datasets/plates/`: images and annotations used for training and evaluation; keep these while running experiments.
- `data/generated/plates.yaml`: Ultralytics dataset paths.
- `data/generated/plates.data`, `plates-test.data`, `plates.names`, `plates-{train,val,test}.txt`: Darknet inputs.
- `data/generated/plates-summary.json`: validation counts from preparation.
- `weights/*-plates/`: training outputs, best/last checkpoints and ONNX exports. Darknet also stores copies of training/inference configs and the training log.
- `benchmark-results/plates/`: comparison logs.

Downloaded data, generated paths, weights and benchmark logs are ignored by Git.
