# License plate character detection

The third stage of the vehicle -> plate -> OCR cascade: detect individual characters inside a cropped license plate. The initial comparison uses YOLOv3-tiny, YOLOv4-tiny, YOLOv5nu, YOLOv8n, YOLOv9t and YOLO11n at **224x64**, width x height, with letterbox. Full-number decoding, row ordering and transcription metrics are not implemented yet.

## Table of contents

- [Model settings](#model-settings)
- [Training inputs](#training-inputs)
- [Train models](#train-models)
- [Export to ONNX](#export-to-onnx)
- [Benchmark](#benchmark)
- [Output files](#output-files)

## Model settings

The initial alphabet has 23 classes. [classes.names](classes.names) defines the ID order, starting at zero:

```text
0 1 2 3 4 5 6 7 8 9 A B C E H K M O P T X Y D
```

Letters use Latin characters. `D` is class 22, appended for diplomatic plates. These IDs are the model contract, not an assumed mapping from any external annotation source. A change to the alphabet requires matching changes to the annotations, Darknet classes/filters, training configuration and benchmark class list.

Darknet uses 23 classes and 84 filters before each detection layer: `(23 + 5) * 3`. Both training and inference configs use 224x64. Letterbox preserves each crop's proportions. Flipping, mosaic, crop jitter and random input resizing are disabled. Anchors are provisional character-sized values, not fitted to annotations. Batch 64, subdivisions 4 and 46000 iterations with learning-rate drops at 36800/41400 are starting settings, not measured optimal values.

Ultralytics training uses `imgsz=224` and `rect=True`; batch shapes vary with the image proportions and are not fixed at 224x64. Export fixes the input to `[1,3,64,224]`. Flipping, mosaic, mixup, rotation, shear, perspective and hue/saturation augmentation are disabled for this task.

## Training inputs

Training commands are wired up, but preparation and automatic annotation are not included yet. Before running them, provide `data/generated/ocr.yaml` for Ultralytics or `data/generated/ocr.data` for Darknet, using the exact ID order from `ocr/classes.names`. The Darknet data file must declare `classes = 23`, reference that names file and valid train/validation image lists with matching character labels. These generated files are machine-local and ignored by Git.

The commands below do not create annotations. Existing vehicle and plate training inputs are separate.

## Train models

Run commands from the project root. Install Darknet as described in the [main README](../README.md#1-install-dependencies). For Ultralytics with the CUDA 12.4 dependency set:

```bash
python3 -m venv .venv-ocr
source .venv-ocr/bin/activate
python3 -m pip install -r requirements-cu124.txt
```

Download initialization weights if they are not already present:

```bash
python3 scripts/download_pretrained.py --model all
```

Run each model separately after providing the training inputs:

```bash
bash scripts/train_darknet.sh v3-tiny --task ocr
```

```bash
bash scripts/train_darknet.sh v4-tiny --task ocr
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

Darknet uses the configs in `ocr/configs/`. COCO configs are only used to extract initialization weights for the first 11 layers of v3-tiny or 29 layers of v4-tiny. OCR detection heads use the new class count. Fresh runs reject existing output directories.

Resume an interrupted run with its original task:

```bash
bash scripts/train_darknet.sh v3-tiny --task ocr --resume weights/yolov3-tiny-ocr/yolov3-tiny-ocr_last.weights
```

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

Build the shared program using the [benchmark instructions](../benchmark/README.md#build). Add `--cuda` to the commands below when using a compatible CUDA build. Replace the placeholder paths with your evaluation inputs. Remove model arguments for models that have not finished training.

Character detection quality:

```bash
benchmark/target/release/benchmark --task ocr --width 224 --height 64 \
  --val-images /path/to/images/val --val-labels /path/to/labels/val \
  --max-images 0 --detailed \
  --v3-onnx weights/yolov3-tiny-ocr/yolov3-tiny-ocr_best.onnx \
  --v4-onnx weights/yolov4-tiny-ocr/yolov4-tiny-ocr_best.onnx \
  --v5-onnx weights/yolov5nu-ocr/weights/best.onnx \
  --v8-onnx weights/yolov8n-ocr/weights/best.onnx \
  --v9-onnx weights/yolov9t-ocr/weights/best.onnx \
  --v11-onnx weights/yolo11n-ocr/weights/best.onnx
```

Speed on one fixed crop:

```bash
benchmark/target/release/benchmark --task ocr --width 224 --height 64 \
  --image /path/to/crop.jpg --iterations 1000 --warmup 50 \
  --v3-onnx weights/yolov3-tiny-ocr/yolov3-tiny-ocr_best.onnx \
  --v4-onnx weights/yolov4-tiny-ocr/yolov4-tiny-ocr_best.onnx \
  --v5-onnx weights/yolov5nu-ocr/weights/best.onnx \
  --v8-onnx weights/yolov8n-ocr/weights/best.onnx \
  --v9-onnx weights/yolov9t-ocr/weights/best.onnx \
  --v11-onnx weights/yolo11n-ocr/weights/best.onnx
```

The evaluator measures character AP, precision, recall and detector speed. It does not assemble plate strings or compute character error rate or exact plate accuracy. See [AP conventions](../benchmark/README.md#reading-ap-results) before comparing its mAP with training logs. No OCR quality or speed results have been measured yet.

## Output files

- `ocr/configs/`: source Darknet training and inference configs.
- `ocr/classes.names`: character ID order.
- `data/generated/ocr.*`: machine-local training inputs, supplied separately.
- `weights/*-ocr/`: task-specific checkpoints, exported ONNX and Darknet config copies.
- `benchmark-results/ocr/`: location for saved benchmark logs.

The `-ocr` suffix separates these runs from `-vehicles` and `-plates`. Common pretrained COCO weights are reused only for initialization.
