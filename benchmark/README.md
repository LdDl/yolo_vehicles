# Benchmarking on Junction + MIO-TCD

Compare YOLOv3-tiny, YOLOv4-tiny, YOLOv5nu, YOLOv8n, YOLOv9t, and YOLO11n through ONNX Runtime. The benchmark uses crate [`od_opencv v0.8.2`](https://docs.rs/od_opencv/0.8.2/od_opencv/), and requires no OpenCV. All models use letterbox resizing, 416x256 input, confidence threshold 0.25, NMS IoU threshold 0.45, and ground-truth matching IoU threshold 0.50. Classes: `car`, `motorbike`, `bus`, `truck`.

## Table of contents

- [Before benchmarking](#before-benchmarking)
- [Build](#build)
- [Darknet to ONNX](#darknet-to-onnx)
- [Run benchmark](#run-benchmark)
  - [Speed + mAP evaluation (recommended)](#speed--map-evaluation-recommended)
  - [With CUDA acceleration](#with-cuda-acceleration)
  - [Compare multiple models](#compare-multiple-models)
  - [Speed only](#speed-only)
  - [mAP only](#map-only)
- [Reading AP results](#reading-ap-results)
- [Reproducible measurements](#reproducible-measurements)

## Before benchmarking

Follow the [training workflow](../README.md#quick-start) first: prepare the merged dataset once, train each model separately, and export ONNX. The Ultralytics wrapper exports `best.pt` automatically; Darknet models use the conversion commands below.

Keep the model directories under `weights/` and the original evaluation images and labels under `datasets/vehicles/merged/`. Accuracy evaluation needs both `images/val` and `labels/val`, or the corresponding `test` directories. The generated training paths in `data/generated/` are not used by this program. A speed-only run needs ONNX and one image.

## Build

Requires Rust 1.91+ and a C/C++ linker. The `ort` dependency is pinned to `2.0.0-rc.12` (ONNX Runtime 1.24). A standard build downloads a prebuilt runtime for the platform through `ort-sys`. The Rust `image` crate reads images; `od_opencv` handles preprocessing and NMS.

From the project root, for CPU:

```bash
cargo build --release --manifest-path benchmark/Cargo.toml
mkdir -p benchmark-results
```

For CUDA 12 and cuDNN 9:

```bash
ORT_CUDA_VERSION=12 cargo build --release \
  --manifest-path benchmark/Cargo.toml \
  --no-default-features --features ort-cuda
```

Run the binary at `benchmark/target/release/`. Keep the `libonnxruntime_providers_*.so` libraries produced by the CUDA build alongside it. When moving the binary, include these libraries from the same build. If they are symlinks into the Cargo cache, copy the file contents with `cp -L`. CUDA and cuDNN must be available to the system loader. With `--cuda`, a failed CUDA provider registration stops the program with an error, preventing a silent fallback to CPU.

> **Note:** This CUDA build targets CUDA 12 and cuDNN 9. The standard JetPack for Jetson Nano with CUDA 10.2 cannot run it. Use a runtime compatible with the device. See [ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Darknet to ONNX

Convert YOLOv3/v4-tiny with [darknet2onnx](https://github.com/LdDl/darknet2onnx) first. After training, run this from the project root with `darknet2onnx` installed. Each conversion uses the inference config saved with that run:

```bash
darknet2onnx --format yolov8 \
  --cfg weights/yolov3-tiny-vehicles/yolov3-tiny-vehicles-infer.cfg \
  --weights weights/yolov3-tiny-vehicles/yolov3-tiny-vehicles_best.weights \
  --output weights/yolov3-tiny-vehicles/yolov3-tiny-vehicles_best.onnx

darknet2onnx --format yolov8 \
  --cfg weights/yolov4-tiny-vehicles/yolov4-tiny-vehicles-infer.cfg \
  --weights weights/yolov4-tiny-vehicles/yolov4-tiny-vehicles_best.weights \
  --output weights/yolov4-tiny-vehicles/yolov4-tiny-vehicles_best.onnx
```

The `--format yolov8` option belongs to `darknet2onnx`. It keeps the v3/v4 architecture and changes the output layout to `[1,8,N]` for four classes. The converter already multiplies class probabilities by objectness. YOLOv5u/8/9/11 exports also use `[1,8,N]`, so all six models share the same output parser. Legacy YOLOv5 exports with `[1,N,9]` are rejected.

All models must have static float32 input `[1,3,256,416]`, one float32 output without embedded NMS, and class order `car`, `motorbike`, `bus`, `truck`. The program checks input dimensions and output layout when loading a model. Class order comes from dataset preparation and training.

## Run benchmark

The `--v3-onnx`, `--v4-onnx`, `--v5-onnx`, `--v8-onnx`, `--v9-onnx`, and `--v11-onnx` options belong to the benchmark binary and take paths to exported ONNX files. They associate each file with a model name in the results; all use the same loader and output parser. For this comparison, pass the YOLOv5nu export to `--v5-onnx`. The program checks tensor shapes, but does not verify the model architecture against the option name.

Run from the project root in Bash. Set up these variables once and keep using the same terminal:

```bash
set -o pipefail
mkdir -p benchmark-results
dataset_dir="$PWD/datasets/vehicles/merged"
split=val
sample_image="$(python3 -c 'from pathlib import Path; import sys; print(sorted(Path(sys.argv[1]).glob("*.jpg"))[0])' "$dataset_dir/images/val")"
printf '%s\n' "$sample_image" > benchmark-results/speed-image.txt
```

### Speed + mAP evaluation (recommended)

For one model on CPU, measure speed on the fixed image and mAP on the full selected split:

```bash
benchmark/target/release/benchmark \
  --image "$sample_image" --iterations 1000 --warmup 50 \
  --val-images "$dataset_dir/images/$split" \
  --val-labels "$dataset_dir/labels/$split" \
  --max-images 0 --detailed \
  --v3-onnx weights/yolov3-tiny-vehicles/yolov3-tiny-vehicles_best.onnx \
  2>&1 | tee "benchmark-results/$split-v3-cpu.log"
```

When both `--image` and `--val-images` are supplied, FPS comes from the warmed-up speed run on that image, while mAP comes from the entire selected split.

### With CUDA acceleration

Use the CUDA build above and add `--cuda`:

```bash
benchmark/target/release/benchmark \
  --cuda \
  --image "$sample_image" --iterations 1000 --warmup 50 \
  --val-images "$dataset_dir/images/$split" \
  --val-labels "$dataset_dir/labels/$split" \
  --max-images 0 --detailed \
  --v3-onnx weights/yolov3-tiny-vehicles/yolov3-tiny-vehicles_best.onnx \
  2>&1 | tee "benchmark-results/$split-v3-cuda.log"
```

### Compare multiple models

Pass all six ONNX paths in one command. The program loads and benchmarks each model sequentially, then prints a comparison table:

```bash
benchmark/target/release/benchmark \
  --cuda \
  --image "$sample_image" --iterations 1000 --warmup 50 \
  --val-images "$dataset_dir/images/$split" \
  --val-labels "$dataset_dir/labels/$split" \
  --max-images 0 --detailed \
  --v3-onnx weights/yolov3-tiny-vehicles/yolov3-tiny-vehicles_best.onnx \
  --v4-onnx weights/yolov4-tiny-vehicles/yolov4-tiny-vehicles_best.onnx \
  --v5-onnx weights/yolov5nu-vehicles/weights/best.onnx \
  --v8-onnx weights/yolov8n-vehicles/weights/best.onnx \
  --v9-onnx weights/yolov9t-vehicles/weights/best.onnx \
  --v11-onnx weights/yolo11n-vehicles/weights/best.onnx \
  2>&1 | tee "benchmark-results/$split-cuda.log"
```

For CPU comparison, remove `--cuda` and save to `$split-cpu.log`. If only some models are ready, pass only their model arguments.

For a quick check, use `--max-images 100`. This selects the first 100 filenames, not a representative random sample. Use `--max-images 0` for reported results.

After model selection on validation, set `split=test` and repeat the comparison command. This changes both image and label directories and writes `test-cuda.log`. Keep the same `sample_image` for comparable speed measurements. Test data is not used to tune models or thresholds.

### Speed only

Measure speed without an accuracy pass on the dataset:

```bash
benchmark/target/release/benchmark \
  --cuda \
  --image "$sample_image" --iterations 1000 --warmup 50 \
  --v3-onnx weights/yolov3-tiny-vehicles/yolov3-tiny-vehicles_best.onnx \
  --v4-onnx weights/yolov4-tiny-vehicles/yolov4-tiny-vehicles_best.onnx \
  --v5-onnx weights/yolov5nu-vehicles/weights/best.onnx \
  --v8-onnx weights/yolov8n-vehicles/weights/best.onnx \
  --v9-onnx weights/yolov9t-vehicles/weights/best.onnx \
  --v11-onnx weights/yolo11n-vehicles/weights/best.onnx \
  2>&1 | tee "benchmark-results/speed-cuda.log"
```

For CPU measurements, remove `--cuda` and save to `speed-cpu.log`. Timing covers `detect`/`forward`: preprocessing, inference, and postprocessing. Reading the JPEG from disk is excluded. FPS describes the detector on this device and image; a full video pipeline also needs decoding and tracking. Measure directly on Jetson Nano for Nano performance figures.

### mAP only

Use the multiple-model command with the `--image ... --iterations ... --warmup ...` line removed. It evaluates only the selected split. In this mode, the FPS field is calculated from inference calls during validation and is not the separate warmed-up speed measurement.

## Reading AP results

`metrics.rs` uses 11-point interpolation and averages AP over classes present in the selected split's ground truth. Detections are filtered at confidence 0.25 before evaluation. The reported `mAP@0.50` therefore refers to this evaluator configuration. It is not directly interchangeable with Darknet `-points 0`, Ultralytics `mAP50`, or `mAP50-95`.

For reproducible comparisons, keep the same annotation files, model outputs, and evaluator version. Each image must have a TXT file in `labels/`, including an empty file for a background image. A missing label file does not indicate background.

## Reproducible measurements

Keep benchmark logs with the dataset's `summary.json`, training configs, and checkpoint filenames so you can repeat a measurement later. The logs record the runtime version, execution mode, and image count; `speed-image.txt` identifies the fixed image used for speed measurements. Record the CPU and GPU models and memory size alongside them.

Save environment details and model checksums:

```bash
mkdir -p benchmark-results
uname -a > benchmark-results/environment.txt
nvidia-smi >> benchmark-results/environment.txt
cargo tree --manifest-path benchmark/Cargo.toml --features ort-cuda \
  >> benchmark-results/environment.txt
sha256sum weights/*/*best.onnx weights/*/weights/best.onnx \
  > benchmark-results/model-sha256.txt
```

Run the checksum command after exporting the models, adjusting the paths if needed. For a CPU-only build, omit `nvidia-smi` and `--features ort-cuda`.
