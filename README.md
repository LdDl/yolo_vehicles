# YOLO Vehicles Detection

Training and benchmarking YOLO models for vehicle detection:

- **Darknet**: YOLOv3-tiny, YOLOv4-tiny
- **Ultralytics**: YOLOv5nu, YOLOv8n, YOLOv9t, YOLOv11n

> **Note on YOLOv5nu:** `n` means nano, and `u` identifies the updated Ultralytics variant. Compared with the original YOLOv5n, it uses a YOLOv8-style detection head without predefined anchor boxes or a separate objectness score. This changes both the architecture and output layout. Keep the `u` suffix in benchmark results to identify the model correctly. See [YOLOv5u details](https://docs.ultralytics.com/models/yolov5/).

All models use the combined **Junction + MIO-TCD** dataset, the same train/val/test split, and **416x256** input for benchmarking. The target is vehicle detection on edge devices like Jetson Nano.

> **Note on input size:** 416x256 is width x height, with a 13:8 aspect ratio. Letterbox adds padding to preserve the original image proportions.

> **Note on Ultralytics training:** The `imgsz` parameter only accepts a single integer during training (e.g., `imgsz=416`). Use `rect=True` to enable rectangular batching that adapts to each batch's aspect ratio. See [ultralytics#235](https://github.com/ultralytics/ultralytics/issues/235).

> **Note on Ultralytics export:** `imgsz` uses `[height, width]`. The scripts export ONNX with `imgsz=[256,416]` to match the Darknet input size.

## Table of contents

- [Download trained models](#download-trained-models)
- [Classes](#classes)
- [Datasets](#datasets)
- [License plate detection](#license-plate-detection)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
  - [1. Install dependencies](#1-install-dependencies)
  - [2. Download and prepare datasets](#2-download-and-prepare-datasets)
  - [3. Download pretrained weights](#3-download-pretrained-weights)
  - [4. Train models](#4-train-models)
    - [YOLOv3-tiny / YOLOv4-tiny (Darknet)](#yolov3-tiny--yolov4-tiny-darknet)
    - [YOLOv5nu / YOLOv8n / YOLOv9t / YOLO11n (Ultralytics)](#yolov5nu--yolov8n--yolov9t--yolo11n-ultralytics)
  - [5. Export Darknet models to ONNX](#5-export-darknet-models-to-onnx)
- [Output files](#output-files)
  - [Files for benchmarking](#files-for-benchmarking)
- [Benchmarking](#benchmarking)
  - [Build benchmark](#build-benchmark)
  - [Run benchmark](#run-benchmark)
  - [Speed + mAP evaluation (recommended)](#speed--map-evaluation-recommended)
  - [With CUDA acceleration](#with-cuda-acceleration)
  - [Compare multiple models](#compare-multiple-models)
- [Benchmark results](#benchmark-results)
- [Tests](#tests)

## Download trained models

Ready-to-use models trained on **Junction + MIO-TCD** are available in [release v0.0.3](https://github.com/LdDl/yolo_vehicles/releases/tag/v0.0.3). Download them to run inference without training the models yourself.

- Best checkpoints for all six models: `.weights` for Darknet and `.pt` for Ultralytics.
- ONNX exports with input `[1,3,256,416]`, batch 1, and four classes.
- Darknet training and inference configs, `vehicles.names`, and `SHA256SUMS`.
- TensorRT `.engine` files built with FP16 enabled for all six models on each configuration below.

| Device | CUDA | cuDNN | TensorRT |
| :--- | :--- | :--- | :--- |
| Jetson Nano | 10.2.300 | 8.2.1.32 | 8.2.1.8 |
| Jetson Orin Nano | 12.6.68 | 9.3.0 | 10.3.0.30 |

Engine filenames include the model, device, software versions and precision. For example:

```text
yolov8n-vehicles_best_jetson_nano_cuda-10.2.300_cudnn-8.2.1.32_trt-8.2.1.8_fp16.engine
yolov8n-vehicles_best_jetson_orin_nano_cuda-12.6.68_cudnn-9.3.0_trt-10.3.0.30_fp16.engine
```

Choose the engine matching your device and software stack. For a different configuration, build an engine from the ONNX file on the target device. Jetson Nano and Jetson Orin Nano engines are separate builds and are not interchangeable.

The Rust benchmark below uses ONNX files. Release assets have model-specific names, such as `yolov5nu-vehicles_best.onnx`; adjust the benchmark paths to where you downloaded them.

>Note: I want to mention again that traditional YOLOv3 and v4 have been converted ONNX (to the YOLOv8 output format) via [`darknet2onnx`](https://github.com/LdDl/darknet2onnx) and then to TensorRT engines.

## Classes

| ID | Class |
| :--- | :--- |
| 0 | car |
| 1 | motorbike |
| 2 | bus |
| 3 | truck |

## Datasets

| Source | Included subset | Original annotation format |
| :--- | :--- | :--- |
| [Junction, version 1](https://data.mendeley.com/datasets/vwjg6b7kpt/1) | Public sample of 3952 images | `sample_labels.csv` and `sampled_images/` |
| [MIO-TCD Localization](https://tcd.miovision.com/challenge/dataset.html) | Annotated `train` subset, 110000 images before filtering | `gt_train.csv` and `train/` |

Both datasets are converted to YOLO annotations and merged by [scripts/prepare_dataset.py](scripts/prepare_dataset.py).

This will:

- Map source categories to our four classes using explicit dictionaries
- Use actual JPG dimensions to fix the incorrect sizes in the Junction CSV
- Keep background images with empty TXT files
- Exclude MIO images containing the ambiguous `motorized_vehicle` category
- Remove exact duplicates and exclude identical images with conflicting labels
- Group related Junction frames and augmentations before splitting

**Prepared split** (`seed=42`, Junction then MIO, 80/10/10):

| Split | Images |
| :--- | ---: |
| train | 62587 |
| val | 7456 |
| test | 7460 |

These are dataset counts, not model results. Each run saves its actual counts to `summary.json`.

> **Note:** Junction provides a public sample, not the complete dataset from the paper. The official MIO test set has no available labels, so our val/test splits come from its annotated train set. MIO camera IDs are unavailable; this split does not guarantee evaluation on unseen cameras. Classes are not automatically balanced.

See [docs/datasets.md](docs/datasets.md) for class mappings, filtering, and cleanup instructions. MIO's original README specifies **CC BY-NC-SA 4.0**; annotation conversion does not change the dataset license.

## License plate detection

A separate [plate detection workflow](plates/README.md) uses my [Russian license plate dataset](https://www.kaggle.com/datasets/dimahkiin/russian-license-plates-5-class-detection) with five categories: civilian, taxi, military, police and diplomatic. It includes dataset preparation, training configs and comparison through the same Rust benchmark with `--task plates`. This is the detection stage of a planned vehicle -> plate -> OCR cascade; OCR is not implemented yet.

## Project Structure

```text
vehicles_yolo/
|-- configs/                         Darknet training and inference configs
|-- data/generated/                  Generated class names and training paths
|-- scripts/
|   |-- prepare_dataset.py           Download, convert, merge, configure paths
|   |-- download_pretrained.py       Download official initial weights
|   |-- generate_file_lists.sh       Configure an existing merged dataset
|   |-- train_darknet.sh             Train v3-tiny, v4-tiny
|   |-- train_ultralytics.py         Train v5nu, v8n, v9t, v11n
|   `-- create_videos.sh             Create a slideshow from validation images
|-- datasets/
|   |-- raw/                         Archives and extracted datasets
|   `-- vehicles/
|       |-- prepared/                Converted source datasets
|       `-- merged/                  Final images/labels and train/val/test splits
|-- benchmark/                      Rust benchmark (uses od_opencv)
|-- docs/datasets.md                Dataset formats, mappings, and cleanup
|-- weights/                        Initial weights and training output
|-- tests/                          Preparation and launcher checks
|-- requirements-data.txt           Dataset preparation dependencies
|-- requirements.txt                Training and export dependencies
`-- README.md
```

## Quick Start

The workflow is:

1. Install dependencies and prepare the combined Junction + MIO-TCD dataset once.
2. Generate training paths and use the supplied four-class model configs.
3. Train YOLOv3-tiny, YOLOv4-tiny, YOLOv5nu, YOLOv8n, YOLOv9t, and YOLO11n one at a time on the same split.
4. Convert the two Darknet models with [`darknet2onnx`](https://github.com/LdDl/darknet2onnx). The Ultralytics training script exports each best checkpoint to ONNX automatically.
5. Keep the trained weights, matching configs, ONNX files, and evaluation data for benchmarking.
6. Compare the models with the Rust benchmark through ONNX Runtime, using 416x256 input for every model.

Run the setup, training, and export commands below from the project root. Then follow [Benchmarking](#benchmarking) to measure speed and accuracy.

### 1. Install Dependencies

Use **Python 3.12** for training. Dataset preparation alone needs Python 3.10+ and [requirements-data.txt](requirements-data.txt).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-cu124.txt
```

`requirements-cu124.txt` installs the training and export dependencies with PyTorch 2.6.0 and torchvision 0.21.0 built for CUDA 12.4. The base `requirements.txt` also pins these versions, while the CUDA-specific file selects the exact `+cu124` builds. Check that the GPU is available:

```bash
nvidia-smi
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Darknet must be available as `darknet`, built with GPU and cuDNN support. OpenCV is needed for Darknet charts. Dataset preparation and the Rust benchmark do not use OpenCV; the benchmark runs ONNX models through ONNX Runtime.

### 2. Download and Prepare Datasets

```bash
python3 scripts/prepare_dataset.py all \
  --directory datasets/raw \
  --output datasets/vehicles \
  --training-dir data/generated
```

This downloads missing datasets, converts labels, creates the merged train/val/test split, and generates files in `data/generated/` for both Darknet and Ultralytics.

- Existing archives or extracted datasets are reused
- Existing output datasets are not overwritten
- Original annotations stay in `labels/`
- TXT copies are also placed next to images for this Darknet fork

**Already prepared the merged dataset?** Just configure its paths:

```bash
python3 scripts/prepare_dataset.py configure \
  --dataset /absolute/path/to/merged \
  --output data/generated --backup weights --darknet-labels
```

> **Note:** Run `configure` again after moving the dataset. Files in `data/generated/` contain absolute paths; the merged dataset's own `data.yaml` and images/labels structure are portable. Generated paths, datasets, and weights are ignored by Git.

`configure` defaults to `data/generated/`. This keeps generated training files separate from the model templates in `configs/`. For an existing merged dataset, this step only regenerates training files and checks image/label pairs.

**Training files prepared by this step:**

| File | Used by |
| :--- | :--- |
| `data/generated/vehicles.yaml` | Ultralytics training on the merged train/val/test split |
| `data/generated/vehicles.data` | Darknet training and validation |
| `data/generated/vehicles-test.data` | Final Darknet evaluation on test |
| `data/generated/vehicles-{train,val,test}.txt` | Absolute image paths for each split |
| `data/generated/vehicles.names` | Shared class order: car, motorbike, bus, truck |

The model configs are already in `configs/yolov3-tiny-vehicles.cfg` and `configs/yolov4-tiny-vehicles.cfg`, with matching `-infer.cfg` files. They specify four classes and 416x256 input. The Darknet wrapper creates a separate backup directory and saves copies of both configs for each run. Ultralytics uses the selected model's architecture and the classes from `data/generated/vehicles.yaml`; it does not need a Darknet CFG.

### 3. Download Pretrained Weights

```bash
python3 scripts/download_pretrained.py --model v3-tiny
python3 scripts/download_pretrained.py --model v4-tiny
python3 scripts/download_pretrained.py --model v5nu
python3 scripts/download_pretrained.py --model v8n
python3 scripts/download_pretrained.py --model v9t
python3 scripts/download_pretrained.py --model v11n
```

Weights are saved to `weights/pretrained/`. Run only the corresponding download command if you are preparing one model at a time. `--model all` downloads all six models. Existing nonempty files are reused; incomplete downloads stay in `.part` files.

For Darknet, the download includes full COCO weights and matching configs. The training script runs `darknet partial` to extract the first **11 layers for v3-tiny** or **29 layers for v4-tiny**. The four-class prediction heads are trained on our dataset. No separate Google Drive download is needed.

The downloaded `*-coco.cfg` describes the original 80-class model and is used only to extract initial weights. Training uses the four-class config from `configs/`:

| Command | Config for `darknet partial` | Config for training |
| :--- | :--- | :--- |
| `bash scripts/train_darknet.sh v3-tiny` | `weights/pretrained/yolov3-tiny-coco.cfg` | `configs/yolov3-tiny-vehicles.cfg` |
| `bash scripts/train_darknet.sh v4-tiny` | `weights/pretrained/yolov4-tiny-coco.cfg` | `configs/yolov4-tiny-vehicles.cfg` |

Edit the training config in `configs/` to change the input size, batch, learning rate or number of iterations. The matching `-infer.cfg` is for inference and export.

### 4. Train Models

Each command below starts one training run. Wait for it to finish before starting the next model. All six runs reuse the dataset prepared above and save to separate output directories. Use `--resume` to continue an interrupted run.

#### YOLOv3-tiny / YOLOv4-tiny (Darknet)

Requires [AlexeyAB's Darknet](https://github.com/AlexeyAB/darknet) compiled with these settings in `darknet/Makefile`:

```makefile
GPU=1
CUDNN=1
OPENCV=1
LIBSO=1
```

`OPENCV=1` enables Darknet charts; `LIBSO=1` also builds `libdarknet.so`. More build options: [darknet/README.md](darknet/README.md).

**Arch Linux / CachyOS users (I'm using Arch btw):** CUDA is installed at `/opt/cuda/` instead of `/usr/local/cuda/`. Edit `darknet/Makefile` and replace all occurrences of `/usr/local/cuda/` with `/opt/cuda/`:

```makefile
COMMON+= -DGPU -I/opt/cuda/include/
LDFLAGS+= -L/opt/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
CFLAGS+= -DCUDNN -I/opt/cuda/include
LDFLAGS+= -L/opt/cuda/lib64 -lcudnn
```

If `nvcc` is not on your `PATH`:

```bash
export PATH="/opt/cuda/bin:$PATH"
nvcc --version
```

Also set `ARCH` for your GPU (e.g., RTX 3060 = Ampere, compute 8.6):

```makefile
ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]
```

For Quadro RTX 6000 (Turing, compute 7.5), use this instead:

```makefile
ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]
```

> **Note:** The bundled Makefile already uses `/opt/cuda/` and the RTX 3060 setting. Adjust the CUDA/cuDNN paths and `ARCH` for your GPU and CUDA installation.

<details>
<summary><strong>CUDA 13+ compatibility fix</strong></summary>

If you get `cudaHostAlloc` incompatible pointer type errors, add `(void**)` casts to the first argument in these files. The casts are already present in the bundled sources; keep this reference when using another Darknet checkout.

**src/network.c** (line ~660):

```c
if (cudaSuccess == cudaHostAlloc((void**)&net->input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped))
```

**src/parser.c** (line ~1761):

```c
if (cudaSuccess == cudaHostAlloc((void**)&net.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped))
```

**src/yolo_layer.c** (lines ~67, 74, 105, 114):

```c
if (cudaSuccess == cudaHostAlloc((void**)&l.output, ...))
if (cudaSuccess == cudaHostAlloc((void**)&l.delta, ...))
if (cudaSuccess != cudaHostAlloc((void**)&l->output, ...))
if (cudaSuccess != cudaHostAlloc((void**)&l->delta, ...))
```

**src/gaussian_yolo_layer.c** (lines ~70, 77, 109, 118): same pattern as `yolo_layer.c`, add `(void**)` to all four `cudaHostAlloc` calls. The `...` above stands for the existing size and flags arguments; leave those unchanged.

</details>

**Build Darknet** (from the project root):

```bash
make -C darknet clean
make -C darknet -j"$(nproc)"
```

**Install system-wide (optional):**

```bash
sudo cp darknet/darknet /usr/local/bin/
sudo cp darknet/libdarknet.so /usr/local/lib/
sudo ldconfig
```

Without system-wide installation, add the build directory to `PATH` so the training scripts can find `darknet`:

```bash
export PATH="$PWD/darknet:$PATH"
```

**Train YOLOv3-tiny:**

```bash
bash scripts/train_darknet.sh v3-tiny
```

**Then train YOLOv4-tiny:**

```bash
bash scripts/train_darknet.sh v4-tiny
```

Current training settings:

- `batch=64`, `subdivisions=4` - Increase subdivisions if GPU memory is insufficient
- `max_batches=64000`, `steps=51200,57600` - Training budget and learning-rate steps
- `letter_box=1`, `mosaic=0` - Preserve image proportions
- `random=1` - Vary the training resolution; validation uses 416x256

These are starting settings for the new dataset. Their accuracy still needs to be measured.

The script uses `-clear` for new runs and `-mAP_epochs 1` for validation. With 62587 train images and batch 64, the first mAP check is at iteration 1000 (after warmup), then approximately every 977 iterations.

- `*_best.weights` - Best checkpoint on validation
- `*_final.weights` - Last training state
- Test images are not used for training or checkpoint selection

**Resume:**

```bash
bash scripts/train_darknet.sh v3-tiny --resume \
  weights/yolov3-tiny-vehicles/yolov3-tiny-vehicles_last.weights
```

> **Note on resume:** Darknet resets its best-mAP counter on each launch. The script keeps a copy of the previous best as `*best-before-resume*.weights`. Compare it with the new best on the same validation split.

#### YOLOv5nu / YOLOv8n / YOLOv9t / YOLO11n (Ultralytics)

The same script trains and exports all four models. YOLOv5nu uses the updated [Ultralytics YOLOv5u head](https://docs.ultralytics.com/models/yolov5/). All four models use the installed `ultralytics` package. Run them one at a time:

**YOLOv5nu:**

```bash
python3 scripts/train_ultralytics.py --model v5nu --epochs 100 --batch 16
```

**YOLOv8n:**

```bash
python3 scripts/train_ultralytics.py --model v8n --epochs 100 --batch 16
```

**YOLOv9t:**

```bash
python3 scripts/train_ultralytics.py --model v9t --epochs 100 --batch 16
```

**YOLO11n:**

```bash
python3 scripts/train_ultralytics.py --model v11n --epochs 100 --batch 16
```

Options:

- `--model v5nu|v5su|v5mu|v8n|v9t|v11n` - Model variant (default: v8n)
- `--batch 16` - Adjust batch size for your GPU
- `--device 0` - CUDA device ID
- `--scratch` - Train without pretrained weights

For larger YOLOv5u variants, download `--model v5su` or `v5mu` and pass the same model name to the training script. For `--scratch`, the script selects the matching Ultralytics YAML (`yolov5n.yaml`, `yolov5s.yaml`, or `yolov5m.yaml`); these config filenames omit the checkpoint suffix `u`.

The Ultralytics training script uses COCO weights, `seed=42`, `rect=True`, `imgsz=416`, and `patience=20`. Training stops after 20 epochs without improvement.

Output weights are saved to `weights/<model>-vehicles/weights/best.pt` and automatically exported to ONNX: float32 `[1,3,256,416]`, batch 1, opset 12, no embedded NMS. See [Ultralytics export](https://docs.ultralytics.com/modes/export/).

> **Note on NMS:** In the pinned Ultralytics 8.4.153, `nms=None` exports for external NMS. `nms=False` selects the NMS-free branch where available. The wrapper handles this. See the [version's settings](https://github.com/ultralytics/ultralytics/blob/v8.4.153/ultralytics/cfg/default.yaml).

**Resume Ultralytics:**

```bash
python3 scripts/train_ultralytics.py --resume weights/yolo11n-vehicles/weights/last.pt
```

### 5. Export Darknet Models to ONNX

After the two Darknet runs finish, convert the weights with [darknet2onnx](https://github.com/LdDl/darknet2onnx). Run from the project root with `darknet2onnx` installed:

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

Each command uses the inference config saved with its training run. Run the corresponding command when that model finishes training. `--format yolov8` changes the output layout; the networks remain YOLOv3-tiny and YOLOv4-tiny.

YOLOv5nu, YOLOv8n, YOLOv9t, and YOLO11n already have `best.onnx` from the training script. All six ONNX files use float32 input `[1,3,256,416]`, batch 1, and output `[1,8,N]` for my four classes. NMS runs in the Rust detector.

## Output Files

| Model | Best weights | ONNX for the Rust benchmark |
| :--- | :--- | :--- |
| YOLOv3-tiny | `weights/yolov3-tiny-vehicles/yolov3-tiny-vehicles_best.weights` | `yolov3-tiny-vehicles_best.onnx` in the same directory |
| YOLOv4-tiny | `weights/yolov4-tiny-vehicles/yolov4-tiny-vehicles_best.weights` | `yolov4-tiny-vehicles_best.onnx` in the same directory |
| YOLOv5nu | `weights/yolov5nu-vehicles/weights/best.pt` | `best.onnx` in the same directory |
| YOLOv8n | `weights/yolov8n-vehicles/weights/best.pt` | `best.onnx` in the same directory |
| YOLOv9t | `weights/yolov9t-vehicles/weights/best.pt` | `best.onnx` in the same directory |
| YOLO11n | `weights/yolo11n-vehicles/weights/best.pt` | `best.onnx` in the same directory |

Darknet also saves train/infer configs in the run directory. Keep them together with the matching weights.

### Files for Benchmarking

After all planned runs finish, keep:

- The six model directories from `weights/`, including ONNX, original `.weights`/`.pt`, saved CFG files, and training reports. `weights/pretrained/` is not needed for benchmarking.
- `datasets/vehicles/merged/images/val` and `labels/val`, plus `images/test` and `labels/test` for the final evaluation. Keep the original evaluation split, including empty TXT files.
- The merged dataset's `data.yaml`, `summary.json`, and metadata for reference.

Preserve the relative paths shown above, or adjust the benchmark arguments. The Rust benchmark reads the image and label directories directly, so it does not need the generated training paths from `data/generated/`. A speed-only run needs just the ONNX files and one image; mAP also needs evaluation images and labels.

## Benchmarking

Run this section after training and export. The Rust benchmark uses **od_opencv 0.8.2 / ONNX Runtime** for all six models, with 416x256 input and the same thresholds. OpenCV is not required. Keep each model at the path listed above and run commands from the project root.

### Build Benchmark

**CPU:**

```bash
cargo build --release --manifest-path benchmark/Cargo.toml
```

**CUDA 12 / cuDNN 9:**

```bash
ORT_CUDA_VERSION=12 cargo build --release \
  --manifest-path benchmark/Cargo.toml \
  --no-default-features --features ort-cuda
```

Keep the CUDA provider libraries alongside the executable. Runtime setup and Jetson Nano compatibility: [benchmark/README.md](benchmark/README.md#build).

For CUDA 13, unsupported GPU kernels, or system runtime linking, see the optional [CUDA troubleshooting](benchmark/README.md#cuda-troubleshooting) section. The commands below assume the runtime libraries are available without additional overrides.

### Run Benchmark

Open Bash in the project root and run this setup once before the examples below. Use the same terminal for each command; change `dataset_dir` if your dataset is elsewhere:

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

For speed-only runs, evaluator details, and the list of results to save, see [benchmark/README.md](benchmark/README.md).

**Native Darknet mAP check (optional):**

```bash
darknet detector map data/generated/vehicles.data \
  configs/yolov3-tiny-vehicles-infer.cfg \
  weights/yolov3-tiny-vehicles/yolov3-tiny-vehicles_best.weights -points 0
```

Use validation for tuning. For the final test, replace `data/generated/vehicles.data` with `data/generated/vehicles-test.data`. Evaluate selected models on test once the settings are fixed.

> **Note:** The Rust benchmark uses Pascal VOC 11-point interpolation after filtering at confidence 0.25. Its mAP is not directly interchangeable with Darknet `-points 0` or Ultralytics mAP. Use the same evaluator, split, and thresholds when comparing models.

## Benchmark results

Validation results on **Junction + MIO-TCD**, measured for all six models in one run on 2026-10-01. Test-set results are not available yet.

| Model | mAP@0.50 | Mean time (ms) | FPS |
| :--- | ---: | ---: | ---: |
| YOLOv3-tiny | 75.61% | 2.21 | 453.01 |
| YOLOv4-tiny | 75.78% | **2.12** | **470.78** |
| YOLOv5nu | 82.88% | 2.21 | 452.47 |
| YOLOv8n | 82.74% | 2.32 | 430.85 |
| YOLOv9t | 82.88% | 3.94 | 254.12 |
| YOLO11n | **84.70%** | 2.50 | 399.78 |

**Per-class AP@0.50:**

| Model | Car | Motorbike | Bus | Truck |
| :--- | ---: | ---: | ---: | ---: |
| YOLOv3-tiny | 80.33% | 60.88% | 90.27% | 70.95% |
| YOLOv4-tiny | 72.05% | 60.83% | 90.63% | 79.63% |
| YOLOv5nu | 89.50% | 71.21% | 90.61% | 80.21% |
| YOLOv8n | 89.50% | 70.64% | 90.57% | 80.27% |
| YOLOv9t | 89.55% | 70.89% | 90.67% | 80.39% |
| YOLO11n | 89.52% | 79.06% | 90.17% | 80.06% |

**Measurement setup:**

- NVIDIA GeForce RTX 5060 Ti, AMD Ryzen 7 7800X3D, ONNX Runtime 1.29.0 / CUDA (system build).
- Float32 ONNX input, batch 1, 416x256 with letterbox. Confidence 0.25, NMS IoU 0.45, evaluation IoU 0.50.
- Accuracy: all 7456 validation images, AP with 11-point interpolation after confidence filtering. These values are not directly interchangeable with Darknet or Ultralytics mAP.
- Speed: 50 warmup calls and 1000 timed calls on `images/val/00_1112-5923_frame_007783_1.jpg` (800x450). Timing includes preprocessing, inference and postprocessing, excluding image loading from disk.
- System runtime linked through `ORT_LIB_PATH=/usr/lib` and `ORT_PREFER_DYNAMIC_LINK=1`; cuDNN preloaded with `LD_PRELOAD=/usr/lib/libcudnn.so.9`. See [CUDA troubleshooting](benchmark/README.md#cuda-troubleshooting).

FPS is measured on this GPU and fixed image, not on Jetson Nano or an entire video pipeline. Small differences between models need repeated measurements before drawing conclusions. The validation set contains only 89 motorbike objects, so that class has fewer evaluation examples than the others.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/prepare_dataset.py --help
python3 scripts/train_ultralytics.py --help
bash -n scripts/train_darknet.sh scripts/generate_file_lists.sh scripts/create_videos.sh
```

Tests use small synthetic datasets and mocked training calls. No training or dataset downloads are started.
