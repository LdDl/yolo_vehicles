# YOLO Vehicles Detection

Training and benchmarking YOLOv3-tiny, YOLOv4-tiny, and YOLOv8n for vehicle detection.

All models are configured for **416x256** input size (16:9 aspect ratio) for fair performance comparison and optimized for edge devices like Jetson Nano.

> **Note on YOLOv8 training**: Ultralytics requires square images during training (`imgsz=416`), but YOLO models are fully convolutional and can be exported to any size. I've export it to 416x256 anyways for inference. The model works directly with rectangular input - no resizing to square required during inference (I believe so).

## Classes

| ID | Class |
|----|-------|
| 0 | car |
| 1 | motorbike |
| 2 | bus |
| 3 | truck |

## Project Structure

```
vehicles_yolo/
├── configs/
│   ├── yolov3-tiny-vehicles.cfg    # Darknet config (416x256, 4 classes)
│   ├── yolov4-tiny-vehicles.cfg    # Darknet config (416x256, 4 classes)
│   └── yolov8n-vehicles.yaml       # Ultralytics config (4 classes)
├── data/
│   ├── vehicles.names              # Class names
│   ├── vehicles.data               # Darknet data file
│   └── vehicles.yaml               # Ultralytics data file
├── scripts/
│   ├── prepare_dataset.py          # Dataset preparation
│   ├── generate_file_lists.sh      # Generate train/val file lists
│   ├── train_darknet.sh            # Train v3-tiny, v4-tiny
│   └── train_ultralytics.py        # Train v8n
├── benchmark/                      # Rust benchmark (uses od_opencv crate)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                 # CLI and orchestration
│       ├── benchmark.rs            # Speed and mAP evaluation
│       ├── metrics.rs              # IoU, AP, mAP calculation
│       ├── models.rs               # YoloModel trait
│       └── types.rs                # Constants and structs
├── weights/                        # Trained weights output
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Download Dataset

Download the AIC HCMC 2020 dataset from [Kaggle](https://www.kaggle.com/datasets/hungkhoi/vehicle-counting-aic-hcmc-2020).

```bash
# Extract the dataset
tar -xzf aic_hcmc2020.tar.gz
```

### 2. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies (for YOLOv8)
pip install -r requirements.txt
```

### 3. Prepare Dataset

```bash
python scripts/prepare_dataset.py --dataset-dir ./aic_hcmc2020/aic_hcmc2020
```

This will:
- Remap class IDs if needed
- Copy labels to images directory
- Generate train/val file lists for both Darknet and Ultralytics

If your dataset already has the correct structure (`images/train`, `images/val`, `labels/train`, `labels/val`), you can skip the Python script and just generate file lists:

```bash
./scripts/generate_file_lists.sh
```

**Note:** Darknet expects label files in the same directory as images. Copy them if needed:

```bash
cp aic_hcmc2020/labels/train/*.txt aic_hcmc2020/images/train/
cp aic_hcmc2020/labels/val/*.txt aic_hcmc2020/images/val/
```

### 4. Train Models

#### YOLOv3-tiny / YOLOv4-tiny (Darknet)

Requires [AlexeyAB's Darknet](https://github.com/AlexeyAB/darknet) compiled with:
```makefile
GPU=1
CUDNN=1
OPENCV=1
```

**Arch Linux / CachyOS users (I'm using Arch btw):** CUDA is installed at `/opt/cuda/` instead of `/usr/local/cuda/`.
Edit `darknet/Makefile` and replace all occurrences of `/usr/local/cuda/` with `/opt/cuda/`:
```makefile
COMMON+= -DGPU -I/opt/cuda/include/
LDFLAGS+= -L/opt/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
CFLAGS+= -DCUDNN -I/opt/cuda/include
LDFLAGS+= -L/opt/cuda/lib64 -lcudnn
```

Also set ARCH for your GPU (e.g., RTX 3060 = Ampere, compute 8.6):
```makefile
ARCH= -gencode arch=compute_86,code=[sm_86,compute_86]
```

<details>
<summary><strong>CUDA 13+ compatibility fix</strong></summary>

If you get `cudaHostAlloc` incompatible pointer type errors, you need to add `(void**)` casts in these files:

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

**src/gaussian_yolo_layer.c** (lines ~70, 77, 109, 118):
```c
// Same pattern as yolo_layer.c - add (void**) cast to all cudaHostAlloc calls
```

</details>

Build Darknet:
```bash
cd darknet
make clean && make -j$(nproc)
```

Install system-wide (optional):
```bash
sudo cp darknet /usr/local/bin/
sudo cp libdarknet.so /usr/local/lib/
sudo ldconfig
```

Train:
```bash
./scripts/train_darknet.sh v3-tiny
```

```bash
./scripts/train_darknet.sh v4-tiny
```

#### YOLOv8n (Ultralytics)

```bash
python scripts/train_ultralytics.py --epochs 100
```

Options:
- `--pretrained` - Start from COCO pretrained weights (recommended)
- `--batch 16` - Adjust batch size for your GPU
- `--device 0` - CUDA device ID

## Inference (Testing Detection)

### YOLOv3-tiny / YOLOv4-tiny (Darknet)

```bash
darknet detector test data/vehicles.data \
    configs/yolov3-tiny-vehicles-infer.cfg \
    weights/yolov3-tiny-vehicles_best.weights \
    path/to/image.jpg
```

Save output to file instead of displaying:

```bash
darknet detector test data/vehicles.data \
    configs/yolov3-tiny-vehicles-infer.cfg \
    weights/yolov3-tiny-vehicles_best.weights \
    path/to/image.jpg \
    -dont_show -out_filename predictions.jpg
```

### YOLOv8n (Ultralytics)

```bash
yolo detect predict model=weights/yolov8n-vehicles/weights/best.pt source=path/to/image.jpg
```

Results are saved to `runs/detect/predict/` by default.

## Benchmarking

The benchmark uses [od_opencv](https://crates.io/crates/od_opencv) Rust crate for realistic deployment performance on edge devices like Jetson Nano. It measures both **FPS** and **mAP@0.50** using Pascal VOC 11-point interpolation.

### Build Benchmark

```bash
cd benchmark
cargo build --release
```

### Run Benchmark

**Speed + mAP evaluation (recommended):**
```bash
./target/release/benchmark \
    --val-images ../aic_hcmc2020/images/val \
    --val-labels ../aic_hcmc2020/labels/val \
    --v3-weights ../weights/yolov3-tiny-vehicles_best.weights \
    --v3-cfg ../configs/yolov3-tiny-vehicles-infer.cfg
```

**With CUDA acceleration:**
```bash
./target/release/benchmark \
    --cuda \
    --val-images ../aic_hcmc2020/images/val \
    --val-labels ../aic_hcmc2020/labels/val \
    --v3-weights ../weights/yolov3-tiny-vehicles_best.weights \
    --v3-cfg ../configs/yolov3-tiny-vehicles-infer.cfg
```

**Compare multiple models:**
```bash
./target/release/benchmark \
    --val-images ../aic_hcmc2020/images/val \
    --val-labels ../aic_hcmc2020/labels/val \
    --v3-weights ../weights/yolov3-tiny-vehicles_best.weights \
    --v3-cfg ../configs/yolov3-tiny-vehicles-infer.cfg \
    --v4-weights ../weights/yolov4-tiny-vehicles_best.weights \
    --v4-cfg ../configs/yolov4-tiny-vehicles-infer.cfg \
    --v8-onnx ../weights/yolov8n-vehicles/weights/best.onnx
```

**Single image speed test:**
```bash
./target/release/benchmark \
    --image ../aic_hcmc2020/images/val/cam_01_000001.jpg \
    --iterations 100 \
    --warmup 10 \
    --v3-weights ../weights/yolov3-tiny-vehicles_best.weights \
    --v3-cfg ../configs/yolov3-tiny-vehicles-infer.cfg
```

Options:
- `--cuda` - Use CUDA backend (requires OpenCV with CUDA)
- `--val-images` / `--val-labels` - Validation set for mAP calculation
- `--max-images N` - Limit images for faster testing
- `--image` - Single image for dedicated speed benchmark
- `--iterations N` - Number of speed benchmark iterations
- `--warmup N` - Warmup iterations before benchmarking

### Example Output

```
+----------------------------------------------------------+
|                    BENCHMARK SUMMARY                     |
+----------------------------------------------------------+

==================================================
Model: YOLOv3-tiny
==================================================
Iterations: 4021
Total time: 19.27s
Mean time:  4.79ms
Min time:   2.12ms
Max time:   441.03ms
FPS:        208.70

mAP@0.50:   70.13%

Per-class AP@0.50:
  car: 76.80%
  motorbike: 68.20%
  bus: 66.34%
  truck: 69.18%
```

## Benchmark Results

### Speed Comparison (416x256)

| Model | Backend | Mean (ms) | Min (ms) | FPS |
|-------|---------|-----------|----------|-----|
| YOLOv3-tiny | CPU | 16.60 | 12.17 | 60.24 |
| YOLOv3-tiny | CUDA (RTX 3060) | 4.79 | 2.12 | 208.70 |
| YOLOv4-tiny | CPU | - | - | - |
| YOLOv4-tiny | CUDA (RTX 3060) | - | - | - |
| YOLOv8n | CPU | - | - | - |
| YOLOv8n | CUDA (RTX 3060) | - | - | - |

### mAP Comparison (416x256, IoU=0.50)

| Model | mAP@0.50 | car | motorbike | bus | truck |
|-------|----------|-----|-----------|-----|-------|
| YOLOv3-tiny | 70.13% | 76.80% | 68.20% | 66.34% | 69.18% |
| YOLOv4-tiny | - | - | - | - | - |
| YOLOv8n | - | - | - | - | - |

> **Note:** mAP calculated using Pascal VOC 11-point interpolation. Darknet reports higher values (~80%) using all-point interpolation (COCO style).

## Model Comparison

| Model | Parameters | Format | Input Size |
|-------|------------|--------|------------|
| YOLOv3-tiny | ~8.7M | .cfg + .weights | 416x256 |
| YOLOv4-tiny | ~6M | .cfg + .weights | 416x256 |
| YOLOv8n | ~3.2M | .onnx | 416x256 |

## Deployment to Jetson Nano

For Jetson Nano deployment using Rust:

1. Cross-compile the benchmark or build on device
2. Use `--cuda` flag for GPU acceleration
3. OpenCV must be compiled with CUDA support

Example with od_opencv in your Rust project:

```rust
use od_opencv::model_classic::ModelYOLOClassic;
use opencv::dnn::{DNN_BACKEND_CUDA, DNN_TARGET_CUDA};

let model = ModelYOLOClassic::new_from_darknet_file(
    "weights/yolov4-tiny-vehicles.weights",
    "configs/yolov4-tiny-vehicles.cfg",
    (416, 256),
    DNN_BACKEND_CUDA,
    DNN_TARGET_CUDA,
    vec![],
)?;
```

## Legacy Files

Old configuration files are kept for reference:
- `vehicles.cfg` - Original YOLOv3-tiny at 352x256
- `vehicles_v4.cfg` - Original YOLOv4-tiny at 384x384

## License

MIT
