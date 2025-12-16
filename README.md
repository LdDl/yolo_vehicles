# YOLO Vehicles Detection

Training and benchmarking YOLOv3-tiny, YOLOv4-tiny, and YOLOv8n for vehicle detection.

All models are configured for **416x256** input size (16:9 aspect ratio) for fair performance comparison and optimized for edge devices like Jetson Nano.

> **Note on YOLOv8 training**: Ultralytics `imgsz` only accepts a single integer during training (e.g., `imgsz=416`). Use `rect=True` to enable rectangular batching that adapts to each batch's aspect ratio. See [ultralytics#235](https://github.com/ultralytics/ultralytics/issues/235).
>
> **Note on YOLOv8 export**: For export, `imgsz` accepts `[height, width]`. Ultralytics uses height-first order, while Darknet uses width-first. To export a 416x256 (width x height) ONNX matching Darknet configs, use `imgsz=256,416`.

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
    --detailed \
    --val-images ../aic_hcmc2020/images/val \
    --val-labels ../aic_hcmc2020/labels/val \
    --v3-weights ../weights/yolov3-tiny-vehicles_best.weights \
    --v3-cfg ../configs/yolov3-tiny-vehicles-infer.cfg
```

**With CUDA acceleration:**
```bash
./target/release/benchmark \
    --cuda \
    --detailed \
    --val-images ../aic_hcmc2020/images/val \
    --val-labels ../aic_hcmc2020/labels/val \
    --v3-weights ../weights/yolov3-tiny-vehicles_best.weights \
    --v3-cfg ../configs/yolov3-tiny-vehicles-infer.cfg
```

**Compare multiple models:**
```bash
./target/release/benchmark \
    --detailed \
    --cuda \
    --val-images ../aic_hcmc2020/images/val \
    --val-labels ../aic_hcmc2020/labels/val \
    --v3-weights ../weights/yolov3-tiny-vehicles_best.weights \
    --v3-cfg ../configs/yolov3-tiny-vehicles-infer.cfg \
    --v4-weights ../weights/yolov4-tiny-vehicles_final.weights \
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
Comparison (416x256, CUDA):
---------------------------------------------------------------------------
Model              Mean (ms)          FPS     mAP@0.50 Relative FPS
---------------------------------------------------------------------------
YOLOv3-tiny             4.69       213.10       70.13%        1.00x
YOLOv4-tiny             4.95       202.04       63.66%        0.95x
YOLOv8n                 5.84       171.13       65.37%        0.80x
---------------------------------------------------------------------------
```

## Benchmark Results

### Speed Comparison (416x256)

| Model | Backend | Mean (ms) | Min (ms) | FPS |
|-------|---------|-----------|----------|-----|
| YOLOv3-tiny | CUDA (RTX 3060) | 4.69 | 2.15 | 213.10 |
| YOLOv4-tiny | CUDA (RTX 3060) | 4.95 | 2.41 | 202.04 |
| YOLOv8n | CUDA (RTX 3060) | 5.84 | 5.30 | 171.13 |

### mAP Comparison (416x256, IoU=0.50)

| Model | mAP@0.50 | car | motorbike | bus | truck |
|-------|----------|-----|-----------|-----|-------|
| YOLOv3-tiny | **69.96%** | 76.68% | 68.11% | 66.19% | 68.88% |
| YOLOv8n | 65.27% | 69.06% | 70.82% | 51.94% | 69.25% |
| YOLOv4-tiny | 63.52% | 61.95% | 44.58% | 69.41% | 78.13% |

> **Note:** mAP calculated using Pascal VOC 11-point interpolation. Darknet reports higher values (~80%) using all-point interpolation (COCO style). Results may vary slightly (~0.2%) between runs due to GPU floating-point non-determinism.

### F1 Score Comparison (416x256, IoU=0.50)

| Model | Micro F1 | Macro F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| YOLOv8n | **82.19%** | **81.74%** | 87.25% | 77.68% |
| YOLOv3-tiny | 78.53% | 80.20% | 79.81% | 77.29% |
| YOLOv4-tiny | 67.33% | 77.00% | 89.73% | 53.88% |

**Per-class F1 Scores:**

| Model | car | motorbike | bus | truck |
|-------|-----|-----------|-----|-------|
| YOLOv8n | **82.78%** | 82.16% | 80.35% | 81.69% |
| YOLOv3-tiny | 80.77% | 77.52% | **81.15%** | **81.37%** |
| YOLOv4-tiny | 77.79% | 60.74% | **85.13%** | 84.35% |

> **Key insight:** YOLOv8n has the best overall F1 score despite lower mAP, indicating more balanced precision/recall. YOLOv4-tiny has very high precision (89.73%) but low recall (53.88%), missing many objects.

### Confusion Matrices

<details>
<summary><strong>YOLOv3-tiny Confusion Matrix</strong></summary>

```
 Actual\Pred       car motorbike       bus     truck        BG
--------------------------------------------------------------
         car      5621        53        19        47      1204
   motorbike        45     20364         .        12      6676
         bus       101         .       921        61       169
       truck       105        25        66      2322       539
          BG      1222      5058       174       404         .
```

</details>

<details>
<summary><strong>YOLOv4-tiny Confusion Matrix</strong></summary>

```
 Actual\Pred       car motorbike       bus     truck        BG
--------------------------------------------------------------
         car      4731        13        16        32      2152
   motorbike        32     12336         .        12     14717
         bus        43         .       973        53       183
       truck        49         7        28      2469       504
          BG       425      1210       113       315         .
```

Note: YOLOv4-tiny misses 14,717 motorbikes (54% FN rate), explaining its low recall.

</details>

<details>
<summary><strong>YOLOv8n Confusion Matrix</strong></summary>

```
 Actual\Pred       car motorbike       bus     truck        BG
--------------------------------------------------------------
         car      5501        27        18        45      1353
   motorbike        49     20752         .        13      6283
         bus       125         .       734       157       236
       truck        93        26        39      2342       557
          BG       669      2679        66       278         .
```

</details>

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
