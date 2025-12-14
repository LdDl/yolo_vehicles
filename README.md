# YOLO Vehicles Detection

Training and benchmarking YOLOv3-tiny, YOLOv4-tiny, and YOLOv8n for vehicle detection.

All models are configured for **416x416** input size for fair performance comparison.

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
│   ├── yolov3-tiny-vehicles.cfg    # Darknet config (416x416, 4 classes)
│   ├── yolov4-tiny-vehicles.cfg    # Darknet config (416x416, 4 classes)
│   └── yolov8n-vehicles.yaml       # Ultralytics config (4 classes)
├── data/
│   ├── vehicles.names              # Class names
│   ├── vehicles.data               # Darknet data file
│   └── vehicles.yaml               # Ultralytics data file
├── scripts/
│   ├── prepare_dataset.py          # Dataset preparation
│   ├── train_darknet.sh            # Train v3-tiny, v4-tiny
│   └── train_ultralytics.py        # Train v8n
├── benchmark/                      # Rust benchmark (uses od_opencv crate)
│   ├── Cargo.toml
│   └── src/main.rs
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
# Python dependencies (for YOLOv8 and dataset preparation)
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
python scripts/train_ultralytics.py --epochs 100 --imgsz 416
```

Options:
- `--pretrained` - Start from COCO pretrained weights (recommended)
- `--batch 16` - Adjust batch size for your GPU
- `--device 0` - CUDA device ID

## Benchmarking

The benchmark uses [od_opencv](https://crates.io/crates/od_opencv) Rust crate for realistic deployment performance on edge devices like Jetson Nano.

### Build Benchmark

```bash
cd benchmark
cargo build --release
```

### Run Benchmark

```bash
cargo run --release -- \
    --image ../images_samples/1.jpg \
    --v3-weights ../weights/yolov3-tiny-vehicles.weights \
    --v3-cfg ../configs/yolov3-tiny-vehicles.cfg \
    --v4-weights ../weights/yolov4-tiny-vehicles.weights \
    --v4-cfg ../configs/yolov4-tiny-vehicles.cfg \
    --v8-onnx ../weights/yolov8n-vehicles.onnx \
    --iterations 100
```

Options:
- `--cuda` - Use CUDA backend (requires OpenCV with CUDA)
- `--iterations N` - Number of benchmark iterations
- `--warmup N` - Warmup iterations before benchmarking

### Expected Output

```
╔════════════════════════════════════════════════════════╗
║                    BENCHMARK SUMMARY                    ║
╚════════════════════════════════════════════════════════╝

Comparison (416x416, CPU):
------------------------------------------------------------
Model           Mean (ms)          FPS     Relative
------------------------------------------------------------
YOLOv3-tiny         XX.XX        XX.XX        1.00x
YOLOv4-tiny         XX.XX        XX.XX        X.XXx
YOLOv8n             XX.XX        XX.XX        X.XXx
------------------------------------------------------------
```

## Model Comparison

| Model | Parameters | Format | Input Size |
|-------|------------|--------|------------|
| YOLOv3-tiny | ~8.7M | .cfg + .weights | 416x416 |
| YOLOv4-tiny | ~6M | .cfg + .weights | 416x416 |
| YOLOv8n | ~3.2M | .onnx | 416x416 |

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
    (416, 416),
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
