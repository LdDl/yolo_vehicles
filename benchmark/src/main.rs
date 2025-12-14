use clap::Parser;
use od_opencv::{model_classic::ModelYOLOClassic, model_ultralytics::ModelUltralyticsV8};
use opencv::{
    dnn::{DNN_BACKEND_CUDA, DNN_BACKEND_OPENCV, DNN_TARGET_CPU, DNN_TARGET_CUDA},
    imgcodecs::imread,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const CLASSES: [&str; 4] = ["car", "motorbike", "bus", "truck"];
const NET_WIDTH: i32 = 416;
const NET_HEIGHT: i32 = 256;
const CONF_THRESHOLD: f32 = 0.25;
const NMS_THRESHOLD: f32 = 0.45;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to test image
    #[arg(short, long)]
    image: PathBuf,

    /// Number of inference iterations for benchmarking
    #[arg(short, long, default_value_t = 100)]
    iterations: u32,

    /// Use CUDA backend (requires OpenCV with CUDA support)
    #[arg(long)]
    cuda: bool,

    /// Path to YOLOv3-tiny weights (.weights)
    #[arg(long)]
    v3_weights: Option<PathBuf>,

    /// Path to YOLOv3-tiny config (.cfg)
    #[arg(long)]
    v3_cfg: Option<PathBuf>,

    /// Path to YOLOv4-tiny weights (.weights)
    #[arg(long)]
    v4_weights: Option<PathBuf>,

    /// Path to YOLOv4-tiny config (.cfg)
    #[arg(long)]
    v4_cfg: Option<PathBuf>,

    /// Path to YOLOv8n ONNX model (.onnx)
    #[arg(long)]
    v8_onnx: Option<PathBuf>,

    /// Warmup iterations before benchmarking
    #[arg(long, default_value_t = 10)]
    warmup: u32,
}

struct BenchmarkResult {
    model_name: String,
    iterations: u32,
    total_time: Duration,
    mean_time: Duration,
    min_time: Duration,
    max_time: Duration,
    fps: f64,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("\n{}", "=".repeat(50));
        println!("Model: {}", self.model_name);
        println!("{}", "=".repeat(50));
        println!("Iterations: {}", self.iterations);
        println!("Total time: {:.2?}", self.total_time);
        println!("Mean time:  {:.2?}", self.mean_time);
        println!("Min time:   {:.2?}", self.min_time);
        println!("Max time:   {:.2?}", self.max_time);
        println!("FPS:        {:.2}", self.fps);
    }
}

fn benchmark_model<F>(
    model_name: &str,
    iterations: u32,
    warmup: u32,
    mut inference_fn: F,
) -> BenchmarkResult
where
    F: FnMut() -> Result<(), opencv::Error>,
{
    // Warmup
    println!("  Warming up ({} iterations)...", warmup);
    for _ in 0..warmup {
        inference_fn().expect("Warmup inference failed");
    }

    // Benchmark
    println!("  Benchmarking ({} iterations)...", iterations);
    let mut times = Vec::with_capacity(iterations as usize);

    for i in 0..iterations {
        let start = Instant::now();
        inference_fn().expect("Benchmark inference failed");
        let elapsed = start.elapsed();
        times.push(elapsed);

        if (i + 1) % 20 == 0 {
            println!("    Progress: {}/{}", i + 1, iterations);
        }
    }

    let total_time: Duration = times.iter().sum();
    let mean_time = total_time / iterations;
    let min_time = *times.iter().min().unwrap();
    let max_time = *times.iter().max().unwrap();
    let fps = iterations as f64 / total_time.as_secs_f64();

    BenchmarkResult {
        model_name: model_name.to_string(),
        iterations,
        total_time,
        mean_time,
        min_time,
        max_time,
        fps,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Determine backend/target
    let (backend, target, backend_name) = if args.cuda {
        (DNN_BACKEND_CUDA, DNN_TARGET_CUDA, "CUDA")
    } else {
        (DNN_BACKEND_OPENCV, DNN_TARGET_CPU, "CPU")
    };

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║        YOLO Vehicles Detection Benchmark               ║");
    println!("╚════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  Image: {:?}", args.image);
    println!("  Backend: {}", backend_name);
    println!("  Input size: {}x{}", NET_WIDTH, NET_HEIGHT);
    println!("  Iterations: {}", args.iterations);
    println!("  Warmup: {}", args.warmup);
    println!("  Confidence threshold: {}", CONF_THRESHOLD);
    println!("  NMS threshold: {}", NMS_THRESHOLD);

    // Load test image
    let image = imread(args.image.to_str().unwrap(), 1)?;
    if image.empty() {
        return Err("Failed to load image".into());
    }
    println!("  Image loaded: {}x{}", image.cols(), image.rows());

    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Benchmark YOLOv3-tiny
    if let (Some(weights), Some(cfg)) = (&args.v3_weights, &args.v3_cfg) {
        println!("\n[YOLOv3-tiny]");
        println!("  Loading model...");

        let mut model = ModelYOLOClassic::new_from_darknet_file(
            weights.to_str().unwrap(),
            cfg.to_str().unwrap(),
            (NET_WIDTH, NET_HEIGHT),
            backend,
            target,
            vec![],
        )?;

        let image_clone = image.clone();
        let result = benchmark_model("YOLOv3-tiny", args.iterations, args.warmup, || {
            model.forward(&image_clone, CONF_THRESHOLD, NMS_THRESHOLD)?;
            Ok(())
        });
        results.push(result);
    }

    // Benchmark YOLOv4-tiny
    if let (Some(weights), Some(cfg)) = (&args.v4_weights, &args.v4_cfg) {
        println!("\n[YOLOv4-tiny]");
        println!("  Loading model...");

        let mut model = ModelYOLOClassic::new_from_darknet_file(
            weights.to_str().unwrap(),
            cfg.to_str().unwrap(),
            (NET_WIDTH, NET_HEIGHT),
            backend,
            target,
            vec![],
        )?;

        let image_clone = image.clone();
        let result = benchmark_model("YOLOv4-tiny", args.iterations, args.warmup, || {
            model.forward(&image_clone, CONF_THRESHOLD, NMS_THRESHOLD)?;
            Ok(())
        });
        results.push(result);
    }

    // Benchmark YOLOv8n
    if let Some(onnx) = &args.v8_onnx {
        println!("\n[YOLOv8n]");
        println!("  Loading model...");

        let mut model = ModelUltralyticsV8::new_from_onnx_file(
            onnx.to_str().unwrap(),
            (NET_WIDTH, NET_HEIGHT),
            backend,
            target,
            vec![],
        )?;

        let image_clone = image.clone();
        let result = benchmark_model("YOLOv8n", args.iterations, args.warmup, || {
            model.forward(&image_clone, CONF_THRESHOLD, NMS_THRESHOLD)?;
            Ok(())
        });
        results.push(result);
    }

    // Print summary
    println!("\n");
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK SUMMARY                    ║");
    println!("╚════════════════════════════════════════════════════════╝");

    for result in &results {
        result.print();
    }

    // Comparison table
    if results.len() > 1 {
        println!("\n\nComparison ({}x{}, {}):", NET_WIDTH, NET_HEIGHT, backend_name);
        println!("{:-<60}", "");
        println!(
            "{:<15} {:>12} {:>12} {:>12}",
            "Model", "Mean (ms)", "FPS", "Relative"
        );
        println!("{:-<60}", "");

        let baseline_fps = results[0].fps;
        for result in &results {
            let relative = result.fps / baseline_fps;
            println!(
                "{:<15} {:>12.2} {:>12.2} {:>11.2}x",
                result.model_name,
                result.mean_time.as_secs_f64() * 1000.0,
                result.fps,
                relative
            );
        }
        println!("{:-<60}", "");
    }

    if results.is_empty() {
        println!("\nNo models were benchmarked. Please provide model paths.");
        println!("Example:");
        println!("  cargo run --release -- \\");
        println!("    --image ../images_samples/1.jpg \\");
        println!("    --v3-weights ../weights/yolov3-tiny-vehicles.weights \\");
        println!("    --v3-cfg ../configs/yolov3-tiny-vehicles.cfg \\");
        println!("    --v4-weights ../weights/yolov4-tiny-vehicles.weights \\");
        println!("    --v4-cfg ../configs/yolov4-tiny-vehicles.cfg \\");
        println!("    --v8-onnx ../weights/yolov8n-vehicles.onnx");
    }

    Ok(())
}
