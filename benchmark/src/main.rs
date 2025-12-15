mod benchmark;
mod metrics;
mod models;
mod types;

use clap::Parser;
use od_opencv::{model_classic::ModelYOLOClassic, model_ultralytics::ModelUltralyticsV8};
use opencv::{
    dnn::{DNN_BACKEND_CUDA, DNN_BACKEND_OPENCV, DNN_TARGET_CPU, DNN_TARGET_CUDA},
    imgcodecs::imread,
    prelude::*,
};
use std::path::PathBuf;

use benchmark::{benchmark_speed, run_map_evaluation};
use types::{BenchmarkResult, CONF_THRESHOLD, NET_HEIGHT, NET_WIDTH, NMS_THRESHOLD, IOU_THRESHOLD};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to test image (for speed benchmark)
    #[arg(short, long)]
    image: Option<PathBuf>,

    /// Path to validation images directory (for mAP calculation)
    #[arg(long)]
    val_images: Option<PathBuf>,

    /// Path to validation labels directory (for mAP calculation)
    #[arg(long)]
    val_labels: Option<PathBuf>,

    /// Number of inference iterations for speed benchmarking
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

    /// Maximum images to process for mAP (0 = all)
    #[arg(long, default_value_t = 0)]
    max_images: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Determine backend/target
    let (backend, target, backend_name) = if args.cuda {
        (DNN_BACKEND_CUDA, DNN_TARGET_CUDA, "CUDA")
    } else {
        (DNN_BACKEND_OPENCV, DNN_TARGET_CPU, "CPU")
    };

    print_header();
    print_config(&args, backend_name);

    // Load test image for speed benchmark
    let speed_image = load_speed_image(&args)?;

    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Benchmark YOLOv3-tiny
    if let (Some(weights), Some(cfg)) = (&args.v3_weights, &args.v3_cfg) {
        let result = benchmark_darknet_model(
            "YOLOv3-tiny",
            weights,
            cfg,
            &args,
            &speed_image,
            backend,
            target,
        )?;
        results.push(result);
    }

    // Benchmark YOLOv4-tiny
    if let (Some(weights), Some(cfg)) = (&args.v4_weights, &args.v4_cfg) {
        let result = benchmark_darknet_model(
            "YOLOv4-tiny",
            weights,
            cfg,
            &args,
            &speed_image,
            backend,
            target,
        )?;
        results.push(result);
    }

    // Benchmark YOLOv8n
    if let Some(onnx) = &args.v8_onnx {
        let result = benchmark_onnx_model("YOLOv8n", onnx, &args, &speed_image, backend, target)?;
        results.push(result);
    }

    // Print results
    print_summary(&results, backend_name);

    if results.is_empty() {
        print_usage();
    }

    Ok(())
}

fn print_header() {
    println!("+----------------------------------------------------------+");
    println!("|        YOLO Vehicles Detection Benchmark                 |");
    println!("+----------------------------------------------------------+");
    println!();
}

fn print_config(args: &Args, backend_name: &str) {
    println!("Configuration:");
    println!("  Backend: {}", backend_name);
    println!("  Input size: {}x{}", NET_WIDTH, NET_HEIGHT);
    println!("  Confidence threshold: {}", CONF_THRESHOLD);
    println!("  NMS threshold: {}", NMS_THRESHOLD);
    println!("  IoU threshold (mAP): {}", IOU_THRESHOLD);

    if let Some(ref img) = args.image {
        println!("  Speed test image: {:?}", img);
        println!("  Iterations: {}", args.iterations);
        println!("  Warmup: {}", args.warmup);
    }

    if let (Some(ref val_img), Some(ref val_lbl)) = (&args.val_images, &args.val_labels) {
        println!("  Validation images: {:?}", val_img);
        println!("  Validation labels: {:?}", val_lbl);
        if args.max_images > 0 {
            println!("  Max images for mAP: {}", args.max_images);
        }
    }
}

fn load_speed_image(args: &Args) -> Result<Option<opencv::core::Mat>, Box<dyn std::error::Error>> {
    if let Some(ref img_path) = args.image {
        let image = imread(img_path.to_str().unwrap(), 1)?;
        if image.empty() {
            return Err("Failed to load speed test image".into());
        }
        println!("  Speed test image loaded: {}x{}", image.cols(), image.rows());
        Ok(Some(image))
    } else {
        Ok(None)
    }
}

fn benchmark_darknet_model(
    model_name: &str,
    weights: &PathBuf,
    cfg: &PathBuf,
    args: &Args,
    speed_image: &Option<opencv::core::Mat>,
    backend: i32,
    target: i32,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    println!("\n[{}]", model_name);
    println!("  Loading model...");

    let mut model = ModelYOLOClassic::new_from_darknet_file(
        weights.to_str().unwrap(),
        cfg.to_str().unwrap(),
        (NET_WIDTH, NET_HEIGHT),
        backend,
        target,
        vec![],
    )?;

    let mut result = BenchmarkResult::new(model_name, args.iterations);

    // Speed benchmark
    if let Some(ref image) = speed_image {
        let image_clone = image.clone();
        let (total, min, max) = benchmark_speed(model_name, args.iterations, args.warmup, || {
            model.forward(&image_clone, CONF_THRESHOLD, NMS_THRESHOLD)?;
            Ok(())
        });
        result.total_time = total;
        result.min_time = min;
        result.max_time = max;
        result.mean_time = total / args.iterations;
        result.fps = args.iterations as f64 / total.as_secs_f64();
    }

    // mAP evaluation
    if let (Some(ref val_img), Some(ref val_lbl)) = (&args.val_images, &args.val_labels) {
        let map_result = run_map_evaluation(&mut model, val_img, val_lbl, args.max_images)?;
        result.map50 = Some(map_result.map);
        result.per_class_ap = Some(map_result.per_class_ap);

        // Use mAP timing if no speed benchmark was run
        if speed_image.is_none() {
            result.iterations = map_result.num_images as u32;
            result.total_time = map_result.total_inference_time;
            result.mean_time = map_result.mean_inference_time;
            result.min_time = map_result.min_inference_time;
            result.max_time = map_result.max_inference_time;
            if map_result.total_inference_time.as_secs_f64() > 0.0 {
                result.fps = map_result.num_images as f64 / map_result.total_inference_time.as_secs_f64();
            }
        }
    }

    Ok(result)
}

fn benchmark_onnx_model(
    model_name: &str,
    onnx: &PathBuf,
    args: &Args,
    speed_image: &Option<opencv::core::Mat>,
    backend: i32,
    target: i32,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    println!("\n[{}]", model_name);
    println!("  Loading model...");

    let mut model = ModelUltralyticsV8::new_from_onnx_file(
        onnx.to_str().unwrap(),
        (NET_WIDTH, NET_HEIGHT),
        backend,
        target,
        vec![],
    )?;

    let mut result = BenchmarkResult::new(model_name, args.iterations);

    // Speed benchmark
    if let Some(ref image) = speed_image {
        let image_clone = image.clone();
        let (total, min, max) = benchmark_speed(model_name, args.iterations, args.warmup, || {
            model.forward(&image_clone, CONF_THRESHOLD, NMS_THRESHOLD)?;
            Ok(())
        });
        result.total_time = total;
        result.min_time = min;
        result.max_time = max;
        result.mean_time = total / args.iterations;
        result.fps = args.iterations as f64 / total.as_secs_f64();
    }

    // mAP evaluation
    if let (Some(ref val_img), Some(ref val_lbl)) = (&args.val_images, &args.val_labels) {
        let map_result = run_map_evaluation(&mut model, val_img, val_lbl, args.max_images)?;
        result.map50 = Some(map_result.map);
        result.per_class_ap = Some(map_result.per_class_ap);

        // Use mAP timing if no speed benchmark was run
        if speed_image.is_none() {
            result.iterations = map_result.num_images as u32;
            result.total_time = map_result.total_inference_time;
            result.mean_time = map_result.mean_inference_time;
            result.min_time = map_result.min_inference_time;
            result.max_time = map_result.max_inference_time;
            if map_result.total_inference_time.as_secs_f64() > 0.0 {
                result.fps = map_result.num_images as f64 / map_result.total_inference_time.as_secs_f64();
            }
        }
    }

    Ok(result)
}

fn print_summary(results: &[BenchmarkResult], backend_name: &str) {
    println!("\n");
    println!("+----------------------------------------------------------+");
    println!("|                    BENCHMARK SUMMARY                     |");
    println!("+----------------------------------------------------------+");

    for result in results {
        result.print();
    }

    // Comparison table
    if results.len() > 1 {
        println!("\n\nComparison ({}x{}, {}):", NET_WIDTH, NET_HEIGHT, backend_name);
        println!("{:-<75}", "");
        println!(
            "{:<15} {:>12} {:>12} {:>12} {:>12}",
            "Model", "Mean (ms)", "FPS", "mAP@0.50", "Relative FPS"
        );
        println!("{:-<75}", "");

        let baseline_fps = results[0].fps;
        for result in results {
            let relative = if baseline_fps > 0.0 {
                result.fps / baseline_fps
            } else {
                0.0
            };
            let map_str = result
                .map50
                .map(|m| format!("{:.2}%", m * 100.0))
                .unwrap_or_else(|| "-".to_string());

            println!(
                "{:<15} {:>12.2} {:>12.2} {:>12} {:>11.2}x",
                result.model_name,
                result.mean_time.as_secs_f64() * 1000.0,
                result.fps,
                map_str,
                relative
            );
        }
        println!("{:-<75}", "");
    }
}

fn print_usage() {
    println!("\nNo models were benchmarked. Please provide model paths.");
    println!("\nSpeed benchmark example:");
    println!("  cargo run --release -- \\");
    println!("    --image ../aic_hcmc2020/images/val/cam_01_000001.jpg \\");
    println!("    --v3-weights ../weights/yolov3-tiny-vehicles_best.weights \\");
    println!("    --v3-cfg ../configs/yolov3-tiny-vehicles-infer.cfg");
    println!("\nmAP evaluation example:");
    println!("  cargo run --release -- \\");
    println!("    --val-images ../aic_hcmc2020/images/val \\");
    println!("    --val-labels ../aic_hcmc2020/labels/val \\");
    println!("    --v3-weights ../weights/yolov3-tiny-vehicles_best.weights \\");
    println!("    --v3-cfg ../configs/yolov3-tiny-vehicles-infer.cfg");
    println!("\nFull benchmark (speed + mAP) example:");
    println!("  cargo run --release -- \\");
    println!("    --image ../aic_hcmc2020/images/val/cam_01_000001.jpg \\");
    println!("    --val-images ../aic_hcmc2020/images/val \\");
    println!("    --val-labels ../aic_hcmc2020/labels/val \\");
    println!("    --v3-weights ../weights/yolov3-tiny-vehicles_best.weights \\");
    println!("    --v3-cfg ../configs/yolov3-tiny-vehicles-infer.cfg \\");
    println!("    --v4-weights ../weights/yolov4-tiny-vehicles_best.weights \\");
    println!("    --v4-cfg ../configs/yolov4-tiny-vehicles-infer.cfg \\");
    println!("    --v8-onnx ../weights/yolov8n-vehicles/weights/best.onnx");
}
