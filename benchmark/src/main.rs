mod benchmark;
mod metrics;
mod models;
mod types;

use clap::Parser;
use models::{check_device, load_image, load_model};
use od_opencv::ImageBuffer;
use std::path::{Path, PathBuf};

use benchmark::{benchmark_speed, run_map_evaluation, run_map_evaluation_debug};
use types::{
    BenchmarkResult, PerClassMetrics, CLASSES, CONF_THRESHOLD, IOU_THRESHOLD, NET_HEIGHT,
    NET_WIDTH, NMS_THRESHOLD, NUM_CLASSES,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(group(clap::ArgGroup::new("models").required(true).multiple(true)
    .args(["v3_onnx", "v4_onnx", "v5_onnx", "v8_onnx", "v9_onnx", "v11_onnx"])))]
#[command(group(clap::ArgGroup::new("input").required(true).multiple(true)
    .args(["image", "val_images"])))]
struct Args {
    /// Path to test image (for speed benchmark)
    #[arg(short, long)]
    image: Option<PathBuf>,

    /// Path to validation images directory (for mAP calculation)
    #[arg(long, requires = "val_labels")]
    val_images: Option<PathBuf>,

    /// Path to validation labels directory (for mAP calculation)
    #[arg(long, requires = "val_images")]
    val_labels: Option<PathBuf>,

    /// Number of inference iterations for speed benchmarking
    #[arg(short = 'n', long, default_value_t = 100, value_parser = clap::value_parser!(u32).range(1..))]
    iterations: u32,

    /// Use ONNX Runtime CUDA (build with --features ort-cuda)
    #[arg(long)]
    cuda: bool,

    /// YOLOv3-tiny ONNX from darknet2onnx --format yolov8
    #[arg(long)]
    v3_onnx: Option<PathBuf>,

    /// YOLOv4-tiny ONNX from darknet2onnx --format yolov8
    #[arg(long)]
    v4_onnx: Option<PathBuf>,

    /// Path to YOLOv5nu ONNX model (.onnx), with output [1, 8, N]
    #[arg(long)]
    v5_onnx: Option<PathBuf>,

    /// Path to YOLOv8n ONNX model (.onnx)
    #[arg(long)]
    v8_onnx: Option<PathBuf>,

    /// Path to YOLOv9t ONNX model (.onnx)
    #[arg(long)]
    v9_onnx: Option<PathBuf>,

    /// Path to YOLOv11n ONNX model (.onnx)
    #[arg(long)]
    v11_onnx: Option<PathBuf>,

    /// Warmup iterations before benchmarking
    #[arg(long, default_value_t = 10)]
    warmup: u32,

    /// Maximum images to process for mAP (0 = all)
    #[arg(long, default_value_t = 0)]
    max_images: usize,

    /// Enable debug output for detection comparison
    #[arg(long)]
    debug: bool,

    /// Print detailed metrics (confusion matrix, F1 scores)
    #[arg(long)]
    detailed: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    check_device(args.cuda)?;
    let backend_name = if args.cuda {
        "ONNX Runtime / CUDA"
    } else {
        "ONNX Runtime / CPU"
    };
    print_header();
    print_config(&args, backend_name);
    println!("  Runtime: {}", ort::info());
    let speed_image = load_speed_image(&args)?;
    let mut results = Vec::new();
    for (name, path) in [
        ("YOLOv3-tiny", &args.v3_onnx),
        ("YOLOv4-tiny", &args.v4_onnx),
        ("YOLOv5nu", &args.v5_onnx),
        ("YOLOv8n", &args.v8_onnx),
        ("YOLOv9t", &args.v9_onnx),
        ("YOLO11n", &args.v11_onnx),
    ] {
        if let Some(path) = path {
            results.push(benchmark_model(name, path, &args, &speed_image)?);
        }
    }
    print_summary(&results, backend_name, args.detailed);
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

fn load_speed_image(args: &Args) -> Result<Option<ImageBuffer>, Box<dyn std::error::Error>> {
    if let Some(ref img_path) = args.image {
        let image = load_image(img_path)?;
        println!(
            "  Speed test image loaded: {}x{}",
            image.width(),
            image.height()
        );
        Ok(Some(image))
    } else {
        Ok(None)
    }
}

fn benchmark_model(
    model_name: &str,
    onnx: &Path,
    args: &Args,
    speed_image: &Option<ImageBuffer>,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    println!("\n[{}]", model_name);
    println!("  Loading model: {}", onnx.display());
    let mut model = load_model(onnx, args.cuda)?;

    let mut result = BenchmarkResult::new(model_name, args.iterations);

    // Speed benchmark
    if let Some(ref image) = speed_image {
        let (total, min, max) = benchmark_speed(model_name, args.iterations, args.warmup, || {
            model.detect(image, CONF_THRESHOLD, NMS_THRESHOLD)?;
            Ok(())
        })?;
        result.total_time = total;
        result.min_time = min;
        result.max_time = max;
        result.mean_time = total / args.iterations;
        result.fps = args.iterations as f64 / total.as_secs_f64();
    }

    // mAP evaluation
    if let (Some(ref val_img), Some(ref val_lbl)) = (&args.val_images, &args.val_labels) {
        let map_result = if args.debug {
            run_map_evaluation_debug(model.as_mut(), val_img, val_lbl, args.max_images)?
        } else {
            run_map_evaluation(model.as_mut(), val_img, val_lbl, args.max_images)?
        };

        // Store detailed metrics for summary output
        result.confusion_matrix = Some(map_result.confusion_matrix);
        result.class_metrics = Some(PerClassMetrics {
            tp: map_result.per_class_metrics.iter().map(|m| m.tp).collect(),
            fp: map_result.per_class_metrics.iter().map(|m| m.fp).collect(),
            fn_: map_result.per_class_metrics.iter().map(|m| m.fn_).collect(),
        });

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
                result.fps =
                    map_result.num_images as f64 / map_result.total_inference_time.as_secs_f64();
            }
        }
    }

    Ok(result)
}

fn print_summary(results: &[BenchmarkResult], backend_name: &str, detailed: bool) {
    println!("\n");
    println!("+----------------------------------------------------------+");
    println!("|                    BENCHMARK SUMMARY                     |");
    println!("+----------------------------------------------------------+");

    for result in results {
        result.print();
    }

    // Comparison table
    if results.len() > 1 {
        println!(
            "\n\nComparison ({}x{}, {}):",
            NET_WIDTH, NET_HEIGHT, backend_name
        );
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

    // Detailed metrics (confusion matrix and F1 scores) for each model
    if detailed {
        for result in results {
            if let (Some(ref matrix), Some(ref metrics)) =
                (&result.confusion_matrix, &result.class_metrics)
            {
                println!("\n\n{}", "=".repeat(80));
                println!("DETAILED METRICS: {}", result.model_name);
                println!("{}", "=".repeat(80));

                // Print confusion matrix
                println!("\nConfusion Matrix:");
                let classes_with_bg: Vec<&str> = CLASSES
                    .iter()
                    .copied()
                    .chain(std::iter::once("BG"))
                    .collect();
                print!("{:>12}", "Actual\\Pred");
                for class in &classes_with_bg {
                    print!("{:>10}", class);
                }
                println!();
                println!("{}", "-".repeat(12 + 10 * classes_with_bg.len()));
                for (i, row) in matrix.iter().enumerate() {
                    print!("{:>12}", classes_with_bg[i]);
                    for &count in row {
                        if count > 0 {
                            print!("{:>10}", count);
                        } else {
                            print!("{:>10}", ".");
                        }
                    }
                    println!();
                }

                // Print F1 table
                println!("\nPrecision / Recall / F1:");
                println!(
                    "{:>12} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}",
                    "Class", "TP", "FP", "FN", "Precision", "Recall", "F1"
                );
                println!("{}", "-".repeat(78));

                let mut total_tp = 0usize;
                let mut total_fp = 0usize;
                let mut total_fn = 0usize;
                let mut f1_sum = 0.0f64;

                for i in 0..NUM_CLASSES {
                    let tp = metrics.tp[i];
                    let fp = metrics.fp[i];
                    let fn_ = metrics.fn_[i];
                    total_tp += tp;
                    total_fp += fp;
                    total_fn += fn_;

                    let precision = if tp + fp > 0 {
                        tp as f64 / (tp + fp) as f64
                    } else {
                        0.0
                    };
                    let recall = if tp + fn_ > 0 {
                        tp as f64 / (tp + fn_) as f64
                    } else {
                        0.0
                    };
                    let f1 = if precision + recall > 0.0 {
                        2.0 * precision * recall / (precision + recall)
                    } else {
                        0.0
                    };
                    f1_sum += f1;

                    println!(
                        "{:>12} {:>8} {:>8} {:>8} {:>10.2}% {:>10.2}% {:>10.2}%",
                        CLASSES[i],
                        tp,
                        fp,
                        fn_,
                        precision * 100.0,
                        recall * 100.0,
                        f1 * 100.0
                    );
                }

                let micro_precision = if total_tp + total_fp > 0 {
                    total_tp as f64 / (total_tp + total_fp) as f64
                } else {
                    0.0
                };
                let micro_recall = if total_tp + total_fn > 0 {
                    total_tp as f64 / (total_tp + total_fn) as f64
                } else {
                    0.0
                };
                let micro_f1 = if micro_precision + micro_recall > 0.0 {
                    2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
                } else {
                    0.0
                };
                let macro_f1 = f1_sum / NUM_CLASSES as f64;

                println!("{}", "-".repeat(78));
                println!(
                    "{:>12} {:>8} {:>8} {:>8} {:>10.2}% {:>10.2}% {:>10.2}% (micro)",
                    "Total",
                    total_tp,
                    total_fp,
                    total_fn,
                    micro_precision * 100.0,
                    micro_recall * 100.0,
                    micro_f1 * 100.0
                );
                println!("{:>64} {:>10.2}% (macro)", "", macro_f1 * 100.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_requires_a_model_and_complete_input_arguments() {
        assert!(Args::try_parse_from(["benchmark"]).is_err());
        assert!(Args::try_parse_from(["benchmark", "--image", "test.jpg"]).is_err());
        assert!(Args::try_parse_from([
            "benchmark",
            "--v3-onnx",
            "v3.onnx",
            "--val-images",
            "images"
        ])
        .is_err());
        assert!(Args::try_parse_from([
            "benchmark",
            "--v3-onnx",
            "v3.onnx",
            "--image",
            "test.jpg",
            "--iterations",
            "0"
        ])
        .is_err());
        assert!(Args::try_parse_from([
            "benchmark",
            "--v3-onnx",
            "v3.onnx",
            "--v5-onnx",
            "v5nu.onnx",
            "--image",
            "test.jpg",
            "--iterations",
            "1",
            "--warmup",
            "0"
        ])
        .is_ok());
    }

    #[cfg(not(feature = "ort-cuda"))]
    #[test]
    fn rejects_cuda_when_not_compiled_in() {
        assert!(check_device(true).is_err());
        assert!(check_device(false).is_ok());
    }
}
