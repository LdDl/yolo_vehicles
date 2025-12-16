use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use opencv::{imgcodecs::imread, prelude::*};

use crate::metrics::{calculate_map, load_labels, ClassMetrics};
use crate::models::{convert_detections, YoloModel};
use crate::types::{Detection, GroundTruth, CONF_THRESHOLD, IOU_THRESHOLD, NMS_THRESHOLD};

/// Run speed benchmark on a single image
pub fn benchmark_speed<F>(
    _model_name: &str,
    iterations: u32,
    warmup: u32,
    mut inference_fn: F,
) -> (Duration, Duration, Duration)
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
    let min_time = *times.iter().min().unwrap();
    let max_time = *times.iter().max().unwrap();

    (total_time, min_time, max_time)
}

/// mAP evaluation result with timing info
pub struct MapResult {
    pub map: f64,
    pub per_class_ap: Vec<f64>,
    pub per_class_metrics: Vec<ClassMetrics>,
    pub confusion_matrix: Vec<Vec<usize>>,
    pub num_images: usize,
    pub total_inference_time: Duration,
    pub mean_inference_time: Duration,
    pub min_inference_time: Duration,
    pub max_inference_time: Duration,
}

/// Run mAP evaluation on validation set
pub fn run_map_evaluation<M: YoloModel>(
    model: &mut M,
    val_images_dir: &Path,
    val_labels_dir: &Path,
    max_images: usize,
) -> Result<MapResult, Box<dyn std::error::Error>> {
    run_map_evaluation_impl(model, val_images_dir, val_labels_dir, max_images, false)
}

/// Run mAP evaluation with debug output
pub fn run_map_evaluation_debug<M: YoloModel>(
    model: &mut M,
    val_images_dir: &Path,
    val_labels_dir: &Path,
    max_images: usize,
) -> Result<MapResult, Box<dyn std::error::Error>> {
    run_map_evaluation_impl(model, val_images_dir, val_labels_dir, max_images, true)
}

fn run_map_evaluation_impl<M: YoloModel>(
    model: &mut M,
    val_images_dir: &Path,
    val_labels_dir: &Path,
    max_images: usize,
    debug: bool,
) -> Result<MapResult, Box<dyn std::error::Error>> {
    let mut all_detections: HashMap<String, Vec<Detection>> = HashMap::new();
    let mut all_ground_truths: HashMap<String, Vec<GroundTruth>> = HashMap::new();
    let mut inference_times: Vec<Duration> = Vec::new();

    // Get list of images
    let mut image_files: Vec<_> = fs::read_dir(val_images_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "jpg" || ext == "png")
                .unwrap_or(false)
        })
        .collect();

    image_files.sort_by_key(|e| e.path());

    let total_images = if max_images > 0 && max_images < image_files.len() {
        max_images
    } else {
        image_files.len()
    };

    println!("  Processing {} images for mAP...", total_images);

    for (idx, entry) in image_files.iter().take(total_images).enumerate() {
        let image_path = entry.path();
        let stem = image_path.file_stem().unwrap().to_str().unwrap();
        let label_path = val_labels_dir.join(format!("{}.txt", stem));

        // Load image
        let image = imread(image_path.to_str().unwrap(), 1)?;
        if image.empty() {
            continue;
        }

        let img_width = image.cols();
        let img_height = image.rows();

        // Run inference with timing
        let start = Instant::now();
        let (boxes, class_ids, confidences) =
            model.forward(&image, CONF_THRESHOLD, NMS_THRESHOLD)?;
        inference_times.push(start.elapsed());

        // Debug output for first 3 images
        if debug && idx < 3 {
            println!("\n  DEBUG [{}]: image {}x{}", stem, img_width, img_height);
            println!("    Raw detections: {} boxes", boxes.len());
            for (i, rect) in boxes.iter().enumerate().take(5) {
                println!(
                    "      [{}] class={} conf={:.3} box=({}, {}, {}x{})",
                    i, class_ids[i], confidences[i], rect.x, rect.y, rect.width, rect.height
                );
            }
            if boxes.len() > 5 {
                println!("      ... and {} more", boxes.len() - 5);
            }
        }

        // Convert detections
        let detections = convert_detections(&boxes, &class_ids, &confidences, img_width, img_height);
        all_detections.insert(stem.to_string(), detections.clone());

        // Load ground truth
        let ground_truths = load_labels(&label_path);
        all_ground_truths.insert(stem.to_string(), ground_truths.clone());

        // Debug: compare detections vs ground truth
        if debug && idx < 3 {
            println!("    Normalized detections:");
            for (i, det) in detections.iter().enumerate().take(5) {
                println!(
                    "      [{}] class={} conf={:.3} center=({:.3}, {:.3}) size=({:.3}, {:.3})",
                    i, det.class_id, det.confidence, det.x, det.y, det.width, det.height
                );
            }
            println!("    Ground truth: {} objects", ground_truths.len());
            for (i, gt) in ground_truths.iter().enumerate().take(5) {
                println!(
                    "      [{}] class={} center=({:.3}, {:.3}) size=({:.3}, {:.3})",
                    i, gt.class_id, gt.x, gt.y, gt.width, gt.height
                );
            }
        }

        if (idx + 1) % 500 == 0 || idx + 1 == total_images {
            println!("    Progress: {}/{}", idx + 1, total_images);
        }
    }

    // Calculate mAP and metrics
    let eval_results = calculate_map(&all_detections, &all_ground_truths, IOU_THRESHOLD);

    // Calculate timing stats
    let total_inference_time: Duration = inference_times.iter().sum();
    let num_inferences = inference_times.len();
    let mean_inference_time = if num_inferences > 0 {
        total_inference_time / num_inferences as u32
    } else {
        Duration::ZERO
    };
    let min_inference_time = inference_times.iter().min().copied().unwrap_or(Duration::ZERO);
    let max_inference_time = inference_times.iter().max().copied().unwrap_or(Duration::ZERO);

    Ok(MapResult {
        map: eval_results.map,
        per_class_ap: eval_results.per_class_ap,
        per_class_metrics: eval_results.per_class_metrics,
        confusion_matrix: eval_results.confusion_matrix,
        num_images: total_images,
        total_inference_time,
        mean_inference_time,
        min_inference_time,
        max_inference_time,
    })
}
