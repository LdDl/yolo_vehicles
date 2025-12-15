use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use opencv::{imgcodecs::imread, prelude::*};

use crate::metrics::{calculate_map, load_labels};
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

        // Convert detections
        let detections = convert_detections(&boxes, &class_ids, &confidences, img_width, img_height);
        all_detections.insert(stem.to_string(), detections);

        // Load ground truth
        let ground_truths = load_labels(&label_path);
        all_ground_truths.insert(stem.to_string(), ground_truths);

        if (idx + 1) % 500 == 0 || idx + 1 == total_images {
            println!("    Progress: {}/{}", idx + 1, total_images);
        }
    }

    // Calculate mAP
    let (map, per_class_ap) = calculate_map(&all_detections, &all_ground_truths, IOU_THRESHOLD);

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
        map,
        per_class_ap,
        num_images: total_images,
        total_inference_time,
        mean_inference_time,
        min_inference_time,
        max_inference_time,
    })
}
