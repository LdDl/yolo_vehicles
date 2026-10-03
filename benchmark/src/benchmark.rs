use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::metrics::{calculate_map, load_labels, ClassMetrics};
use crate::models::{convert_detections, load_image, YoloModel};
use crate::types::{Detection, GroundTruth, CONF_THRESHOLD, IOU_THRESHOLD, NMS_THRESHOLD};

/// Run speed benchmark on a single image
pub fn benchmark_speed<F>(
    _model_name: &str,
    iterations: u32,
    warmup: u32,
    mut inference_fn: F,
) -> Result<(Duration, Duration, Duration), Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
{
    if iterations == 0 {
        return Err("Speed benchmark needs at least one iteration".into());
    }
    // Warmup
    println!("  Warming up ({} iterations)...", warmup);
    for _ in 0..warmup {
        inference_fn()?;
    }

    // Benchmark
    println!("  Benchmarking ({} iterations)...", iterations);
    let mut times = Vec::with_capacity(iterations as usize);

    for i in 0..iterations {
        let start = Instant::now();
        inference_fn()?;
        let elapsed = start.elapsed();
        times.push(elapsed);

        if (i + 1) % 20 == 0 {
            println!("    Progress: {}/{}", i + 1, iterations);
        }
    }

    let total_time: Duration = times.iter().sum();
    let min_time = *times.iter().min().unwrap();
    let max_time = *times.iter().max().unwrap();

    Ok((total_time, min_time, max_time))
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
pub fn run_map_evaluation(
    model: &mut YoloModel,
    val_images_dir: &Path,
    val_labels_dir: &Path,
    max_images: usize,
    num_classes: usize,
) -> Result<MapResult, Box<dyn std::error::Error>> {
    run_map_evaluation_impl(
        model,
        val_images_dir,
        val_labels_dir,
        max_images,
        num_classes,
        false,
    )
}

/// Run mAP evaluation with debug output
pub fn run_map_evaluation_debug(
    model: &mut YoloModel,
    val_images_dir: &Path,
    val_labels_dir: &Path,
    max_images: usize,
    num_classes: usize,
) -> Result<MapResult, Box<dyn std::error::Error>> {
    run_map_evaluation_impl(
        model,
        val_images_dir,
        val_labels_dir,
        max_images,
        num_classes,
        true,
    )
}

fn run_map_evaluation_impl(
    model: &mut YoloModel,
    val_images_dir: &Path,
    val_labels_dir: &Path,
    max_images: usize,
    num_classes: usize,
    debug: bool,
) -> Result<MapResult, Box<dyn std::error::Error>> {
    let mut all_detections: HashMap<String, Vec<Detection>> = HashMap::new();
    let mut all_ground_truths: HashMap<String, Vec<GroundTruth>> = HashMap::new();
    let mut inference_times: Vec<Duration> = Vec::new();

    // Get list of images
    let mut image_files: Vec<_> = fs::read_dir(val_images_dir)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| {
                    matches!(
                        ext.to_string_lossy().to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png"
                    )
                })
                .unwrap_or(false)
        })
        .collect();

    image_files.sort_by_key(|e| e.path());

    let total_images = if max_images > 0 && max_images < image_files.len() {
        max_images
    } else {
        image_files.len()
    };

    if total_images == 0 {
        return Err(format!("No JPG/PNG images in {}", val_images_dir.display()).into());
    }
    println!("  Processing {} images for mAP...", total_images);

    for (idx, entry) in image_files.iter().take(total_images).enumerate() {
        let image_path = entry.path();
        let stem = image_path.file_stem().unwrap().to_str().unwrap();
        let label_path = val_labels_dir.join(format!("{}.txt", stem));

        if !label_path.is_file() {
            return Err(format!(
                "Missing label: {} (background images need an empty TXT)",
                label_path.display()
            )
            .into());
        }
        let image = load_image(&image_path)?;
        let img_width = image.width();
        let img_height = image.height();

        // Run inference with timing
        let start = Instant::now();
        let (boxes, class_ids, confidences) =
            model.detect(&image, CONF_THRESHOLD, NMS_THRESHOLD)?;
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
        let detections =
            convert_detections(&boxes, &class_ids, &confidences, img_width, img_height);
        all_detections.insert(stem.to_string(), detections.clone());

        // Load ground truth
        let ground_truths = load_labels(&label_path);
        if ground_truths.iter().any(|gt| gt.class_id >= num_classes) {
            return Err(format!("{}: class ID outside selected task", label_path.display()).into());
        }
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
    let eval_results = calculate_map(
        &all_detections,
        &all_ground_truths,
        IOU_THRESHOLD,
        num_classes,
    );

    // Calculate timing stats
    let total_inference_time: Duration = inference_times.iter().sum();
    let num_inferences = inference_times.len();
    let mean_inference_time = if num_inferences > 0 {
        total_inference_time / num_inferences as u32
    } else {
        Duration::ZERO
    };
    let min_inference_time = inference_times
        .iter()
        .min()
        .copied()
        .unwrap_or(Duration::ZERO);
    let max_inference_time = inference_times
        .iter()
        .max()
        .copied()
        .unwrap_or(Duration::ZERO);

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

#[cfg(test)]
mod tests {
    use super::*;
    use od_opencv::backend_ort::OrtModelError;
    use od_opencv::{BBox, ImageBuffer, ObjectDetector};

    struct FixtureDetector;

    impl ObjectDetector for FixtureDetector {
        type Input = ImageBuffer;
        type Error = OrtModelError;

        fn detect(
            &mut self,
            input: &ImageBuffer,
            _conf: f32,
            _nms: f32,
        ) -> Result<(Vec<BBox>, Vec<usize>, Vec<f32>), OrtModelError> {
            if input.as_slice().unwrap()[0] > 0 {
                Ok((vec![BBox::new(1, 1, 2, 2)], vec![0], vec![0.9]))
            } else {
                Ok((vec![], vec![], vec![]))
            }
        }
    }

    #[test]
    fn evaluates_rgb_images_and_empty_background_labels() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        image::RgbImage::from_pixel(4, 4, image::Rgb([255, 0, 0]))
            .save(root.join("object.png"))
            .unwrap();
        image::RgbImage::new(4, 4)
            .save(root.join("background.png"))
            .unwrap();
        fs::write(root.join("object.txt"), "0 0.5 0.5 0.5 0.5\n").unwrap();
        fs::write(root.join("background.txt"), "").unwrap();
        let result = run_map_evaluation(&mut FixtureDetector, root, root, 0, 4).unwrap();
        assert_eq!(result.num_images, 2);
        assert!((result.map - 1.0).abs() < 1e-9);
        assert_eq!(result.per_class_metrics[0].tp, 1);
        assert_eq!(result.per_class_metrics[0].fp, 0);
        fs::remove_file(root.join("background.txt")).unwrap();
        assert!(run_map_evaluation(&mut FixtureDetector, root, root, 0, 4).is_err());
    }

    #[test]
    fn rejects_empty_image_directory_and_propagates_inference_errors() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        assert!(run_map_evaluation(&mut FixtureDetector, root, root, 0, 4).is_err());
        assert!(benchmark_speed("fixture", 1, 0, || Err("fixture failure".into())).is_err());
        assert!(benchmark_speed("fixture", 0, 0, || Ok(())).is_err());
    }
}
