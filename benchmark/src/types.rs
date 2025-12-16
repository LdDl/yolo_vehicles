use std::time::Duration;

pub const CLASSES: [&str; 4] = ["car", "motorbike", "bus", "truck"];
pub const NUM_CLASSES: usize = 4;
pub const NET_WIDTH: i32 = 416;
pub const NET_HEIGHT: i32 = 256;
pub const CONF_THRESHOLD: f32 = 0.25;
pub const NMS_THRESHOLD: f32 = 0.45;
pub const IOU_THRESHOLD: f32 = 0.5;

/// Detection from model inference
#[derive(Debug, Clone)]
pub struct Detection {
    pub class_id: usize,
    pub confidence: f32,
    pub x: f32,      // center x (normalized 0-1)
    pub y: f32,      // center y (normalized 0-1)
    pub width: f32,  // width (normalized 0-1)
    pub height: f32, // height (normalized 0-1)
}

/// Ground truth from label file
#[derive(Debug, Clone)]
pub struct GroundTruth {
    pub class_id: usize,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Per-class metrics for detailed output
#[derive(Clone)]
pub struct PerClassMetrics {
    pub tp: Vec<usize>,
    pub fp: Vec<usize>,
    pub fn_: Vec<usize>,
}

/// Results from benchmarking a single model
pub struct BenchmarkResult {
    pub model_name: String,
    pub iterations: u32,
    pub total_time: Duration,
    pub mean_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub fps: f64,
    pub map50: Option<f64>,
    pub per_class_ap: Option<Vec<f64>>,
    pub confusion_matrix: Option<Vec<Vec<usize>>>,
    pub class_metrics: Option<PerClassMetrics>,
}

impl BenchmarkResult {
    pub fn new(model_name: &str, iterations: u32) -> Self {
        Self {
            model_name: model_name.to_string(),
            iterations,
            total_time: Duration::ZERO,
            mean_time: Duration::ZERO,
            min_time: Duration::ZERO,
            max_time: Duration::ZERO,
            fps: 0.0,
            map50: None,
            per_class_ap: None,
            confusion_matrix: None,
            class_metrics: None,
        }
    }

    pub fn print(&self) {
        println!("\n{}", "=".repeat(50));
        println!("Model: {}", self.model_name);
        println!("{}", "=".repeat(50));
        println!("Iterations: {}", self.iterations);
        println!("Total time: {:.2?}", self.total_time);
        println!("Mean time:  {:.2?}", self.mean_time);
        println!("Min time:   {:.2?}", self.min_time);
        println!("Max time:   {:.2?}", self.max_time);
        println!("FPS:        {:.2}", self.fps);

        if let Some(map) = self.map50 {
            println!("\nmAP@0.50:   {:.2}%", map * 100.0);
        }

        if let Some(ref aps) = self.per_class_ap {
            println!("\nPer-class AP@0.50:");
            for (i, ap) in aps.iter().enumerate() {
                println!("  {}: {:.2}%", CLASSES[i], ap * 100.0);
            }
        }
    }
}
