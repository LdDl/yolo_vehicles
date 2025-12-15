use od_opencv::{model_classic::ModelYOLOClassic, model_ultralytics::ModelUltralyticsV8};
use opencv::{core::Rect, prelude::*};

use crate::types::Detection;

/// Trait for unified model interface
pub trait YoloModel {
    fn forward(
        &mut self,
        image: &Mat,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), opencv::Error>;
}

impl YoloModel for ModelYOLOClassic {
    fn forward(
        &mut self,
        image: &Mat,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), opencv::Error> {
        self.forward(image, conf_threshold, nms_threshold)
    }
}

impl YoloModel for ModelUltralyticsV8 {
    fn forward(
        &mut self,
        image: &Mat,
        conf_threshold: f32,
        nms_threshold: f32,
    ) -> Result<(Vec<Rect>, Vec<usize>, Vec<f32>), opencv::Error> {
        self.forward(image, conf_threshold, nms_threshold)
    }
}

/// Convert od_opencv detections to our Detection struct
pub fn convert_detections(
    boxes: &[Rect],
    class_ids: &[usize],
    confidences: &[f32],
    img_width: i32,
    img_height: i32,
) -> Vec<Detection> {
    let mut detections = Vec::new();

    for i in 0..boxes.len() {
        let rect = &boxes[i];
        let class_id = class_ids[i];
        let confidence = confidences[i];

        // Convert to normalized center format
        let x = (rect.x as f32 + rect.width as f32 / 2.0) / img_width as f32;
        let y = (rect.y as f32 + rect.height as f32 / 2.0) / img_height as f32;
        let width = rect.width as f32 / img_width as f32;
        let height = rect.height as f32 / img_height as f32;

        detections.push(Detection {
            class_id,
            confidence,
            x,
            y,
            width,
            height,
        });
    }

    detections
}
