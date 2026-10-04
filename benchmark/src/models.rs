use std::path::Path;

use od_opencv::backend_ort::{ModelUltralyticsOrt, OrtModelError};
use od_opencv::{BBox, ImageBuffer, ObjectDetector};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{TensorElementType, ValueType};

use crate::types::Detection;

pub type YoloModel = dyn ObjectDetector<Input = ImageBuffer, Error = OrtModelError>;

pub fn check_device(cuda: bool) -> Result<(), Box<dyn std::error::Error>> {
    if cuda && !cfg!(feature = "ort-cuda") {
        return Err("--cuda requires a build with --features ort-cuda".into());
    }
    Ok(())
}

pub fn load_model(
    path: &Path,
    cuda: bool,
    width: u32,
    height: u32,
    num_classes: usize,
) -> Result<Box<YoloModel>, Box<dyn std::error::Error>> {
    check_device(cuda)?;
    let mut builder =
        Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)?;
    #[cfg(feature = "ort-cuda")]
    if cuda {
        // A failed CUDA registration must not produce a misleading CPU benchmark.
        builder = builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])?;
    }
    let session = builder.commit_from_file(path)?;
    if session.inputs().len() != 1 || session.outputs().len() != 1 {
        return Err(format!(
            "{}: expected one input and one output without embedded NMS",
            path.display()
        )
        .into());
    }
    let input = session.inputs()[0].dtype();
    let output = session.outputs()[0].dtype();
    validate_interface(input, output, width, height, num_classes)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    println!("  ONNX input: {input:?}");
    println!("  ONNX output: {output:?}");
    let size = (width, height);
    Ok(Box::new(ModelUltralyticsOrt::from_session(
        session,
        size,
        vec![],
    )))
}

fn validate_interface(
    input: &ValueType,
    output: &ValueType,
    width: u32,
    height: u32,
    num_classes: usize,
) -> Result<(), String> {
    let ValueType::Tensor {
        ty: TensorElementType::Float32,
        shape,
        ..
    } = input
    else {
        return Err("expected float32 input".into());
    };
    if shape.as_ref() != [1, 3, height as i64, width as i64] {
        return Err(format!(
            "expected static input [1, 3, {height}, {width}], got {shape:?}"
        ));
    }
    let ValueType::Tensor {
        ty: TensorElementType::Float32,
        shape,
        ..
    } = output
    else {
        return Err("expected float32 output".into());
    };
    let dims: &[i64] = shape.as_ref();
    let channels = 4 + num_classes as i64;
    if dims.len() != 3 || dims[0] != 1 || dims[1] != channels || !(dims[2] > 0 || dims[2] == -1) {
        return Err(format!(
            "expected output [1, {channels}, N] for {num_classes} classes, got {shape:?}"
        ));
    }
    Ok(())
}

pub fn load_image(path: &Path) -> Result<ImageBuffer, Box<dyn std::error::Error>> {
    let image = image::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(ImageBuffer::from_dynamic_image(image))
}

/// Convert od_opencv detections to our Detection struct
pub fn convert_detections(
    boxes: &[BBox],
    class_ids: &[usize],
    confidences: &[f32],
    img_width: usize,
    img_height: usize,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ort::value::{Shape, SymbolicDimensions};

    fn tensor(dims: &[i64]) -> ValueType {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape: Shape::new(dims.iter().copied()),
            dimension_symbols: SymbolicDimensions::new(vec![String::new(); dims.len()]),
        }
    }

    #[test]
    fn accepts_five_classes_at_custom_size_but_rejects_vehicle_model() {
        let input = tensor(&[1, 3, 192, 320]);
        assert!(validate_interface(&input, &tensor(&[1, 9, 1260]), 320, 192, 5).is_ok());
        assert!(validate_interface(&input, &tensor(&[1, 8, 1260]), 320, 192, 5).is_err());
        assert!(validate_interface(&input, &tensor(&[1, 9, 1260]), 416, 256, 5).is_err());
    }

    #[test]
    fn ocr_layout_matches_alphabet_and_rejects_plate_output() {
        let names: Vec<_> = include_str!("../../ocr/classes.names").lines().collect();
        assert_eq!(crate::types::Task::Ocr.classes(), names.as_slice());
        let input = tensor(&[1, 3, 64, 224]);
        assert!(validate_interface(&input, &tensor(&[1, 27, 210]), 224, 64, names.len()).is_ok());
        assert!(validate_interface(&input, &tensor(&[1, 9, 210]), 224, 64, names.len()).is_err());
        assert!(validate_interface(&input, &tensor(&[1, 27, 210]), 192, 64, names.len()).is_err());
    }

    #[test]
    fn checks_input_dimensions_and_class_layout() {
        let input = tensor(&[1, 3, 256, 416]);
        assert!(validate_interface(&input, &tensor(&[1, 8, 1560]), 416, 256, 4).is_ok());
        assert!(validate_interface(&input, &tensor(&[1, 8, 2184]), 416, 256, 4).is_ok());
        for wrong_input in [
            tensor(&[1, 3, 416, 256]),
            tensor(&[1, 3, 960, 960]),
            tensor(&[-1, 3, -1, -1]),
        ] {
            assert!(validate_interface(&wrong_input, &tensor(&[1, 8, 1560]), 416, 256, 4).is_err());
        }
        for wrong_output in [
            tensor(&[1, 84, 2184]),
            tensor(&[1, 300, 6]),
            tensor(&[1, 6552, 9]),
        ] {
            assert!(validate_interface(&input, &wrong_output, 416, 256, 4).is_err());
        }
    }

    #[test]
    fn normalizes_boxes_in_original_image_coordinates() {
        let detections =
            convert_detections(&[BBox::new(480, 270, 960, 540)], &[3], &[0.9], 1920, 1080);
        let detection = &detections[0];
        assert_eq!(detection.class_id, 3);
        assert_eq!(
            (detection.x, detection.y, detection.width, detection.height),
            (0.5, 0.5, 0.5, 0.5)
        );
    }

    #[test]
    fn reads_rgb_without_swapping_red_and_blue() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("rgb.png");
        image::RgbImage::from_pixel(4, 2, image::Rgb([251, 17, 3]))
            .save(&path)
            .unwrap();
        let loaded = load_image(&path).unwrap();
        assert_eq!(loaded.shape(), (2, 4, 3));
        assert_eq!(&loaded.as_slice().unwrap()[..3], &[251, 17, 3]);
    }
}
