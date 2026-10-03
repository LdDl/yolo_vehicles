use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use crate::types::{Detection, GroundTruth};

/// Per-class metrics (TP, FP, FN counts)
#[derive(Debug, Clone)]
pub struct ClassMetrics {
    pub tp: usize,
    pub fp: usize,
    // fn is reserved keyword
    pub fn_: usize,
}

/// Extended evaluation results
#[derive(Debug)]
pub struct EvalResults {
    pub map: f64,
    pub per_class_ap: Vec<f64>,
    pub per_class_metrics: Vec<ClassMetrics>,
    // [actual][predicted]
    pub confusion_matrix: Vec<Vec<usize>>,
}

/// Calculate IoU between a detection and ground truth box (center format, normalized)
pub fn calculate_iou(det: &Detection, gt: &GroundTruth) -> f32 {
    // Convert center format to corner format
    let det_x1 = det.x - det.width / 2.0;
    let det_y1 = det.y - det.height / 2.0;
    let det_x2 = det.x + det.width / 2.0;
    let det_y2 = det.y + det.height / 2.0;

    let gt_x1 = gt.x - gt.width / 2.0;
    let gt_y1 = gt.y - gt.height / 2.0;
    let gt_x2 = gt.x + gt.width / 2.0;
    let gt_y2 = gt.y + gt.height / 2.0;

    // Intersection
    let inter_x1 = det_x1.max(gt_x1);
    let inter_y1 = det_y1.max(gt_y1);
    let inter_x2 = det_x2.min(gt_x2);
    let inter_y2 = det_y2.min(gt_y2);

    let inter_width = (inter_x2 - inter_x1).max(0.0);
    let inter_height = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_width * inter_height;

    // Union
    let det_area = det.width * det.height;
    let gt_area = gt.width * gt.height;
    let union_area = det_area + gt_area - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}

/// Calculate Average Precision for a single class using 11-point interpolation
/// tuple `detections`: Vector of (confidence, is_true_positive)
pub fn calculate_ap(detections: &mut Vec<(f32, bool)>, num_ground_truths: usize) -> f64 {
    if num_ground_truths == 0 {
        return 0.0;
    }

    // Sort by confidence (descending)
    detections.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut tp_cumsum = 0;
    let mut fp_cumsum = 0;
    let mut precisions = Vec::new();
    let mut recalls = Vec::new();

    for (_, is_tp) in detections.iter() {
        if *is_tp {
            tp_cumsum += 1;
        } else {
            fp_cumsum += 1;
        }

        let precision = tp_cumsum as f64 / (tp_cumsum + fp_cumsum) as f64;
        let recall = tp_cumsum as f64 / num_ground_truths as f64;

        precisions.push(precision);
        recalls.push(recall);
    }

    // Calculate AP using 11-point interpolation (Pascal VOC style)
    let mut ap = 0.0;
    for t in 0..=10 {
        let threshold = t as f64 / 10.0;
        let mut max_precision = 0.0;

        for (i, &recall) in recalls.iter().enumerate() {
            if recall >= threshold && precisions[i] > max_precision {
                max_precision = precisions[i];
            }
        }
        ap += max_precision;
    }
    ap /= 11.0;

    ap
}

/// Calculate mAP, confusion matrix, and F1 scores across all images
pub fn calculate_map(
    all_detections: &HashMap<String, Vec<Detection>>,
    all_ground_truths: &HashMap<String, Vec<GroundTruth>>,
    iou_threshold: f32,
    num_classes: usize,
) -> EvalResults {
    let mut per_class_detections: Vec<Vec<(f32, bool)>> = vec![Vec::new(); num_classes];
    let mut per_class_num_gt: Vec<usize> = vec![0; num_classes];

    // Confusion matrix: [actual_class][predicted_class]
    // +1 for "background" (false positives with no GT match)
    let mut confusion_matrix: Vec<Vec<usize>> = vec![vec![0; num_classes + 1]; num_classes + 1];

    // TP/FP/FN counters per class
    let mut per_class_tp: Vec<usize> = vec![0; num_classes];
    let mut per_class_fp: Vec<usize> = vec![0; num_classes];

    // Count ground truths per class
    for gts in all_ground_truths.values() {
        for gt in gts {
            if gt.class_id < num_classes {
                per_class_num_gt[gt.class_id] += 1;
            }
        }
    }

    // Keep equal-confidence detections ordered consistently across runs.
    let image_names: BTreeSet<_> = all_detections
        .keys()
        .chain(all_ground_truths.keys())
        .collect();
    for image_name in image_names {
        let detections = all_detections
            .get(image_name)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let ground_truths = all_ground_truths
            .get(image_name)
            .cloned()
            .unwrap_or_default();
        let mut gt_matched: Vec<bool> = vec![false; ground_truths.len()];
        let mut ap_matched = vec![false; ground_truths.len()];

        // Sort detections by confidence (process high confidence first)
        let mut sorted_dets = detections.to_vec();
        sorted_dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        for det in &sorted_dets {
            if det.class_id >= num_classes {
                continue;
            }

            // AP matches only within the predicted class, independently of the confusion matrix.
            let mut ap_best_iou = 0.0;
            let mut ap_best_gt = None;
            for (gt_idx, gt) in ground_truths.iter().enumerate() {
                if ap_matched[gt_idx] || gt.class_id != det.class_id {
                    continue;
                }
                let iou = calculate_iou(det, gt);
                if iou >= iou_threshold && iou > ap_best_iou {
                    ap_best_iou = iou;
                    ap_best_gt = Some(gt_idx);
                }
            }
            let is_tp = ap_best_gt.is_some();
            if let Some(gt_idx) = ap_best_gt {
                ap_matched[gt_idx] = true;
                per_class_tp[det.class_id] += 1;
            } else {
                per_class_fp[det.class_id] += 1;
            }
            per_class_detections[det.class_id].push((det.confidence, is_tp));

            // Find best matching ground truth (any class, for confusion matrix)
            let mut best_iou = 0.0;
            let mut best_gt_idx = None;

            for (gt_idx, gt) in ground_truths.iter().enumerate() {
                if gt_matched[gt_idx] || gt.class_id >= num_classes {
                    continue;
                }
                let iou = calculate_iou(det, gt);
                if iou > best_iou && iou >= iou_threshold {
                    best_iou = iou;
                    best_gt_idx = Some(gt_idx);
                }
            }

            // Record spatial matches, including class mistakes, for the confusion matrix.
            if let Some(gt_idx) = best_gt_idx {
                let gt_class = ground_truths[gt_idx].class_id;
                gt_matched[gt_idx] = true;
                confusion_matrix[gt_class][det.class_id] += 1;
            } else {
                // background -> predicted
                confusion_matrix[num_classes][det.class_id] += 1;
            }
        }

        // Record objects with no spatial match in the confusion matrix.
        for (gt_idx, gt) in ground_truths.iter().enumerate() {
            if !gt_matched[gt_idx] && gt.class_id < num_classes {
                // actual -> background (missed)
                confusion_matrix[gt.class_id][num_classes] += 1;
            }
        }
    }

    // Calculate AP for each class
    let mut per_class_ap = Vec::new();
    let mut total_ap = 0.0;
    let mut valid_classes = 0;

    for class_id in 0..num_classes {
        let ap = calculate_ap(
            &mut per_class_detections[class_id],
            per_class_num_gt[class_id],
        );
        per_class_ap.push(ap);

        if per_class_num_gt[class_id] > 0 {
            total_ap += ap;
            valid_classes += 1;
        }
    }

    let map = if valid_classes > 0 {
        total_ap / valid_classes as f64
    } else {
        0.0
    };

    // Collect per-class metrics (TP, FP, FN counts)
    let per_class_metrics: Vec<ClassMetrics> = (0..num_classes)
        .map(|class_id| ClassMetrics {
            tp: per_class_tp[class_id],
            fp: per_class_fp[class_id],
            // Every ground-truth object without a correct-class match is a false negative.
            fn_: per_class_num_gt[class_id] - per_class_tp[class_id],
        })
        .collect();

    EvalResults {
        map,
        per_class_ap,
        per_class_metrics,
        confusion_matrix,
    }
}

/// Load ground truth labels from YOLO format file
pub fn load_labels(label_path: &Path) -> Vec<GroundTruth> {
    let mut labels = Vec::new();

    if let Ok(content) = fs::read_to_string(label_path) {
        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 5 {
                if let (Ok(class_id), Ok(x), Ok(y), Ok(w), Ok(h)) = (
                    parts[0].parse::<usize>(),
                    parts[1].parse::<f32>(),
                    parts[2].parse::<f32>(),
                    parts[3].parse::<f32>(),
                    parts[4].parse::<f32>(),
                ) {
                    labels.push(GroundTruth {
                        class_id,
                        x,
                        y,
                        width: w,
                        height: h,
                    });
                }
            }
        }
    }

    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ground_truth(class_id: usize) -> GroundTruth {
        GroundTruth {
            class_id,
            x: 0.5,
            y: 0.5,
            width: 0.2,
            height: 0.2,
        }
    }

    fn detection(class_id: usize, confidence: f32) -> Detection {
        Detection {
            class_id,
            confidence,
            x: 0.5,
            y: 0.5,
            width: 0.2,
            height: 0.2,
        }
    }

    #[test]
    fn plate_class_four_is_not_background() {
        let detections = HashMap::from([("plate.jpg".into(), vec![detection(4, 0.9)])]);
        let truth = HashMap::from([
            ("plate.jpg".into(), vec![ground_truth(4)]),
            ("missed.jpg".into(), vec![ground_truth(3)]),
        ]);
        let result = calculate_map(&detections, &truth, 0.5, 5);
        assert_eq!(result.per_class_ap.len(), 5);
        assert_eq!(result.confusion_matrix.len(), 6);
        assert_eq!(result.confusion_matrix[4][4], 1);
        assert_eq!(result.confusion_matrix[3][5], 1);
        assert!((result.per_class_ap[4] - 1.0).abs() < 1e-9);
        assert!((result.map - 0.5).abs() < 1e-9);
    }

    #[test]
    fn wrong_class_counts_as_false_negative_for_actual_class() {
        let detections = HashMap::from([("bus.jpg".into(), vec![detection(0, 0.9)])]);
        let truth = HashMap::from([("bus.jpg".into(), vec![ground_truth(2)])]);
        let result = calculate_map(&detections, &truth, 0.5, 4);
        assert_eq!(result.per_class_metrics[0].fp, 1);
        assert_eq!(result.per_class_metrics[2].tp, 0);
        assert_eq!(result.per_class_metrics[2].fn_, 1);
        assert_eq!(result.confusion_matrix[2][0], 1);
        assert_eq!(result.map, 0.0);
    }

    #[test]
    fn wrong_class_cannot_consume_ap_match_and_duplicates_stay_false_positive() {
        let detections = HashMap::from([(
            "bus.jpg".into(),
            vec![detection(0, 0.9), detection(2, 0.8), detection(2, 0.7)],
        )]);
        let truth = HashMap::from([("bus.jpg".into(), vec![ground_truth(2)])]);
        let result = calculate_map(&detections, &truth, 0.5, 4);
        assert_eq!(result.per_class_metrics[0].fp, 1);
        assert_eq!(result.per_class_metrics[2].tp, 1);
        assert_eq!(result.per_class_metrics[2].fp, 1);
        assert_eq!(result.per_class_metrics[2].fn_, 0);
        assert!((result.per_class_ap[2] - 1.0).abs() < 1e-9);
        assert_eq!(result.confusion_matrix[2][0], 1);
    }

    #[test]
    fn counts_objects_in_images_without_predictions_and_background_detections() {
        let detections = HashMap::from([("background.jpg".into(), vec![detection(0, 0.9)])]);
        let truth = HashMap::from([("missed.jpg".into(), vec![ground_truth(2)])]);
        let result = calculate_map(&detections, &truth, 0.5, 4);
        assert_eq!(result.per_class_metrics[0].fp, 1);
        assert_eq!(result.per_class_metrics[2].fn_, 1);
        assert_eq!(result.confusion_matrix[4][0], 1);
        assert_eq!(result.confusion_matrix[2][4], 1);
    }
}
