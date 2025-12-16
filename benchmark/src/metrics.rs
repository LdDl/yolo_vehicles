use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::types::{Detection, GroundTruth, NUM_CLASSES};

/// Per-class metrics (TP, FP, FN counts)
#[derive(Debug, Clone)]
pub struct ClassMetrics {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,  // fn is reserved keyword
}

/// Extended evaluation results
#[derive(Debug)]
pub struct EvalResults {
    pub map: f64,
    pub per_class_ap: Vec<f64>,
    pub per_class_metrics: Vec<ClassMetrics>,
    pub confusion_matrix: Vec<Vec<usize>>,  // [actual][predicted]
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
pub fn calculate_ap(
    detections: &mut Vec<(f32, bool)>,
    num_ground_truths: usize,
) -> f64 {
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
) -> EvalResults {
    let mut per_class_detections: Vec<Vec<(f32, bool)>> = vec![Vec::new(); NUM_CLASSES];
    let mut per_class_num_gt: Vec<usize> = vec![0; NUM_CLASSES];

    // Confusion matrix: [actual_class][predicted_class]
    // +1 for "background" (false positives with no GT match)
    let mut confusion_matrix: Vec<Vec<usize>> = vec![vec![0; NUM_CLASSES + 1]; NUM_CLASSES + 1];

    // TP/FP/FN counters per class
    let mut per_class_tp: Vec<usize> = vec![0; NUM_CLASSES];
    let mut per_class_fp: Vec<usize> = vec![0; NUM_CLASSES];
    let mut per_class_fn: Vec<usize> = vec![0; NUM_CLASSES];

    // Count ground truths per class
    for gts in all_ground_truths.values() {
        for gt in gts {
            if gt.class_id < NUM_CLASSES {
                per_class_num_gt[gt.class_id] += 1;
            }
        }
    }

    // Match detections to ground truths
    for (image_name, detections) in all_detections {
        let ground_truths = all_ground_truths.get(image_name).cloned().unwrap_or_default();
        let mut gt_matched: Vec<bool> = vec![false; ground_truths.len()];
        let mut gt_matched_by: Vec<Option<usize>> = vec![None; ground_truths.len()]; // which class matched it

        // Sort detections by confidence (process high confidence first)
        let mut sorted_dets = detections.clone();
        sorted_dets.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        for det in &sorted_dets {
            if det.class_id >= NUM_CLASSES {
                continue;
            }

            // Find best matching ground truth (any class, for confusion matrix)
            let mut best_iou = 0.0;
            let mut best_gt_idx = None;

            for (gt_idx, gt) in ground_truths.iter().enumerate() {
                if gt_matched[gt_idx] {
                    continue;
                }
                let iou = calculate_iou(det, gt);
                if iou > best_iou && iou >= iou_threshold {
                    best_iou = iou;
                    best_gt_idx = Some(gt_idx);
                }
            }

            // Determine if true positive or false positive
            if let Some(gt_idx) = best_gt_idx {
                let gt_class = ground_truths[gt_idx].class_id;
                gt_matched[gt_idx] = true;
                gt_matched_by[gt_idx] = Some(det.class_id);

                if gt_class == det.class_id {
                    // True positive: correct class
                    per_class_detections[det.class_id].push((det.confidence, true));
                    per_class_tp[det.class_id] += 1;
                    confusion_matrix[gt_class][det.class_id] += 1;
                } else {
                    // Class mismatch: detection matched GT but wrong class
                    per_class_detections[det.class_id].push((det.confidence, false));
                    per_class_fp[det.class_id] += 1;
                    confusion_matrix[gt_class][det.class_id] += 1; // actual -> predicted
                }
            } else {
                // False positive: no matching GT
                per_class_detections[det.class_id].push((det.confidence, false));
                per_class_fp[det.class_id] += 1;
                confusion_matrix[NUM_CLASSES][det.class_id] += 1; // background -> predicted
            }
        }

        // Count false negatives (unmatched ground truths)
        for (gt_idx, gt) in ground_truths.iter().enumerate() {
            if !gt_matched[gt_idx] && gt.class_id < NUM_CLASSES {
                per_class_fn[gt.class_id] += 1;
                confusion_matrix[gt.class_id][NUM_CLASSES] += 1; // actual -> background (missed)
            }
        }
    }

    // Calculate AP for each class
    let mut per_class_ap = Vec::new();
    let mut total_ap = 0.0;
    let mut valid_classes = 0;

    for class_id in 0..NUM_CLASSES {
        let ap = calculate_ap(&mut per_class_detections[class_id], per_class_num_gt[class_id]);
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
    let per_class_metrics: Vec<ClassMetrics> = (0..NUM_CLASSES)
        .map(|class_id| ClassMetrics {
            tp: per_class_tp[class_id],
            fp: per_class_fp[class_id],
            fn_: per_class_fn[class_id],
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
