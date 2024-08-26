""""Implementation of Hard Boundary Box Training (HBBT)
"""
from typing import List
from ultralytics.utils.tal import bbox_iou
import torch

def get_hard_boundary_boxes(gt_boxes: List, pred_boxes: List, iou_thr: float=0.5):
    """Get the list of wrongly predicted boxes (hard boundary boxes) by YOLOv8
    Args:
        gt_boxes (List): List of ground-truth boxes in YOLO format [[x_c, y_c, w, h], ...]
        pred_boxes (List): List of predicted boxes in YOLO format [[x_c, y_c, w, h], ...]
        iou_thr (float): IoU threshold to consider a prediction as correct
    Returns:
        hard_pred_boxes (List): List of hard boundary boxes in YOLO format
    """
    hard_pred_boxes = []
    matched_gt_indices = set()

    if len(gt_boxes) == 0:
        return pred_boxes
    
    for pred_box in pred_boxes:
        is_correct = False
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt_indices:
                continue  # skip if this ground-truth box is already matched

            iou = bbox_iou(torch.Tensor(pred_box), torch.Tensor(gt_box))
            if iou >= iou_thr:
                is_correct = True
                matched_gt_indices.add(gt_idx)
                break  # break since this predicted box is matched

        if not is_correct:
            hard_pred_boxes.append(pred_box)

    return hard_pred_boxes


if __name__ == "__main__":
    # suppose these are the predicted boxes and ground-truth boxes in YOLO format
    gt_boxes = [
        [0.2038, 0.2038, 0.2717, 0.2717], 
        [0.7133, 0.8391, 0.2038, 0.3193],
    ]

    pred_boxes = [
        [0.2072, 0.2014, 0.2921, 0.2629],
        [0.7531, 0.5353, 0.1474, 0.4429],
        [0.7075, 0.8346, 0.2072, 0.3201],
        [0.6793, 0.8492, 0.6793, 0.3397],
        [0.1325, 0.2344, 0.1427, 0.3465],
    ]

    hard_pred_boxes = get_hard_boundary_boxes(gt_boxes=gt_boxes, pred_boxes=pred_boxes, iou_thr=0.5)
    print(hard_pred_boxes)

    # save hard boundary boxes in YOLO format
    hard_sample_class_id = 1 # suppose gastric class ID is 0

    with open("yolo_label_sample.txt", "w") as f:
        for bbox in hard_pred_boxes:
            row_data = [hard_sample_class_id] + bbox
            row_str = " ".join([str(x) for x in row_data])
            
            f.write(f'{row_str}\n')