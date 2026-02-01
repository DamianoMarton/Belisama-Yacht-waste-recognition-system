import torch
import torchvision

def bbox_iou_matrix(boxes1, boxes2) -> torch.Tensor: # vectorized by Copilot
    """
    Computes the IoU between M boxes and N boxes.
    
    Args:
        boxes1 (torch.Tensor): [M, 4] (Target)
        boxes2 (torch.Tensor): [N, 4] (Predictions)

    Returns:
        ious (torch.Tensor): [M, N]
    """
    b1 = boxes1.unsqueeze(1) # [M, 4] -> [M, 1, 4]
    b2 = boxes2.unsqueeze(0) # [N, 4] -> [1, N, 4]

    x1 = torch.max(b1[..., 0], b2[..., 0]) # [M, N]
    y1 = torch.max(b1[..., 1], b2[..., 1]) # [M, N]
    x2 = torch.min(b1[..., 2], b2[..., 2]) # [M, N]
    y2 = torch.min(b1[..., 3], b2[..., 3]) # [M, N]

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0) # [M, N]
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) # [M]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) # [N]
    
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection # [M, N]

    iou = intersection / (union + 1e-6) # [M, N]
    return iou

def batch_target_to_boxes_vect(target_batch) -> list[dict]: # vectorized by Copilot
    """
    Transforms a batch of YOLO-style target (ground truth) tensors into bounding boxes and class labels.
    
    Args:
        target_batch (torch.Tensor): Tensor of shape [batch_size, 5 + num_classes, grid_size, grid_size]
    
    Returns:
        List of length batch_size, where each element is:
        {'boxes': torch.Tensor[N, 4], 'labels': torch.Tensor[N]} N is number of ground truth boxes in the image
    """

    batch_size, _, grid_size, _ = target_batch.shape
    device = target_batch.device
    
    target_batch = target_batch.permute(0, 2, 3, 1) # [B, 5 + num_classes, grid_size, grid_size] -> [B, grid_size, grid_size, 5 + num_classes]
    
    # setup coordinate grid
    rows, cols = torch.meshgrid(torch.arange(grid_size, device=device), 
                                torch.arange(grid_size, device=device), 
                                indexing='ij') # [grid_size, grid_size], [grid_size, grid_size]

    conf = target_batch[..., 0] # [B, grid_size, grid_size]
    x_rel = target_batch[..., 1]
    y_rel = target_batch[..., 2]
    w = target_batch[..., 3]
    h = target_batch[..., 4]
    class_probs = target_batch[..., 5:] # [B, grid_size, grid_size, num_classes]
    class_labels = torch.argmax(class_probs, dim=-1) # [B, grid_size, grid_size] each cell is class index

    # Broadcast cols with x_rel and rows with y_rel to get global coordinates
    x_c = (cols + x_rel) / grid_size # [B, grid_size, grid_size]
    y_c = (rows + y_rel) / grid_size # [B, grid_size, grid_size]
    
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    
    all_boxes = torch.stack([x1, y1, x2, y2], dim=-1) # [B, grid_size, grid_size, 4]

    batch_output = []

    for b in range(batch_size):
        mask = (conf[b] == 1) # [grid_size, grid_size]
        
        img_boxes = all_boxes[b][mask] # [N, 4] where N is number of boxes in image b
        img_labels = class_labels[b][mask] # [N]

        batch_output.append({
            "boxes": img_boxes,
            "labels": img_labels
        })

    return batch_output

def batch_post_process_vect(predictions, 
                            conf_thres=0.6, 
                            iou_thres=0.45) -> list[dict]: # vectorized by Copilot
    """
    Every raw prediction tensor is post-processed to yield final bounding boxes, scores, and class labels
    Also applies Non-Maximum Suppression to filter overlapping boxes 

    Args:
        predictions: Tensor of shape [batch_size, 5 + num_classes, S, S] not normalized
        conf_thres: Confidence threshold to filter boxes before NMS in [0,1]
        iou_thres: IoU threshold for Non-Maximum Suppression in [0,1]

    Returns:
        List of length batch_size, where each element is:
        {'boxes': torch.Tensor[N, 4], 'labels': torch.Tensor[N], 'confidences': torch.Tensor[N]} N is number of final boxes after NMS
    """ 
    batch_size, _, grid_size, _ = predictions.shape
    device = predictions.device
    
    preds = predictions.permute(0, 2, 3, 1) # [batch_size, 5 + num_classes, grid_size, grid_size] -> [B, grid_size, grid_size, 5 + num_classes]
    
    conf = torch.sigmoid(preds[..., 0]) # [B, grid_size, grid_size]
    x_rel = torch.sigmoid(preds[..., 1])
    y_rel = torch.sigmoid(preds[..., 2])         
    w_rel = torch.sigmoid(preds[..., 3])         
    h_rel = torch.sigmoid(preds[..., 4])
    class_probs = torch.softmax(preds[..., 5:], dim=-1) # [B, grid_size, grid_size, num_classes]
    max_scores, labels = torch.max(class_probs, dim=-1) # [B, grid_size, grid_size], [B, grid_size, grid_size]
    
    combined_conf = conf * max_scores # [B, grid_size, grid_size]
    
    # coordinates grid
    rows, cols = torch.meshgrid(torch.arange(grid_size, device=device), 
                                torch.arange(grid_size, device=device), 
                                indexing='ij')

    # Global box coordinates
    x_c = (cols + x_rel) / grid_size
    y_c = (rows + y_rel) / grid_size
    
    x1 = x_c - w_rel / 2
    y1 = y_c - h_rel / 2
    x2 = x_c + w_rel / 2
    y2 = y_c + h_rel / 2
    
    all_boxes = torch.stack([x1, y1, x2, y2], dim=-1) # [B, grid_size, grid_size, 4]

    final_results = []

    for b in range(batch_size):
        mask = combined_conf[b] > conf_thres
        
        img_boxes = all_boxes[b][mask] # [N, 4] where N is number of boxes in image b
        img_scores = combined_conf[b][mask] # [N]
        img_labels = labels[b][mask] # [N]

        if img_boxes.shape[0] == 0:
            final_results.append({
                'boxes': torch.empty((0, 4), device=device),
                'confidences': torch.empty(0, device=device),
                'labels': torch.empty(0, device=device)
            })
            continue

        # Non-Maximum Suppression by torchvision
        keep_indices = torchvision.ops.nms(img_boxes, img_scores, iou_thres)

        final_results.append({
            'boxes': img_boxes[keep_indices],
            'confidences': img_scores[keep_indices],
            'labels': img_labels[keep_indices]
        })

    return final_results

def compute_batch_performance_vect(predictions, targets, 
                              conf_thres=0.6, 
                              iou_match_thres=0.5, 
                              iou_NMS_thres=0.45) -> tuple[float, float, float]: # vectorized by Copilot
    """
    Computes precision, recall, and F1-score for a batch of predictions and targets.

    Args:
        predictions: Tensor of shape [batch_size, 5 + num_classes, grid_size, grid_size]
        targets: Tensor of shape [batch_size, 5 + num_classes, grid_size, grid_size]
        conf_thres: Confidence threshold to filter boxes before matching in [0,1]
        iou_match_thres: IoU threshold to consider a prediction as True Positive in [0,1]
        iou_NMS_thres: IoU threshold for Non-Maximum Suppression in [0,1]
    
    Returns:
        precision: float
        recall: float
        f1: float
    """
    pred_results = batch_post_process_vect(predictions, conf_thres, iou_NMS_thres) # list of dict for every image
    target_results = batch_target_to_boxes_vect(targets) # list of dict for every image
    
    total_tr_pos, total_fa_pos, total_fa_neg = 0, 0, 0

    for i in range(len(pred_results)):
        p = pred_results[i]
        t = target_results[i]
        p_boxes = p['boxes']
        p_labels = p['labels']
        t_boxes = t['boxes']
        t_labels = t['labels']

        num_preds  = p_boxes.shape[0]
        num_targets = t_boxes.shape[0]

        # trivial cases
        if num_targets == 0:
            total_fa_pos += num_preds
            continue
        if num_preds == 0:
            total_fa_neg += num_targets
            continue
        
        ious = bbox_iou_matrix(t_boxes, p_boxes) 
        label_match = (t_labels.unsqueeze(1) == p_labels.unsqueeze(0)) # [num_targets, num_preds] by broadcasting: True where labels match
        
        valid_ious = ious * label_match # [num_targets, num_preds], 0 where labels don't match, else IoU value
        valid_ious[ious < iou_match_thres] = 0 # filter IoUs below threshold

        matched_preds = torch.zeros(num_preds, dtype=torch.bool, device=p_boxes.device) # keep track of already matched predictions
        tr_pos_count = 0
        
        for j in range(num_targets): # for each target, find best matching prediction
            target_ious = valid_ious[j].clone() # [num_preds] IoUs for this target
            target_ious[matched_preds] = 0 # Exclude already matched predictions
            
            best_iou, best_idx = torch.max(target_ious, dim=0) # get best IoU and index for this target
            
            if best_iou > 0: # if there is a valid match
                tr_pos_count += 1
                matched_preds[best_idx] = True

        total_tr_pos += tr_pos_count
        total_fa_neg += (num_targets - tr_pos_count)
        total_fa_pos += (num_preds - tr_pos_count)

    precision = total_tr_pos / (total_tr_pos + total_fa_pos + 1e-7)
    recall = total_tr_pos / (total_tr_pos + total_fa_neg + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return precision, recall, f1