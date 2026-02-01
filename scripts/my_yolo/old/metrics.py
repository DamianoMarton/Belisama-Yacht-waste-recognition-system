import torch
import torchvision.ops as ops

def bbox_iou(box1, box2):
    """
    Computes IoU between two single boxes.
    
    Args:
        box1: Tensor of 4 elements [x1, y1, x2, y2]
        box2: Tensor of 4 elements [x1, y1, x2, y2]
        
    Returns:
        Tensor with the IoU value (scalar)
    """ 
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])

    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    
    inter_area = inter_w * inter_h

    # single areas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # union
    union = area1 + area2 - inter_area
    
    return inter_area / (union + 1e-7)

def batch_target_to_boxes(target_batch):
    """
    Transforms a batch of YOLO-style target (ground truth) tensors into bounding boxes and class labels.
    
    Args:
        target_batch: Tensor of shape [batch_size, 5 + num_classes, S, S]
    
    Returns:
        List of length batch_size, where each element is:
        {'boxes': Tensor[N, 4], 'labels': Tensor[N]} N is number of ground truth boxes in the image
    """
    batch_size = target_batch.shape[0]
    S = target_batch.shape[2]
    device = target_batch.device
    
    target_batch = target_batch.permute(0, 2, 3, 1) # [B, 5 + num_classes, S, S] to [B, S, S, 5 + num_classes]
    
    batch_output = []

    for b in range(batch_size):
        target = target_batch[b] # [S, S, 5 + num_classes]
        
        indices = (target[..., 0] == 1).nonzero(as_tuple=False) #[N, 2] where N is number of objects and each row is [row, col]
        
        if indices.shape[0] == 0: # No objects in this image
            batch_output.append({
                'boxes': torch.empty((0, 4), device=device),
                'labels': torch.empty(0, device=device)
            })
            continue

        img_boxes = []
        img_labels = []

        for idx in indices:
            row, col = idx[0].item(), idx[1].item() # otherwise idx is a tensor, now it's a python int
            cell_data = target[row, col]
            
            x_rel, y_rel, w, h = cell_data[1:5]
            
            class_id = torch.argmax(cell_data[5:])

            # Convert to global coordinates (0-1)
            x_c = (col + x_rel) / S 
            y_c = (row + y_rel) / S
            
            x1 = x_c - w/2
            y1 = y_c - h/2
            x2 = x_c + w/2
            y2 = y_c + h/2
            
            img_boxes.append(torch.stack([x1, y1, x2, y2])) # create tensor [4]
            img_labels.append(class_id)

        batch_output.append({ # append a dict for this image
            "boxes": torch.stack(img_boxes).to(device), # [N, 4]
            "labels": torch.stack(img_labels).to(device) # [N]
        })

    return batch_output

def batch_post_process(predictions, conf_thres=0.5, iou_thres=0.4):
    """
    Every raw prediction tensor is post-processed to yield final bounding boxes, scores, and class labels
    Also applies Non-Maximum Suppression to filter overlapping boxes 

    Args:
        predictions: Tensor of shape [batch_size, 5 + num_classes, S, S] not normalized in any way
        conf_thres: Confidence threshold to filter boxes before NMS in [0,1]
        iou_thres: IoU threshold for Non-Maximum Suppression in [0,1]

    Returns:
        List of length batch_size, where each element is:
        {'boxes': Tensor[N, 4], 'labels': Tensor[N], 'confidences': Tensor[N]} N is number of final boxes after NMS
    """ 
    batch_size = predictions.shape[0]
    S = predictions.shape[2]
    device = predictions.device
    
    final_results = []

    for b in range(batch_size):
        image_boxes = []
        image_scores = []
        image_labels = []

        for row in range(S): # did not vectorize for clarity (to myself)
            for col in range(S):
                cell_data = predictions[b, :, row, col]
                
                conf = torch.sigmoid(cell_data[0])
                
                if conf < 0.1: # rapid filtering to speed up NMS and avoid noise
                    continue
                    
                class_probs = torch.softmax(cell_data[5:], dim=0) # dimension 0 since it's a 1D tensor
                max_score, label = torch.max(class_probs, dim=0)
                
                combined_conf = conf * max_score # total confidence
                
                if combined_conf > conf_thres:
                    x_rel = torch.sigmoid(cell_data[1])
                    y_rel = torch.sigmoid(cell_data[2])
                    w = torch.exp(cell_data[3])
                    h = torch.exp(cell_data[4])
                    
                    # global coordinates
                    x_center = (col + x_rel) / S
                    y_center = (row + y_rel) / S
                    
                    x1 = x_center - w/2
                    y1 = y_center - h/2
                    x2 = x_center + w/2
                    y2 = y_center + h/2
                    
                    image_boxes.append(torch.stack([x1, y1, x2, y2]))
                    image_scores.append(combined_conf)
                    image_labels.append(label)

        if len(image_boxes) == 0:
            final_results.append({
                'boxes': torch.empty((0, 4), device=device),
                'confidences': torch.empty(0, device=device),
                'labels': torch.empty(0, device=device)
            })
            continue

        image_boxes = torch.stack(image_boxes).to(device)
        image_scores = torch.stack(image_scores).to(device)
        image_labels = torch.stack(image_labels).to(device)

        # Non-maxumum suppression
        indices = torch.argsort(image_scores, descending=True) # indices sorted by score
        
        keep_indices = []
        while len(indices) > 0: # structure suggested by Copilot
            current_idx = indices[0]
            keep_indices.append(current_idx.item()) # tensor to python int
            
            if len(indices) == 1:
                break
                
            remaining_indices = indices[1:]
            filtered_indices = []
            
            for idx in remaining_indices:
                iou = bbox_iou(image_boxes[current_idx], image_boxes[idx])
                if iou <= iou_thres: # if there is not too much overlap, keep the box
                    filtered_indices.append(idx.item())
            
            indices = torch.tensor(filtered_indices, device=device, dtype=torch.long) # dtype long (int64) for indexing

        final_results.append({
            'boxes': image_boxes[keep_indices],
            'confidences': image_scores[keep_indices],
            'labels': image_labels[keep_indices]
        })

    return final_results

def compute_batch_performance(predictions, targets, 
                              conf_thres=0.4, iou_match_thres=0.5, iou_NMS_thres=0.4):
    """
    Computes precision, recall, and F1-score for a batch of predictions and targets.

    Args:
        predictions: Tensor of shape [batch_size, 5 + num_classes, S, S]
        targets: Tensor of shape [batch_size, 5 + num_classes, S, S]
        conf_thres: Confidence threshold to consider a prediction as True Positive in [0,1]
        iou_NMS_thres: IoU threshold for Non-Maximum Suppression in [0,1]
        iou_match_thres: IoU threshold to consider a prediction as True Positive in [0,1]
    """
    pred_results = batch_post_process(predictions, conf_thres, iou_NMS_thres) # [batch_size, {...}]
    target_results = batch_target_to_boxes(targets) # [batch_size, {...}]
    
    tr_pos, fa_pos, fa_neg = 0, 0, 0

    for i in range(len(pred_results)): # for each image in the batch
        pred = pred_results[i]
        targ = target_results[i]
        
        p_boxes = pred['boxes']
        p_labels = pred['labels']
        t_boxes = targ['boxes']
        t_labels = targ['labels']

        if t_boxes.shape[0] == 0:
            fa_pos += p_boxes.shape[0] # All predictions are false positives
            continue

        if p_boxes.shape[0] == 0:
            fa_neg += t_boxes.shape[0] # All real objects are false negatives
            continue
        
        # first, we look for True Positives
        # track which predictions have already been matched (0 = not matched, 1 = matched)
        matched_preds = torch.zeros(p_boxes.shape[0], device=p_boxes.device)

        for j in range(t_boxes.shape[0]):
            best_iou = -1.0
            best_idx = -1.0
            
            target_box = t_boxes[j]
            target_label = t_labels[j]

            for k in range(p_boxes.shape[0]):
                # if the prediction has already been assigned or has the wrong class, skip
                if matched_preds[k] == 1 or p_labels[k] != target_label:
                    continue
                
                iou = bbox_iou(target_box, p_boxes[k])
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = k

            if best_iou >= iou_match_thres:
                tr_pos += 1 # True Positive
                matched_preds[best_idx] = 1 # mark this prediction as used
            else:
                # if we didn't find a good box for this real object
                fa_neg += 1

        # predictions still at zero are False Positives
        fa_pos += (matched_preds == 0).sum().item() # False Positives (tensor to python int)

    # final metrics
    precision = tr_pos / (tr_pos + fa_pos + 1e-7)
    recall = tr_pos / (tr_pos + fa_neg + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return precision, recall, f1