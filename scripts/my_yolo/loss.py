import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, num_classes, 
                 lambda_box=3.0, 
                 lambda_obj=2.5, 
                 lambda_noobj=5.0, 
                 lambda_class=1.0):
        
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        self.obj_loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(self.device) # reduction='none' to obtain loss with every element without averaging
        self.class_loss_fn = nn.CrossEntropyLoss().to(self.device)

        self.lambda_box = lambda_box # Weight for coordinate error
        self.lambda_obj = lambda_obj # Weight for object presence
        self.lambda_noobj = lambda_noobj # Weight for background (no object)
        self.lambda_class = lambda_class # Weight for class error

    def forward(self, predictions, targets) -> torch.Tensor:
        """
        Computes the YOLO loss given predictions and targets.

        Args:
            predictions (torch.Tensor): Tensor of shape [B, 5 + num_classes, grid_size, grid_size]
            targets (torch.Tensor): Tensor of shape [B, 5 + num_classes, grid_size, grid_size]

        Returns:
            torch.Tensor: Scalar tensor (float) representing the total loss
        """
        grid_res = predictions.shape[2]  # Assuming square grid: 20
        
        pred_conf = predictions[:, 0, :, :] # [B, grid_size, grid_size]
        pred_boxes = predictions[:, 1:5, :, :] # [B, 4, grid_size, grid_size]
        pred_classes = predictions[:, 5:, :, :] # [B, num_classes, grid_size, grid_size]
        
        target_conf = targets[:, 0, :, :] # [B, grid_size, grid_size]
        target_boxes = targets[:, 1:5, :, :] # [B, 4, grid_size, grid_size]
        target_classes = targets[:, 5:, :, :] # [B, num_classes, grid_size, grid_size]

        # Masks to separate cells with objects from those with background
        obj_mask = (target_conf == 1) # [B, grid_size, grid_size]
        noobj_mask = (target_conf == 0) # [B, grid_size, grid_size]

        # CONFIDENCE LOSS (Object, No-Object)
        full_conf_loss = self.obj_loss_fn(pred_conf, target_conf) # [B, grid_size, grid_size]
        
        if obj_mask.any():
            loss_obj = full_conf_loss[obj_mask].mean() # averages a 1D tensor
        else:
            loss_obj = torch.tensor(0.0, device=predictions.device)
        if noobj_mask.any():
            loss_noobj = full_conf_loss[noobj_mask].mean() # averages a 1D tensor
        else:
            loss_noobj = torch.tensor(0.0, device=predictions.device)

        # COORDINATE LOSS
        if obj_mask.any():
            # Extract only the cells with objects
            p_box = pred_boxes.permute(0, 2, 3, 1)[obj_mask] # [num_obj_cells, 4] (channel last)
            t_box = target_boxes.permute(0, 2, 3, 1)[obj_mask] # [num_obj_cells, 4] (channel last)
             
            # Retrieve the x,y indices of the object cells
            _, y_idx, x_idx = obj_mask.nonzero(as_tuple=True) # Each is [num_obj_cells (true positions)]
            
            # Global coordinates
            p_x_rel = torch.sigmoid(p_box[:, 0]) # [num_obj_cells]
            p_y_rel = torch.sigmoid(p_box[:, 1])
            p_w = torch.sigmoid(p_box[:, 2])
            p_h = torch.sigmoid(p_box[:, 3])
            
            p_x_abs = (x_idx.float() + p_x_rel) / grid_res
            p_y_abs = (y_idx.float() + p_y_rel) / grid_res
            
            t_x_abs = (x_idx.float() + t_box[:, 0]) / grid_res
            t_y_abs = (y_idx.float() + t_box[:, 1]) / grid_res
            t_w, t_h = t_box[:, 2], t_box[:, 3]

            # IoU Calculation
            p_x1, p_y1 = p_x_abs - p_w/2, p_y_abs - p_h/2
            p_x2, p_y2 = p_x_abs + p_w/2, p_y_abs + p_h/2
            t_x1, t_y1 = t_x_abs - t_w/2, t_y_abs - t_h/2
            t_x2, t_y2 = t_x_abs + t_w/2, t_y_abs + t_h/2

            inter_x1 = torch.max(p_x1, t_x1)
            inter_y1 = torch.max(p_y1, t_y1)
            inter_x2 = torch.min(p_x2, t_x2)
            inter_y2 = torch.min(p_y2, t_y2)
            
            inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0) # like NumPy's clip
            p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
            t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
            union = p_area + t_area - inter_area
            
            iou = inter_area / (union + 1e-6)
            loss_box = (1.0 - iou).mean()
        else:
            loss_box = torch.tensor(0.0, device=predictions.device)

        # CLASS LOSS
        if obj_mask.any():
            actual_classes_pred = pred_classes.permute(0, 2, 3, 1)[obj_mask] # [num_obj_cells, num_classes]
            actual_classes_target = torch.argmax(
                            target_classes.permute(0, 2, 3, 1)[obj_mask], dim=1) # [num_obj_cells] every cell has true class index (this is what CrossEntropyLoss needs)
            
            loss_class = self.class_loss_fn(actual_classes_pred, actual_classes_target)
        else:
            loss_class = torch.tensor(0.0, device=predictions.device)

        # TOTAL LOSS
        total_loss = (self.lambda_obj * loss_obj + 
                      self.lambda_noobj * loss_noobj +
                      self.lambda_box * loss_box + 
                      self.lambda_class * loss_class)

        return total_loss