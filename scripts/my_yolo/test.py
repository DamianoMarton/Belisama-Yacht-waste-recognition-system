import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from model_res import MyYolo_res
from dataset import MyYoloDataset
from metrics_vect import batch_post_process_vect, compute_batch_performance_vect

DATASET_PATH = "./dataset/dataset"
MODEL_PATH = "./trained_models/my_yolo/best_model_2401.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_class_colors(num_classes):
    """
    Generates distinct colors for each class.

    Args:
        num_classes (int): Number of classes.

    Returns:
        list: List of RGB color tuples of length num_classes.
    """
    colors = [(i*(255//(2*num_classes)), i*(255//(2*num_classes)), i*(255//(2*num_classes))) for i in range(num_classes)]
    return colors

def unnormalize(tensor):
    """ 
    Converts a normalized tensor to an RGB image with pixel values in [0, 255].
    We use the standard ImageNet mean and std for unnormalization.

    Args:
        tensor (torch.Tensor): Normalized image tensor of shape (3, H, W).

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3) with uint8 type.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) # reshape for broadcasting
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    img = tensor.permute(1, 2, 0).cpu().numpy() # to numpy array (we need the tensor to be in CPU)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def evaluate_at_threshold(model, 
                          dataloader, 
                          device, 
                          threshold) -> tuple[float, float, float]:
    """ 
    Calculates average Precision, Recall, and F1 score for a specific threshold 

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the computations on.
        threshold (float): Confidence threshold.
    
    Returns:
        tuple: Average Precision, Recall, and F1 score.
    """
    model.eval() # Set model to evaluation mode
    precs, recs, f1s = [], [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x) # forward pass
            p, r, f1 = compute_batch_performance_vect(y_pred, y, conf_thres=threshold)
            precs.append(p)
            recs.append(r)
            f1s.append(f1)

    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    mean_f1 = np.mean(f1s)
            
    return mean_prec, mean_rec, mean_f1

def optimize_thresholds(model, 
                        dataloader, 
                        device) -> float:
    """ 
    Finds the optimal confidence threshold that maximizes the F1 score.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the computations on.

    Returns:
        float: Optimal confidence threshold.
    """
    print("Retrieving optimal confidence threshold...\n")
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_f1 = -1
    best_t = 0.5
    results = []

    for t in thresholds:
        p, r, f1 = evaluate_at_threshold(model, dataloader, device, t)
        results.append((t, p, r, f1))
        print(f"Conf: {t:.2f} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    print(f"Best Threshold: {best_t:.2f} (F1: {best_f1:.4f})\n")
    return best_t

def visual_inference(model, 
                     dataset, 
                     device, 
                     threshold, 
                     num_images=3) -> None: # using Copilot to plot images
    """ 
    Performs visual inference on random images from the dataset and saves the results.

    Args:
        model (torch.nn.Module): The trained model.
        dataset (MyYoloDataset): The dataset to sample images from.
        device (torch.device): Device to run the computations on.
        threshold (float): Confidence threshold for predictions.
        num_images (int): Number of random images to process.

    Returns:
        None 
    """
    print(f"Generating visual inference (Threshold: {threshold})")
    model.eval()
    out_dir = os.path.join("runs", "detect", "my_yolo")
    os.makedirs(out_dir, exist_ok=True)
    
    colors = get_class_colors(dataset.num_classes)
    
    indices = np.random.choice(len(dataset), num_images, replace=False) # num_images unique random indices from dataset

    for i, idx in enumerate(indices):
        img_tensor, _ = dataset[idx]
        x = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad(): # no gradient computation
            predictions = model(x) # forward pass
            results = batch_post_process_vect(predictions, 
                                              conf_thres=threshold, 
                                              iou_thres=0.45)
        
        # tensor to BGR image for OpenCV
        img_plot = unnormalize(img_tensor)
        img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR)
        
        res = results[0] # the list has one element since batch size is 1
        boxes = res['boxes'].cpu().numpy() # convert to numpy
        confidences = res['confidences'].cpu().numpy()
        labels = res['labels'].cpu().numpy()
        
        for box, score, label in zip(boxes, confidences, labels): # Copilot suggestion
            x1, y1, x2, y2 = (box * 640).astype(int)
            color = colors[int(label)]
            
            cv2.rectangle(img_plot, (x1, y1), (x2, y2), color, 2)
            
            label_name = dataset.class_names[int(label)]
            caption = f"{label_name} {score:.2f}"
            (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_plot, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img_plot, caption, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        path = os.path.join(out_dir, f"test_sample_{i}.jpg")
        cv2.imwrite(path, img_plot)
        print(f"Saved: {path} (Found {len(boxes)} objects)")

def main():
    try:
        test_dataset = MyYoloDataset(DATASET_PATH, split='test')
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(f"Test Set loaded: {len(test_dataset)} images.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    num_classes = test_dataset.num_classes
    model = MyYolo_res(num_classes).to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Weights loaded successfully from: {MODEL_PATH}")
    else:
        print(f"ERROR: File {MODEL_PATH} not found!")
        exit()


    #best_threshold = optimize_thresholds(model, test_loader, DEVICE)
    best_threshold = 0.68
    
    visual_inference(model, test_dataset, DEVICE, threshold=best_threshold, num_images=10)

main()