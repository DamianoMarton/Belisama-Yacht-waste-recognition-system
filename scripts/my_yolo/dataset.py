import os
import cv2
import yaml
import torch
from torch.utils.data import Dataset

class MyYoloDataset(Dataset):
    def __init__(self, dataset_path: str, split='train'):
        with open(os.path.join(dataset_path, "data.yaml"), 'r') as file:
            data_info = yaml.safe_load(file)
        
        self.class_names = data_info['names']
        self.num_classes = len(self.class_names)
        self.grid_size = 20
        self.image_size = 640
        self.split = split
        
        if split == 'train':
            self.img_dir = os.path.join(dataset_path, data_info['train'])
        elif split == 'val':
            self.img_dir = os.path.join(dataset_path, data_info['val'])
        else:
            self.img_dir = os.path.join(dataset_path, data_info['test'])
            
        self.label_dir = self.img_dir.replace("images", "labels")
        self.file_list = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

        # Normalization parameters (ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Args:
            None.

        Returns:
            Total number of samples.
        """
        return len(self.file_list)

    def augment(self, img: torch.Tensor) -> torch.Tensor:
        """
        Applies random augmentations (saturation, noise, channel shuffle, grayscale) to the input image tensor.
        
        Args:
            img (torch.Tensor): Input image tensor of shape [3, H, W] with values in [0, 1].

        Returns:
            torch.Tensor: Augmented image tensor of shape [3, H, W] with values in [0, 1].
        """
        # Random saturation 50%
        if torch.rand(1) < 0.5:
            factor = torch.rand(1) + 0.5 # Range [0.5, 1.5]
            weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            grayscale = (img * weights).sum(dim=0, keepdim=True)
            img = torch.lerp(grayscale, img, factor)
            img = img.clamp(0, 1)
                    
        # Gaussian noise 30%
        if torch.rand(1) < 0.3:
            noise = torch.randn_like(img) * 0.02 # Reduced to 0.02 to not be too destructive
            img = (img + noise).clamp(0, 1)
            
        # Channel shuffle 20%
        if torch.rand(1) < 0.2:
            indices = torch.randperm(3)
            img = img[indices, :, :]

        # Random grayscale 10%
        if torch.rand(1) < 0.1:
            weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            grayscale = (img * weights).sum(dim=0, keepdim=True)
            img = grayscale.repeat(3, 1, 1)
            
        return img

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Loads an image from the specified path and applies necessary transformations.

        Args:
            path (str): Path to the image file.

        Returns:
            torch.Tensor: Transformed image tensor of shape [3, H, W].
        """ 
        # Loading with OpenCV
        img_cv = cv2.imread(path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Conversion to Tensor [3, H, W] and range [0, 1]
        img = torch.from_numpy(img_cv).float().permute(2, 0, 1) / 255.0
        
        # Apply augmentation only in training
        if self.split == 'train':
            img = self.augment(img)
            
        # Final normalization
        img = (img - self.mean) / self.std
        return img
    
    def _load_label(self, path: str) -> torch.Tensor:
        """
        Loads labels from the specified path and converts them to the required tensor format.

        Args:
            path (str): Path to the label file.

        Returns:
            torch.Tensor: Label tensor of shape [5 + num_classes, grid_size, grid_size].
        """
        labels = torch.zeros((5 + self.num_classes, self.grid_size, self.grid_size))

        if not os.path.exists(path): 
            return labels
        
        with open(path, 'r') as f:
            for line in f:
                class_id, x_c, y_c, w, h = map(float, line.strip().split())

                grid_x = int(x_c * self.grid_size)
                grid_y = int(y_c * self.grid_size)
                grid_x = max(0, min(grid_x, self.grid_size - 1))
                grid_y = max(0, min(grid_y, self.grid_size - 1))

                x_rel = (x_c * self.grid_size) - grid_x
                y_rel = (y_c * self.grid_size) - grid_y

                labels[0, grid_y, grid_x] = 1.0
                labels[1, grid_y, grid_x] = x_rel
                labels[2, grid_y, grid_x] = y_rel
                labels[3, grid_y, grid_x] = w
                labels[4, grid_y, grid_x] = h
                labels[5 + int(class_id), grid_y, grid_x] = 1.0

        return labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches the image and label tensors for the given index.
        Applies random horizontal flip augmentation during training.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the image tensor and label tensor.
        """
        img_name = self.file_list[idx]
        label_name = img_name.replace(".jpg", ".txt")

        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        image = self._load_image(img_path) 
        label = self._load_label(label_path)

        if self.split == 'train':
            # Random horizontal flip 50% (and adjust labels)
            if torch.rand(1) < 0.5:
                image = torch.flip(image, dims=[2])
                label = torch.flip(label, dims=[2])
                mask = label[0, :, :] > 0
                label[1, mask] = 1.0 - label[1, mask]
        
        return image, label