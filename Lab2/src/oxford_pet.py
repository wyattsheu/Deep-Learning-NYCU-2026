import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2


INPUT_SIZE = (572, 572)

class OxfordPetDataset(Dataset):
    def __init__(self, data_dir, split_type, transform=None):
        self.data_dir = os.path.abspath(data_dir)
        self.split_type = split_type

        split_file = os.path.join(os.path.dirname(self.data_dir), f"{split_type}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            self.names = [line.strip() for line in f.readlines()]

        if transform:
            self.transform = transform
        else:
            if split_type == "train":
                self.transform = A.Compose([
                    A.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Use imagenet stats
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        """
        TODO:
        1. 讀取對應的 image 與 mask。
        2. 【關鍵】實作 Binary Mask 轉換：像素值 1 -> 1 (Foreground)，像素值 2 與 3 -> 0 (Background)。
        3. 套用 transform (若有)。
        """
        fileName = self.names[idx]
        imgPath = os.path.join(self.data_dir, "images", fileName + ".jpg")
        
        # Load image as numpy array for albumentations
        image = np.array(Image.open(imgPath).convert("RGB"))

        if self.split_type.startswith("test"):
            # If test, we still need to apply transform (resizing, norm, to tensor) but no mask
            transformed = self.transform(image=image)
            return transformed["image"], fileName

        maskPath = os.path.join(self.data_dir, "annotations/trimaps", fileName + ".png")
        mask_array = np.array(Image.open(maskPath))
        
        binary_mask = np.zeros_like(mask_array, dtype=np.float32)
        binary_mask[mask_array == 1] = 1.0
        
        # apply albumentations transforms to both image and mask concurrently
        transformed = self.transform(image=image, mask=binary_mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].unsqueeze(0) # [1, H, W]

        return image_tensor, mask_tensor
