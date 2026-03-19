import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class OxfordPetDataset(Dataset):
    def __init__(self, data_dir, split_type, transform=None):
        self.data_dir = os.path.abspath(data_dir)
        self.split_type = split_type

        split_file = os.path.join(os.path.dirname(self.data_dir), f"{split_type}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            self.names = [line.strip() for line in f.readlines()]

        self.transform = transform or T.Compose([T.Resize((256, 256)), T.ToTensor()])

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
        image = Image.open(imgPath).convert("RGB")
        image_tensor = self.transform(image)

        if self.split_type == "test":
            return image_tensor, fileName

        maskPath = os.path.join(self.data_dir, "annotations/trimaps", fileName + ".png")
        mask = Image.open(maskPath).resize((256, 256), resample=Image.NEAREST)
        mask_array = np.array(mask)
        binary_mask = np.zeros_like(mask_array, dtype=np.float32)
        binary_mask[mask_array == 1] = 1.0

        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)  # [1, H, W]
        return image_tensor, mask_tensor
