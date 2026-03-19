import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet


def rle_encode(mask: np.ndarray) -> str:
    """Encode a binary mask to RLE using column-major (Fortran) order."""
    pixels = mask.astype(np.uint8).flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def run_inference():
    """
    TODO:
    1. 載入測試集 (test split)。千萬不能用測試集來 train 或 eval 模型！
    2. 載入你訓練好的模型權重 (.pth)。
    3. 對測試集裡的所有圖片進行預測。
    4. 計算整個測試集的「平均」Dice Score。
    5. 將預測結果轉為 Kaggle 需要的格式 (視 Kaggle submission 要求而定)。
    """
    model_type = "UNet"  # 可選擇 "UNet" 或 "ResNet34_UNet"
    model_path = f"saved_models/best_{model_type}.pth"
    data_dir = "dataset/oxford-iiit-pet"
    batch_size = 16
    submission_path = "submission.csv"

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if model_type == "UNet":
        model = UNet().to(device)
    else:
        model = ResNet34_UNet().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. Please train first."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    test_dataset = OxfordPetDataset(data_dir=data_dir, split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    submissions = []

    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).to(torch.uint8).cpu().numpy()

            for pred, image_id in zip(preds, image_ids):
                binary_mask = pred.squeeze(0)
                encoded_mask = rle_encode(binary_mask)
                submissions.append((image_id, encoded_mask))

    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])
        writer.writerows(submissions)

    print(f"Inference complete. Submission saved to {submission_path}")
    print(
        "Note: test split has no visible ground truth, so Dice score is not computed."
    )


if __name__ == "__main__":
    run_inference()
