import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate


def train():
    """
    TODO:
    1. 設定 Hyperparameters (Epochs, Batch size, Learning rate 等)。
    2. 建立 DataLoader (Train & Validation)。
    3. 初始化模型 (UNet 或 ResNet34_UNet)。
    4. 設定 Loss function (例如 BCEWithLogitsLoss) 與 Optimizer。
    5. 實作 Training Loop，並在每個 epoch 呼叫 evaluate() 驗證模型。
    6. 儲存驗證集表現最好的模型權重到 saved_models/。
    """
    Epochs = 100
    Batch_size = 64
    Learning_rate = 0.001
    modlle_type = "UNet"  # 可選擇 "UNet" 或 "ResNet34_UNet"

    project_root = os.path.abspath(os.getcwd())
    data_dir = os.path.join(project_root, "dataset", "oxford-iiit-pet")
    train_dataset = OxfordPetDataset(data_dir, split_type="train")
    val_dataset = OxfordPetDataset(data_dir, split_type="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=2,  # 開啟多執行緒幫忙搬資料 (Colab 免費版建議設 2)
        pin_memory=True,  # 讓資料直通 GPU 記憶體，傳輸更快
    )
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if modlle_type == "UNet":
        model = UNet().to(device)
    else:
        model = ResNet34_UNet().to(device)

    print(f"device: {device}")
    print(f"training by {modlle_type} model")

    CrossEntropy_Loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Learning_rate)

    save_dir = os.path.join(project_root, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(Epochs):
        model.train()
        loss_temp = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Epochs}")

        for image, mask in progress_bar:
            image = image.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            out = model(image)
            loss = CrossEntropy_Loss(out, mask)
            loss.backward()
            optimizer.step()

            loss_temp += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_train_loss = loss_temp / len(train_loader)

        print(f"Evaluating Epoch {epoch+1}...")
        val_dice = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{Epochs}] - Train Loss: {avg_train_loss:.4f} | Val Dice Score: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(save_dir, f"best_{modlle_type}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"new modle save at {save_path}\n")


if __name__ == "__main__":
    train()
