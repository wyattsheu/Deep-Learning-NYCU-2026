import os
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from oxford_pet import OxfordPetDataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate


def dice_loss_from_logits(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits).float()
    targets = targets.float()

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        probs.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return 1.0 - dice.mean()


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
    Batch_size = 16
    Learning_rate = 1e-4
    modlle_type = "UNet"  # 可選擇 "UNet" 或 "ResNet34_UNet"

    project_root = os.path.abspath(os.getcwd())
    data_dir = os.path.join(project_root, "dataset", "oxford-iiit-pet")
    train_dataset = OxfordPetDataset(data_dir, split_type="train")
    val_dataset = OxfordPetDataset(data_dir, split_type="val")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    use_cuda = device.type == "cuda"

    # Fixed-size image training can benefit from cuDNN autotuner on CUDA.
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=2,  # 開啟多執行緒幫忙搬資料 (Colab 免費版建議設 2)
        pin_memory=use_cuda,  # 僅 CUDA 啟用 pin_memory；MPS/CPU 不需要
    )
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False)

    if modlle_type == "UNet":
        model = UNet().to(device)
    else:
        model = ResNet34_UNet().to(device)

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    print(f"device: {device}")
    print(f"training by {modlle_type} model")

    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Epochs, eta_min=1e-6
    )

    amp_enabled = use_cuda
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    save_dir = os.path.join(project_root, "saved_models")
    os.makedirs(save_dir, exist_ok=True)
    best_dice = 0.0
    epochs_no_improve = 0
    early_stop_patience = 10

    for epoch in range(Epochs):
        model.train()
        loss_temp = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Epochs}")

        for image, mask in progress_bar:
            image = image.to(device, non_blocking=use_cuda)
            mask = mask.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)
            if amp_enabled and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_context = torch.amp.autocast(device_type="cuda", enabled=True)
            elif amp_enabled:
                autocast_context = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast_context = nullcontext()

            with autocast_context:
                out = model(image)
                bce_loss = bce_loss_fn(out, mask)
                dice_loss = dice_loss_from_logits(out, mask)
                # 調整 Loss 權重：0.2 BCE + 0.8 Dice
                loss = 0.2 * bce_loss + 0.8 * dice_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_temp += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_train_loss = loss_temp / len(train_loader)

        print(f"Evaluating Epoch {epoch+1}...")
        val_dice = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch+1}/{Epochs}] - Train Loss: {avg_train_loss:.4f} | Val Dice Score: {val_dice:.4f}"
        )

        scheduler.step()  # 更新 CosineAnnealingLR

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_improve = 0
            save_path = os.path.join(save_dir, f"best_{modlle_type}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"new modle save at {save_path}\n")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).\n")
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered. Training stopped.")
                break


if __name__ == "__main__":
    train()
