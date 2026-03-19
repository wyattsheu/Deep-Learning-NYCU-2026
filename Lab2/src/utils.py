import torch
import numpy as np

def calculate_dice_score(pred, target):
    """
    TODO: 實作 Dice Score 邏輯
    公式: 2 * (number of common pixels) / (predicted img size + ground truth img size)
    注意: pred 與 target 應該要是 binary masks (0 與 1)
    """
    pred = pred.float()
    target = target.float()

    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    pred = (pred > 0.5).float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + 1e-8) / (denominator + 1e-8)
    return dice.mean().item()

# 你可以在這裡加入其他的輔助函式，例如視覺化預測結果的 function
