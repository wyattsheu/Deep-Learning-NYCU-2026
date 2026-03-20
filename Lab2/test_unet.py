import torch
import sys

sys.path.insert(0, "/Users/wyattsheu/Downloads/Deep-Learning-NYCU-2026/Lab2/src")

from models.unet import UNet

# 測試 UNet 是否能正常執行 forward pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet().to(device)
print("UNet model created successfully")

# 測試輸入
x = torch.randn(2, 3, 572, 572).to(device)
print(f"Input shape: {x.shape}")

try:
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {y.shape}")
    print("✅ Model forward pass successful!")
except Exception as e:
    print(f"❌ Error during forward pass: {e}")
    import traceback

    traceback.print_exc()
