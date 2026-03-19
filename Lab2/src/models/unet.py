import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        """
        TODO: 從零開始 (From scratch) 搭建 UNet 的 Encoder, Bottleneck, 與 Decoder。
        不可載入任何預訓練權重。
        """
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left1 = DoubleConv(in_channels, 64)
        self.left2 = DoubleConv(64, 128)
        self.left3 = DoubleConv(128, 256)
        self.left4 = DoubleConv(256, 512)

        self.button = DoubleConv(512, 1024)

        self.right1_conv = ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.right1 = DoubleConv(1024, 512)
        self.right2_conv = ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.right2 = DoubleConv(512, 256)
        self.right3_conv = ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.right3 = DoubleConv(256, 128)
        self.right4_conv = ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.right4 = DoubleConv(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

        pass

    def forward(self, x):
        # TODO: 定義前向傳播邏輯 (記得處理 Skip connections)
        l1 = self.left1(x)
        l2 = self.left2(self.pool(l1))
        l3 = self.left3(self.pool(l2))
        l4 = self.left4(self.pool(l3))
        b = self.button(self.pool(l4))
        r1 = self.right1_conv(b)
        r1 = self.right1(torch.cat([r1, l4], dim=1))
        r2 = self.right2_conv(r1)
        r2 = self.right2(torch.cat([r2, l3], dim=1))
        r3 = self.right3_conv(r2)
        r3 = self.right3(torch.cat([r3, l2], dim=1))
        r4 = self.right4_conv(r3)
        r4 = self.right4(torch.cat([r4, l1], dim=1))
        output = self.output(r4)
        return output


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
