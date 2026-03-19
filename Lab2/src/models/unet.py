import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # Use Conv instead of ConvTranspose2d for more controlled upsampling
        self.right1_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.right1_conv = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.right1 = DoubleConv(1024, 512)
        
        self.right2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.right2_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.right2 = DoubleConv(512, 256)
        
        self.right3_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.right3_conv = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.right3 = DoubleConv(256, 128)
        
        self.right4_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.right4_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.right4 = DoubleConv(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    @staticmethod
    def _center_crop(skip, target):
        _, _, h, w = skip.shape
        _, _, th, tw = target.shape
        dh = (h - th) // 2
        dw = (w - tw) // 2
        return skip[:, :, dh : dh + th, dw : dw + tw]

    def forward(self, x):
        original_shape = x.shape
        # Encoder path: padding=1 ensures feature maps keep spatial dimensions
        l1 = self.left1(x)
        l2 = self.left2(self.pool(l1))
        l3 = self.left3(self.pool(l2))
        l4 = self.left4(self.pool(l3))
        b = self.button(self.pool(l4))

        # Decoder path: bilinear upsampling + Conv provides precise spatial control
        r1 = F.interpolate(b, size=l4.shape[2:], mode='bilinear', align_corners=False)
        r1 = self.right1_conv(r1)
        l4_crop = self._center_crop(l4, r1)
        r1 = self.right1(torch.cat([r1, l4_crop], dim=1))

        r2 = F.interpolate(r1, size=l3.shape[2:], mode='bilinear', align_corners=False)
        r2 = self.right2_conv(r2)
        l3_crop = self._center_crop(l3, r2)
        r2 = self.right2(torch.cat([r2, l3_crop], dim=1))

        r3 = F.interpolate(r2, size=l2.shape[2:], mode='bilinear', align_corners=False)
        r3 = self.right3_conv(r3)
        l2_crop = self._center_crop(l2, r3)
        r3 = self.right3(torch.cat([r3, l2_crop], dim=1))

        r4 = F.interpolate(r3, size=l1.shape[2:], mode='bilinear', align_corners=False)
        r4 = self.right4_conv(r4)
        l1_crop = self._center_crop(l1, r4)
        r4 = self.right4(torch.cat([r4, l1_crop], dim=1))

        output = self.output(r4)
        # Final resize to match input shape to handle any remaining discrepancies
        if output.shape[2:] != original_shape[2:]:
            output = F.interpolate(output, size=original_shape[2:], mode='bilinear', align_corners=False)
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
