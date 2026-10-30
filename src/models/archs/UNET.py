import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.archs.utils.network_blocks import DoubleConv

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, norm=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNET(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        widths = (64, 128, 256, 512)
        self.enc1 = DoubleConv(in_channels, widths[0], norm=False)
        self.enc2 = DoubleConv(widths[0], widths[1], norm=False)
        self.enc3 = DoubleConv(widths[1], widths[2], norm=False)
        self.enc4 = DoubleConv(widths[2], widths[3], norm=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(widths[3], widths[3] * 2, norm=False, dropout=0.1)

        self.up4 = UpBlock(widths[3] * 2, widths[3], widths[3])
        self.up3 = UpBlock(widths[3], widths[2], widths[2])
        self.up2 = UpBlock(widths[2], widths[1], widths[1])
        self.up1 = UpBlock(widths[1], widths[0], widths[0])
        self.head = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        x = self.bottleneck(self.pool(enc4))
        x = self.up4(x, enc4)
        x = self.up3(x, enc3)
        x = self.up2(x, enc2)
        x = self.up1(x, enc1)
        return self.head(x)


if __name__ == "__main__":
    model = UNET(in_channels=1, num_classes=2)
    device = "cpu"
    model.train()
    batch_size = 2
    in_channels = 1
    height = width = 128
    x = torch.randn(batch_size, in_channels, height, width)

    out = model(x)
