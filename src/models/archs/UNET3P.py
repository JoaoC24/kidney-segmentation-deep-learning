import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.archs.utils.network_blocks import (
    DoubleConv,
    ConvBlock,
    ResizeProject
)


class UNET3P(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        base_ch=64,
        cat_ch=64,
        deep_supervision=True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        fusion_ch = cat_ch * 5

        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)
        self.enc5 = DoubleConv(base_ch * 8, base_ch * 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.proj_d4 = nn.ModuleDict(
            {
                "e1": ResizeProject(base_ch, cat_ch),
                "e2": ResizeProject(base_ch * 2, cat_ch),
                "e3": ResizeProject(base_ch * 4, cat_ch),
                "e4": ResizeProject(base_ch * 8, cat_ch),
                "e5": ResizeProject(base_ch * 16, cat_ch),
            }
        )
        self.proj_d3 = nn.ModuleDict(
            {
                "e1": ResizeProject(base_ch, cat_ch),
                "e2": ResizeProject(base_ch * 2, cat_ch),
                "e3": ResizeProject(base_ch * 4, cat_ch),
                "d4": ResizeProject(fusion_ch, cat_ch),
                "e5": ResizeProject(base_ch * 16, cat_ch),
            }
        )
        self.proj_d2 = nn.ModuleDict(
            {
                "e1": ResizeProject(base_ch, cat_ch),
                "e2": ResizeProject(base_ch * 2, cat_ch),
                "d3": ResizeProject(fusion_ch, cat_ch),
                "e4": ResizeProject(base_ch * 8, cat_ch),
                "e5": ResizeProject(base_ch * 16, cat_ch),
            }
        )
        self.proj_d1 = nn.ModuleDict(
            {
                "e1": ResizeProject(base_ch, cat_ch),
                "d2": ResizeProject(fusion_ch, cat_ch),
                "e3": ResizeProject(base_ch * 4, cat_ch),
                "e4": ResizeProject(base_ch * 8, cat_ch),
                "e5": ResizeProject(base_ch * 16, cat_ch),
            }
        )

        self.dec4 = ConvBlock(fusion_ch, fusion_ch)
        self.dec3 = ConvBlock(fusion_ch, fusion_ch)
        self.dec2 = ConvBlock(fusion_ch, fusion_ch)
        self.dec1 = ConvBlock(fusion_ch, fusion_ch)

        self.final_conv = nn.Conv2d(fusion_ch, out_channels, kernel_size=1)
        self.supervision = nn.ModuleList(
            [
                nn.Conv2d(base_ch * 16, out_channels, kernel_size=1),
                nn.Conv2d(fusion_ch, out_channels, kernel_size=1),
                nn.Conv2d(fusion_ch, out_channels, kernel_size=1),
                nn.Conv2d(fusion_ch, out_channels, kernel_size=1),
            ]
        )

    def _fuse(self, projections, features, target_size):
        return torch.cat([projections[name](features[name], target_size) for name in projections], dim=1)

    def forward(self, x):
        image_size = x.shape[-2:]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.dec4(
            self._fuse(
                self.proj_d4,
                {"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5},
                e4.shape[-2:],
            )
        )
        d3 = self.dec3(
            self._fuse(
                self.proj_d3,
                {"e1": e1, "e2": e2, "e3": e3, "d4": d4, "e5": e5},
                e3.shape[-2:],
            )
        )
        d2 = self.dec2(
            self._fuse(
                self.proj_d2,
                {"e1": e1, "e2": e2, "d3": d3, "e4": e4, "e5": e5},
                e2.shape[-2:],
            )
        )
        d1 = self.dec1(
            self._fuse(
                self.proj_d1,
                {"e1": e1, "d2": d2, "e3": e3, "e4": e4, "e5": e5},
                e1.shape[-2:],
            )
        )
        output = self.final_conv(d1)

        if not self.deep_supervision:
            return output

        return [
            F.interpolate(self.supervision[0](e5), size=image_size, mode="bilinear", align_corners=False),
            F.interpolate(self.supervision[1](d4), size=image_size, mode="bilinear", align_corners=False),
            F.interpolate(self.supervision[2](d3), size=image_size, mode="bilinear", align_corners=False),
            F.interpolate(self.supervision[3](d2), size=image_size, mode="bilinear", align_corners=False),
            output,
        ]


if __name__ == "__main__":
    model = UNET3P(in_channels=1, out_channels=2, deep_supervision=True)

    device = "cpu"
    model = model.to(device)
    model.train()

    batch_size = 3
    in_channels = 1
    height = width = 128
    x = torch.randn(batch_size, in_channels, height, width, device=device)

    out = model(x)

    print("Output shapes:")
    for i, o in enumerate(out):
        print(f"  Level {i}: {o.shape}")
