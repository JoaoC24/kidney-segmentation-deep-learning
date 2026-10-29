import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.archs.utils.VIT import VisionTransformerFeatureMap
from src.models.archs.utils.network_blocks import (
    DoubleConv,
    EncoderBlock,
    DecoderBlock
)


class TransUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        image_size: int = 128,
        encoder_channels: tuple[int, int, int] = (64, 128, 256),
        vit_patch_size: int | None = 4, # parece me bem: 16 patches de 16 de lado cada um
        vit_num_patches: int | None = None,
        vit_embed_dim: int = 384, # talvez uma potencia de 2, tipo 512 para ser o dobro do c3 (que entra e sai)
        vit_depth: int = 6, # vai ser 1 provavelmente
        vit_num_heads: int = 6, # 4 ou 8 maybe
        vit_mlp_ratio: float = 4.0, # ver no que altera
        dropout: float = 0.0, # considerar, igual para o de baixo
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError("image_size must be divisible by 8 for the 3-layer encoder.")
        if len(encoder_channels) != 3:
            raise ValueError("encoder_channels must contain exactly 3 values.")

        c1, c2, c3 = encoder_channels
        bottleneck_size = image_size // 8

        self.enc1 = EncoderBlock(in_channels, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)

        self.vit = VisionTransformerFeatureMap(
            in_channels=c3,
            image_size=bottleneck_size,
            patch_size=vit_patch_size,
            num_patches=vit_num_patches,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            #out_channels=c3, #podia sair em embed_dim, e o vit_upsample convertia para c3
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        # originalmente: nao mudava o numero de channels, pq o VIT ja entregava em c3; mas assim parece me ser mais util
        self.vit_upsample = nn.Sequential(
            nn.ConvTranspose2d(vit_embed_dim, c3, kernel_size=self.vit.patch_size, stride=self.vit.patch_size),
            DoubleConv(c3, c3),
        )

        self.dec3 = DecoderBlock(c3, c3, c2)
        self.dec2 = DecoderBlock(c2, c2, c1)
        self.dec1 = DecoderBlock(c1, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #inp = 2, 1, 128, 128
        skip1, x = self.enc1(x) # out1: 2, 64, 64, 64; skip1: 2, 64, 128, 128
        skip2, x = self.enc2(x)  # out2: 2, 128, 32, 32; skip2: 2, 128, 64, 64
        skip3, x = self.enc3(x)  # out3: 2, 256, 16, 16; skip3: 2, 256, 32, 32

        x = self.vit(x) # after ViT: 2, 256, 4, 4
        x = self.vit_upsample(x) # after ViT Upsample: 2, 256, 16, 16
        if x.shape[-2:] != skip3.shape[-2:]:
            x = F.interpolate(x, size=skip3.shape[-2:], mode="bilinear", align_corners=False)

        x = self.dec3(x, skip3) # after dec3: 2, 128, 32, 32
        x = self.dec2(x, skip2) # after dec2: 2, 64, 64, 64
        x = self.dec1(x, skip1) # after dec1: 2, 64, 128, 128
        return self.head(x)


if __name__ == "__main__":
    model = TransUNet(
        in_channels=1,
        out_channels=2,
        image_size=128,
        vit_patch_size=4,
        vit_embed_dim=384,
        vit_depth=6,
        vit_num_heads=6,
    )
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    #print(y.shape)
