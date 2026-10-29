import torch
import torch.nn as nn
from src.models.archs.utils.network_blocks import DoubleConv


class UNET2P(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_ch=64, depth=4, prune=0):
        """U-Net++ with optional early branch inference.

        prune=0 computes every decoder branch and returns a list of outputs.
        prune=1..depth computes only up to that branch and returns one output.
        """
        super().__init__()
        if depth < 1:
            raise ValueError("Parameter: depth must be at least 1.")
        if prune < 0 or prune > depth:
            raise ValueError(f"Parameter: prune must be between 0 and {depth}, got {prune}.")

        self.base_ch = base_ch
        self.depth = depth
        self.prune = prune
        self.nodes = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for i in range(depth + 1):
            layer = nn.ModuleList()
            out_ch = base_ch * 2**i
            for j in range(depth + 1 - i):
                if j == 0:
                    in_ch = in_channels if i == 0 else base_ch * 2 ** (i - 1)
                else:
                    in_ch = base_ch * 2**i * (j + 1)
                layer.append(DoubleConv(in_ch, out_ch))
            self.nodes.append(layer)

        self.upsamples = nn.ModuleList()
        self.upsamples.append(nn.ModuleList())
        for i in range(1, depth + 1):
            upsample_layer = nn.ModuleList()
            for _ in range(depth + 1 - i):
                upsample_layer.append(
                    nn.ConvTranspose2d(
                        base_ch * 2**i,
                        base_ch * 2 ** (i - 1),
                        kernel_size=2,
                        stride=2,
                    )
                )
            self.upsamples.append(upsample_layer)

        self.supervision = nn.ModuleList(
            [nn.Conv2d(base_ch, out_channels, kernel_size=1) for _ in range(depth)]
        )

    def forward(self, x, prune=None):
        prune = self.prune if prune is None else prune
        if prune < 0 or prune > self.depth:
            raise ValueError(f"Parameter: prune must be between 0 and {self.depth}, got {prune}.")

        last_branch = self.depth if prune == 0 else prune
        values = [[None] * (self.depth + 1) for _ in range(self.depth + 1)]

        inp = x
        for i in range(last_branch + 1):
            values[i][0] = self.nodes[i][0](inp)
            if i < last_branch:
                inp = self.pool(values[i][0])

        for j in range(1, last_branch + 1):
            for i in range(last_branch - j + 1):
                up = self.upsamples[i + 1][j - 1](values[i + 1][j - 1]) # vai buscar o valor do no abaixo, dá upsample
                cat = torch.cat([up, *[values[i][k] for k in range(j)]], dim=1)
                values[i][j] = self.nodes[i][j](cat)

        if prune == 0:
            return [self.supervision[m](values[0][m + 1]) for m in range(self.depth)]

        return self.supervision[prune - 1](values[0][prune])


if __name__ == "__main__":
    model = UNET2P(in_channels=1, out_channels=2, base_ch=64, depth=4, prune=0)
    device = "cpu"
    model.train()
    batch_size = 2
    in_channels = 1
    height = width = 128
    x = torch.randn(batch_size, in_channels, height, width)

    out = model(x)
    print(isinstance(out[0],torch.Tensor))
    #print(out[-1].shape)
    #print(model)
