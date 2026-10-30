import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:

    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
    batch: int,
) -> torch.Tensor:

    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, -1)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WindowAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, num_tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_windows, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention = attention + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attention = attention.view(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                num_tokens,
                num_tokens,
            )
            attention = attention + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.num_heads, num_tokens, num_tokens)

        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = attention @ v
        x = x.transpose(1, 2).reshape(batch_windows, num_tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        height, width = input_resolution
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = min(window_size, height, width)
        self.shift_size = 0 if min(input_resolution) <= window_size else shift_size
        if self.shift_size >= self.window_size:
            raise ValueError("shift_size must be smaller than window_size.")

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            self.window_size,
            num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.register_buffer("attention_mask", self._build_attention_mask(), persistent=False)

    def _build_attention_mask(self) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None

        height, width = self.input_resolution
        image_mask = torch.zeros((1, height, width, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                image_mask[:, h_slice, w_slice, :] = count
                count += 1

        mask_windows = window_partition(image_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(attention_mask != 0, -100.0)
        return attention_mask.masked_fill(attention_mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = x.shape
        if tokens != height * width:
            raise ValueError(f"Expected {height * width} tokens, got {tokens}.")

        shortcut = x
        x = self.norm1(x)
        x = x.view(batch, height, width, channels)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(x_windows, mask=self.attention_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, channels)
        shifted_x = window_reverse(attn_windows, self.window_size, height, width, batch)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(batch, height * width, channels)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 96,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected input size {(self.image_size, self.image_size)}, got {tuple(x.shape[-2:])}."
            )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = x.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("PatchMerging requires even spatial dimensions.")
        x = x.view(batch, height, width, channels)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(batch, -1, 4 * channels)
        return self.reduction(self.norm(x))


class PatchExpand(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int, out_dim: int | None = None):
        super().__init__()
        self.input_resolution = input_resolution
        out_dim = dim // 2 if out_dim is None else out_dim
        self.expand = nn.Linear(dim, 4 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch, tokens, _ = x.shape
        x = self.expand(x)
        x = x.view(batch, height, width, 2, 2, self.out_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, height * 2, width * 2, self.out_dim)
        x = x.view(batch, -1, self.out_dim)
        return self.norm(x)


class FinalPatchExpand(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int, scale: int = 4):
        super().__init__()
        self.input_resolution = input_resolution
        self.scale = scale
        self.expand = nn.Linear(dim, (scale**2) * dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = self.input_resolution
        batch, tokens, channels = x.shape
        x = self.expand(x)
        x = x.view(batch, height, width, self.scale, self.scale, channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, height * self.scale, width * self.scale, channels)
        x = x.view(batch, -1, channels)
        return self.norm(x)


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        downsample: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class BasicLayerUp(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        upsample: bool = True,
    ):
        super().__init__()
        self.upsample = PatchExpand(input_resolution, dim) if upsample else None
        block_dim = dim // 2 if upsample else dim
        block_resolution = (
            input_resolution[0] * 2,
            input_resolution[1] * 2,
        ) if upsample else input_resolution
        self.block_dim = block_dim
        self.block_resolution = block_resolution
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=block_dim,
                    input_resolution=block_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class SwinUNet(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 96,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        decoder_depths: tuple[int, int, int] = (2, 2, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        grid = image_size // patch_size
        if grid % 8 != 0:
            raise ValueError("image_size / patch_size must be divisible by 8.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.pos_drop = nn.Dropout(dropout)

        resolutions = [
            (grid, grid),
            (grid // 2, grid // 2),
            (grid // 4, grid // 4),
            (grid // 8, grid // 8),
        ]
        dims = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]

        self.encoder = nn.ModuleList(
            [
                BasicLayer(
                    dim=dims[i],
                    input_resolution=resolutions[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    downsample=i < 3,
                )
                for i in range(4)
            ]
        )

        self.skip_fuse3 = nn.Linear(dims[2] * 2, dims[2])
        self.skip_fuse2 = nn.Linear(dims[1] * 2, dims[1])
        self.skip_fuse1 = nn.Linear(dims[0] * 2, dims[0])

        self.decoder3 = BasicLayerUp(
            dim=dims[3],
            input_resolution=resolutions[3],
            depth=decoder_depths[0],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.decoder2 = BasicLayerUp(
            dim=dims[2],
            input_resolution=resolutions[2],
            depth=decoder_depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.decoder1 = BasicLayerUp(
            dim=dims[1],
            input_resolution=resolutions[1],
            depth=decoder_depths[2],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.final_expand = FinalPatchExpand(resolutions[0], dims[0], scale=patch_size)
        self.head = nn.Conv2d(dims[0], num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.pos_drop(self.patch_embed(x))

        skips = []
        for layer in self.encoder:
            x, skip = layer(x)
            skips.append(skip)

        x = self.decoder3(x)
        x = self.skip_fuse3(torch.cat([x, skips[2]], dim=-1))
        x = self.decoder2(x)
        x = self.skip_fuse2(torch.cat([x, skips[1]], dim=-1))
        x = self.decoder1(x)
        x = self.skip_fuse1(torch.cat([x, skips[0]], dim=-1))

        x = self.final_expand(x)
        x = x.transpose(1, 2).reshape(batch, -1, self.image_size, self.image_size)
        return self.head(x)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


if __name__ == "__main__":
    model = SwinUNet(
        image_size=128,
        patch_size=4,
        in_channels=1,
        num_classes=2,
        embed_dim=48,
        num_heads=(3, 6, 12, 24),
        window_size=4,
    )
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(y.shape)
