import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        embed_dim: int,
        patch_size: int | None = None,
        num_patches: int | None = None,
    ):
        super().__init__()
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if patch_size is None and num_patches is None:
            raise ValueError("Either patch_size or num_patches must be provided.")
        if patch_size is not None and num_patches is not None:
            raise ValueError("Provide only one of patch_size or num_patches.")

        if num_patches is not None:
            patch_grid = int(math.sqrt(num_patches))
            if patch_grid * patch_grid != num_patches:
                raise ValueError("num_patches must be a perfect square for square images.")
            if image_size % patch_grid != 0:
                raise ValueError("image_size must be divisible by sqrt(num_patches).")
            patch_size = image_size // patch_grid
        else:
            if image_size % patch_size != 0:
                raise ValueError("image_size must be divisible by patch_size.")
            patch_grid = image_size // patch_size
            num_patches = patch_grid * patch_grid

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_grid = patch_grid
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}.")
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"Expected square input size {(self.image_size, self.image_size)}, got {tuple(x.shape[-2:])}.")

        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch_size, num_tokens, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        x = attention @ v
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attention
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attention_dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if return_attention:
            attn_out, attention = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop1(attn_out)
            x = x + self.mlp(self.norm2(x))
            return x, attention

        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerFeatureMap(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 128,
        patch_size: int | None = 16,
        num_patches: int | None = None,
        embed_dim: int = 256,
        depth: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        out_channels: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            image_size=image_size,
            embed_dim=embed_dim,
            patch_size=patch_size,
            num_patches=num_patches,
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        out_channels = embed_dim if out_channels is None else out_channels
        self.out_channels = out_channels
        self.output_projection = (
            nn.Identity()
            if out_channels == embed_dim
            else nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        )

        self._init_weights()

    @property
    def patch_size(self) -> int:
        return self.patch_embed.patch_size

    @property
    def patch_grid(self) -> int:
        return self.patch_embed.patch_grid

    @property
    def num_patches(self) -> int:
        return self.patch_embed.num_patches

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_tokens: bool = False,
        return_attention: bool = False,
    ):
        tokens = self.patch_embed(x)
        tokens = self.pos_drop(tokens + self.pos_embedding)

        attentions = []
        for block in self.blocks:
            if return_attention:
                tokens, attention = block(tokens, return_attention=True)
                attentions.append(attention)
            else:
                tokens = block(tokens)

        tokens = self.norm(tokens)
        feature_map = self.tokens_to_feature_map(tokens)
        feature_map = self.output_projection(feature_map)

        outputs = {"feature_map": feature_map}
        if return_tokens:
            outputs["tokens"] = tokens
        if return_attention:
            outputs["attention"] = attentions

        if len(outputs) == 1:
            return feature_map
        return outputs

    def tokens_to_feature_map(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = tokens.shape
        if num_tokens != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} tokens, got {num_tokens}.")

        grid = self.patch_grid
        return tokens.transpose(1, 2).reshape(batch_size, embed_dim, grid, grid)


if __name__ == "__main__":
    model = VisionTransformerFeatureMap(
        in_channels=1,
        image_size=128,

        embed_dim=256,
        depth=1,
        num_heads=8,

    )
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(y.shape)
