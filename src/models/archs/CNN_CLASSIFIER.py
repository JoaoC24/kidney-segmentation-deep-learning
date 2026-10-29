import torch
import torch.nn as nn
from src.models.archs.utils.network_blocks import DoubleConv


class BinaryImageClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        widths = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        )

        self.features = nn.Sequential(
            DoubleConv(in_channels, widths[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(widths[0], widths[1]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(widths[1], widths[2], dropout=dropout * 0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(widths[2], widths[3], dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[3], widths[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(widths[2], 1),
        )

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x) >= threshold).long()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)


if __name__ == "__main__":
    model = BinaryImageClassifier(in_channels=1)
    x = torch.randn(2, 1, 128, 128)
    logits = model(x)
    preds = torch.sigmoid(logits)
    print(logits)
    print(preds)
    print((preds >= 0.5).int())
