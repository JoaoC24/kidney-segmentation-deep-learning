import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from pathlib import Path


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        class_weights: tuple[float, float] = (0.25, 1.0),
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        min_target = int(target.min().detach().cpu())
        max_target = int(target.max().detach().cpu())
        num_classes = logits.shape[1]
        if min_target < 0 or max_target >= num_classes:
            unique_values = torch.unique(target.detach().cpu()).tolist()
            raise ValueError(
                f"Target mask values must be in [0, {num_classes - 1}], but got values {unique_values}."
            )
        ce = F.cross_entropy(logits, target, weight=self.class_weights)

        probs = torch.softmax(logits, dim=1)[:, 1]
        foreground = (target == 1).float() # tirar possiveis valores != 1
        intersection = torch.sum(probs * foreground, dim=(1, 2))
        denominator = torch.sum(probs + foreground, dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.ce_weight * ce + self.dice_weight * dice_loss

class DiceFocalLoss(nn.Module):
    """
    Compound Loss: Dice Loss + Focal Loss for Medical Image Segmentation.
    Compatible with both Binary and Multiclass tasks.
    """
    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        gamma=2.0,
        smooth=1e-5,
        class_weights=None,
        include_background_in_dice: bool = False,
    ):
        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha          # Weight for Dice Loss
        self.beta = beta            # Weight for Focal Loss
        self.gamma = gamma          # Focal Loss focusing parameter
        self.smooth = smooth        # Avoid division by zero
        self.include_background_in_dice = include_background_in_dice
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, pred, target):
        """
        Args:
            pred (tensor): Model raw logits. Shape (B, C, H, W) or (B, C, D, H, W) for 3D.
            target (tensor): Ground truth masks. Shape (B, H, W) or (B, 1, H, W) with integer class IDs,
                             or one-hot encoded shape (B, C, H, W).
        """
        num_classes = pred.shape[1]
        class_weights = getattr(self, "class_weights", None)
        if class_weights is not None and class_weights.numel() != num_classes:
            raise ValueError(f"Expected {num_classes} class weights, got {class_weights.numel()}.")

        if target.ndim == pred.ndim and target.shape[1] == num_classes:
            target_one_hot = target.float()
            target_indices = torch.argmax(target_one_hot, dim=1)
        else:
            if target.ndim == pred.ndim and target.shape[1] == 1:
                target = target.squeeze(1)
            target_indices = target.long()
            min_target = int(target_indices.min().detach().cpu())
            max_target = int(target_indices.max().detach().cpu())
            if min_target < 0 or max_target >= num_classes:
                unique_values = torch.unique(target_indices.detach().cpu()).tolist()
                raise ValueError(
                    f"Target mask values must be in [0, {num_classes - 1}], but got values {unique_values}."
                )
            one_hot_dims = (0, -1, *range(1, target_indices.ndim))
            target_one_hot = F.one_hot(target_indices, num_classes=num_classes).permute(*one_hot_dims).float()

        prob = torch.softmax(pred, dim=1)
        prob_flat = prob.reshape(prob.shape[0], num_classes, -1)
        target_flat = target_one_hot.reshape(target_one_hot.shape[0], num_classes, -1)

        # ----------------------------------------
        # PART A: Multi-class DICE LOSS
        # ----------------------------------------
        dice_start_class = 0 if self.include_background_in_dice else 1
        dice_prob_flat = prob_flat[:, dice_start_class:]
        dice_target_flat = target_flat[:, dice_start_class:]

        intersection = torch.sum(dice_prob_flat * dice_target_flat, dim=-1)
        denominator = torch.sum(dice_prob_flat, dim=-1) + torch.sum(dice_target_flat, dim=-1)
        
        dice_per_class = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        
        # Apply class weights to Dice if provided
        if class_weights is not None:
            weights = class_weights[dice_start_class:].to(pred.device)
            weighted_dice = dice_per_class * weights.view(1, -1)
            dice_loss = weighted_dice.sum() / weights.sum().clamp_min(self.smooth) / dice_per_class.shape[0]
        else:
            dice_loss = torch.mean(dice_per_class)

        # ----------------------------------------
        # PART B: FOCAL LOSS
        # ----------------------------------------
        log_prob = F.log_softmax(pred, dim=1)
        target_log_prob = log_prob.gather(1, target_indices.unsqueeze(1)).squeeze(1)
        target_prob = target_log_prob.exp()
        focal_per_pixel = -torch.pow(1.0 - target_prob, self.gamma) * target_log_prob

        if class_weights is not None:
            pixel_weights = class_weights.to(pred.device).gather(0, target_indices.reshape(-1)).view_as(target_indices)
            focal_loss = (focal_per_pixel * pixel_weights).sum() / pixel_weights.sum().clamp_min(self.smooth)
        else:
            focal_loss = focal_per_pixel.mean()

        # ----------------------------------------
        # PART C: COMPOUND LOSS WEIGHTING
        # ----------------------------------------
        total_loss = (self.alpha * dice_loss) + (self.beta * focal_loss)
        
        return total_loss

def init_weights_kaiming(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if module.kernel_size == (1, 1) and module.out_channels <= 3: 
            # Use linear/identity gain because there's no ReLU after this layer
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
        else:
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def load_weights_from_lightning_checkpoint(
    model: LightningModule,
    checkpoint_path: str | Path,
    strict: bool = True,
) -> None:
    """Load the network weights from a Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    network_state_dict = {
        key.removeprefix("network."): value
        for key, value in state_dict.items()
        if key.startswith("network.")
    }
    if not network_state_dict:
        raise ValueError(
            f"No network weights found in checkpoint: {checkpoint_path}. "
            'Expected Lightning state_dict keys starting with "network.".'
        )
    model.network.load_state_dict(network_state_dict, strict=strict)
    print(f"Loaded network weights from checkpoint: {checkpoint_path}")
