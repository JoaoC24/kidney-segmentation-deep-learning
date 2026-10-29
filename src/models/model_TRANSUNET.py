import torch
import lightning as L
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall

from src.models.train_utils import DiceFocalLoss, CrossEntropyDiceLoss, init_weights_kaiming
from src.models.archs.TRANSUNET import TransUNet


class LitTransUNet(L.LightningModule):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        image_size: int = 128,
        encoder_channels: tuple[int, int, int] = (64, 128, 256),
        vit_patch_size: int | None = 4,
        vit_num_patches: int | None = None,
        vit_embed_dim: int = 384,
        vit_depth: int = 1,
        vit_num_heads: int = 6,
        vit_mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: tuple[float, float] = (0.25, 1.0),
        gamma: float | None = 2.0,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        loss: str | None = "BCE_Dice",
        opt: str | None = "AdamW"
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("LitTransUNet currently supports binary segmentation with two output classes.")

        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt = opt
        self.loss = loss
        self.num_classes = num_classes
        self.class_weights = class_weights

        self.network = TransUNet(
        in_channels=input_channels,
        out_channels=num_classes,
        image_size=image_size,
        encoder_channels=encoder_channels,
        vit_patch_size=vit_patch_size,
        vit_num_patches=vit_num_patches,
        vit_embed_dim=vit_embed_dim,
        vit_depth=vit_depth,
        vit_num_heads=vit_num_heads,
        vit_mlp_ratio=vit_mlp_ratio,
        dropout=dropout,
        attention_dropout=attention_dropout,
        )

        if self.loss == "Focal_Dice":
            self.loss_fn = DiceFocalLoss(
                gamma=gamma,
                class_weights=class_weights,
            )
            self.gamma = gamma
        else: # atençao ao binary ou nao binary CE
            self.loss_fn = CrossEntropyDiceLoss(
                class_weights=class_weights,
                ce_weight=ce_weight,
                dice_weight=dice_weight,
            )
            self.ce_weight = ce_weight
            self.dice_weight = dice_weight

        self.train_dice = BinaryF1Score()
        self.train_iou = BinaryJaccardIndex()
        self.val_dice = BinaryF1Score()
        self.val_iou = BinaryJaccardIndex()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.test_dice = BinaryF1Score()
        self.test_iou = BinaryJaccardIndex()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()

        self.network.apply(init_weights_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)

        self.test_dice.update(preds, masks)
        self.test_iou.update(preds, masks)
        self.test_precision.update(preds, masks)
        self.test_recall.update(preds, masks)
        self.log_dict(
        {
                "test_loss": loss,
                "test_dice": self.test_dice,
                "test_iou": self.test_iou,
                "test_precision": self.test_precision,
                "test_recall": self.test_recall,
        },
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        logger=True,
        batch_size=images.size(0),
        )
        return loss

    def _shared_step(self, batch, stage: str):
        images, masks = batch
        logits = self(images)

        loss = self.loss_fn(logits, masks)

        preds = torch.argmax(logits, dim=1)

        dice_metric = getattr(self, f"{stage}_dice")
        iou_metric = getattr(self, f"{stage}_iou")
        dice_metric.update(preds, masks)
        iou_metric.update(preds, masks)

        metrics = {
                f"{stage}_loss": loss,
                f"{stage}_dice": dice_metric,
                f"{stage}_iou": iou_metric,
        }
        if stage == "val":
                self.val_precision.update(preds, masks)
                self.val_recall.update(preds, masks)
                metrics["val_precision"] = self.val_precision
                metrics["val_recall"] = self.val_recall

        self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=images.size(0),
        )
        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self(images)
        return {
        "logits": logits,
        "probabilities": torch.softmax(logits, dim=1),
        "masks": torch.argmax(logits, dim=1),
        }

    def on_train_start(self):
        experiment = self.logger.experiment if self.logger else None
        if experiment and hasattr(experiment, "add_tags"):
            tag_opt = self.opt if self.opt == "SGD" else "AdamW"
            tag_loss = self.loss if self.loss == "Focal_Dice" else "BCE_Dice"
            tag_classes = "Binary" if self.num_classes < 3 else "Multiclass"
            tags = ["TransUNet", tag_classes, tag_opt, tag_loss]
            experiment.add_tags(tags)

    def configure_optimizers(self):
        if self.opt == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,  # Make sure this is tuned higher for SGD!
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        )
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_dice",
                "interval": "epoch",
                "frequency": 1,
        },
        }
