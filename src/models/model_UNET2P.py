import torch
import lightning as L
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex, BinaryPrecision, BinaryRecall

from src.models.train_utils import DiceFocalLoss, CrossEntropyDiceLoss, init_weights_kaiming
from src.models.archs.UNET2P import UNET2P


class LitUNet2P(L.LightningModule):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        base_ch: int = 64,
        depth: int = 4,
        prune: int = 0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: tuple[float, float] = (0.25, 1.0),
        gamma: float | None = 2.0,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        deep_supervision_weights: tuple[float, ...] | None = None,
        inference_prune: int = 0,
        inference_average: bool = False,
        loss: str | None = "BCE_Dice",
        opt: str | None = "AdamW"
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("LitUNet2P currently supports binary segmentation with two output classes.")
        if inference_prune < 0 or inference_prune > depth:
            raise ValueError(f"inference_prune must be between 0 and {depth}, got {inference_prune}.")

        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.prune = prune
        self.inference_prune = inference_prune
        self.inference_average = inference_average
        self.opt = opt
        self.loss = loss
        self.num_classes = num_classes
        self.class_weights = class_weights

        self.network = UNET2P(
        in_channels=input_channels,
        out_channels=num_classes,
        base_ch=base_ch,
        depth=depth,
        prune=prune,
        )

        if self.loss == "Focal_Dice":
            self.loss_fn = DiceFocalLoss(
                gamma=gamma,
                class_weights=class_weights,
            )
            self.gamma = gamma
        else:
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

    def forward(self, x: torch.Tensor, prune: int | None = None) -> torch.Tensor | list[torch.Tensor]:
        return self.network(x, prune=prune)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self._inference_logits(images)
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
        outputs_list = self(images)
        logits = self._final_logits(outputs_list)

        loss_logits = self._average_logits(outputs_list)
        loss = self.loss_fn(loss_logits, masks)

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


    def _average_logits(self, outputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if not isinstance(outputs, list):
            return outputs

        stacked_outputs = torch.stack(outputs)
        weights = self._deep_supervision_weights(
        len(outputs),
        stacked_outputs.device,
        stacked_outputs.dtype,
        )
        return torch.sum(stacked_outputs * weights.view(-1, 1, 1, 1, 1), dim=0)

    def _deep_supervision_weights(
        self,
        num_outputs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        configured = self.hparams.deep_supervision_weights
        if configured is None:
            weights = torch.ones(num_outputs, device=device, dtype=dtype)
        else:
            if len(configured) != num_outputs:
                raise ValueError(
                    f"Expected {num_outputs} deep supervision weights, got {len(configured)}."
                )
            weights = torch.tensor(configured, device=device, dtype=dtype)

        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("Deep supervision weights must sum to a positive value.")
        return weights / weight_sum

    @staticmethod
    def _final_logits(outputs: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(outputs, list):
            return outputs[-1]
        return outputs

    def set_inference_mode(self, prune: int = 0, average: bool = False) -> None:
        if prune < 0 or prune > self.hparams.depth:
            raise ValueError(f"prune must be between 0 and {self.hparams.depth}, got {prune}.")
        self.inference_prune = prune
        self.inference_average = average

    def _inference_logits(self, images: torch.Tensor) -> torch.Tensor:
        if self.inference_average or self.inference_prune == 0:
            return self._average_logits(self(images, prune=0))
        return self(images, prune=self.inference_prune)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self._inference_logits(images)
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
            tags = ["UNet++", tag_classes, tag_opt, tag_loss]
            if self.prune == 0:
                tags.append("Deep_Supervision")
            experiment.add_tags(tags)

    def configure_optimizers(self):
        if self.opt == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
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
