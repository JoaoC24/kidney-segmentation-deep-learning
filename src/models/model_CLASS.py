import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryAUROC

from src.models.archs.CNN_CLASSIFIER import BinaryImageClassifier


class LitClassifier(L.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: tuple[float, float] = (0.25, 1.0),
        opt: str | None = "AdamW"
    ):
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.opt = opt
        self.class_weights = class_weights
                
        self.network = BinaryImageClassifier(
        in_channels=in_channels,
        base_channels=base_channels
        )

        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        self.train_F1 = BinaryF1Score()
        self.train_acc = BinaryAccuracy()
        self.val_F1 = BinaryF1Score()
        self.val_acc = BinaryAccuracy()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_auroc = BinaryAUROC()
        self.test_F1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_auroc = BinaryAUROC()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")
        
    def test_step(self, batch, batch_idx):
        images, classes = batch
        targets = classes.long()
        loss_targets = classes.float()
        logits = self(images)
        loss = self.loss_fn(logits, loss_targets)
        probs = torch.sigmoid(logits).float() # dá a probabilidade entre 0 e 1 de pertencer à Class 1

        self.test_F1.update(probs, targets)
        self.test_acc.update(probs, targets)
        self.test_precision.update(probs, targets)
        self.test_recall.update(probs, targets)
        self.test_auroc.update(probs, targets)
        self.log_dict(
        {
                "test_loss": loss,
                "test_F1": self.test_F1,
                "test_acc": self.test_acc,
                "test_precision": self.test_precision,
                "test_recall": self.test_recall,
                "test_auroc": self.test_auroc,
        },
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        logger=True,
        batch_size=images.size(0),
        )
        return loss

    def _shared_step(self, batch, stage: str):
        images, classes = batch
        targets = classes.long()
        loss_targets = classes.float()
        logits = self(images)

        loss = self.loss_fn(logits, loss_targets)

        probs = torch.sigmoid(logits).float()

        F1_metric = getattr(self, f"{stage}_F1")
        acc_metric = getattr(self, f"{stage}_acc")
        F1_metric.update(probs, targets)
        acc_metric.update(probs, targets)

        metrics = {f"{stage}_loss": loss, f"{stage}_F1": F1_metric, f"{stage}_acc": acc_metric}
        if stage == "val":
                self.val_precision.update(probs, targets)
                self.val_recall.update(probs, targets)
                self.val_auroc.update(probs, targets)
                metrics["val_precision"] = self.val_precision
                metrics["val_recall"] = self.val_recall
                metrics["val_auroc"] = self.val_auroc

        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=images.size(0))
        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        return (torch.sigmoid(self(images)) >= 0.5).long()
    
    def on_train_start(self):
        experiment = self.logger.experiment if self.logger else None
        if experiment and hasattr(experiment, "add_tags"):
            tag_opt = self.opt if self.opt == "SGD" else "AdamW"
            tags = ["Classifier", tag_opt, "BCE"] # compor estas strings todas
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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( # rever o uso de scheduler
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        )
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_F1", # monitor do scheduler: 5 epochs de paciencia e divide por 2 o lr
                "interval": "epoch",
                "frequency": 1,
        },
        }
