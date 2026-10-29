import argparse
import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import comet_ml  # import antes do torch/lightning
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger

import src.configs.config_main as config
from src.data.dataset import MyDataModule
from src.models.model_SWINUNET import LitSwinUNet
from src.models.train_utils import load_weights_from_lightning_checkpoint
from src.training.utils import (
    check_learning_rate,
    check_optimizer_name,
    check_class_weights,
    check_gamma,
    check_loss_name,
    check_loss_weight,
)

parser = argparse.ArgumentParser(description="Train swinunet using a named run configuration.")
parser.add_argument(
    "--config",
    default="run_config",
    help="Name of the run configuration module in src/configs (with or without .py).",
)
args = parser.parse_args()
config_name = args.config.removesuffix(".py")
run_config = importlib.import_module(f"src.configs.{config_name}")

MODEL_NAME = "swinunet"



class CometStatusCallback(Callback):
    def __init__(self, parameters: dict[str, object], run_timestamp: str) -> None:
        super().__init__()
        self.parameters = parameters
        self.run_timestamp = run_timestamp

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logger = trainer.logger
        if not isinstance(logger, CometLogger):
            return
        experiment = logger.experiment
        experiment.set_name(f"run-training-{MODEL_NAME}-{self.run_timestamp}")
        experiment.log_parameters(self.parameters)
        print(f"Comet experiment key: {experiment.get_key()}")
        print(f"Comet project: {config.COMET_WORKSPACE}/{config.COMET_PROJECT_NAME}")

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logger = trainer.logger
        if isinstance(logger, CometLogger):
            print("Running comet callback flush.")
            logger.experiment.flush()


def build_logger() -> CometLogger:
    online = config.COMET_ONLINE
    if online and not config.COMET_API_KEY:
        raise RuntimeError("COMET_API_KEY not set.")
    logger = CometLogger(
        api_key=config.COMET_API_KEY or None,
        project=config.COMET_PROJECT_NAME,
        workspace=config.COMET_WORKSPACE,
        online=online,
        offline_directory=str(config.ROOT_DIR / "logging"),
    )
    if online:
        print(f"Comet online logging to {config.COMET_WORKSPACE}/{config.COMET_PROJECT_NAME}.")
    else:
        print("Comet is logging offline.")
    return logger


def build_callbacks(run_timestamp: str, parameters: dict[str, object]) -> list[Callback]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.MODEL_SAVE_DIRECTORY),
        filename=f"{MODEL_NAME}-{run_timestamp}-{{epoch:03d}}-{{{config.MONITOR_PARAM}:.4f}}",
        monitor=config.MONITOR_PARAM,
        mode=config.MONITOR_MODE,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_stop_callback = EarlyStopping(
        monitor=config.MONITOR_PARAM,
        patience=config.EARLY_STOPPING_PATIENCE,
        mode=config.MONITOR_MODE,
        min_delta=config.EARLY_STOPPING_DELTA,
        verbose=True,
    )
    return [
        checkpoint_callback,
        early_stop_callback,
        LearningRateMonitor(logging_interval="epoch"),
        CometStatusCallback(parameters, run_timestamp),
    ]


def main() -> None:
    config.MODEL_SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for optimizer_candidate in run_config.OPT_FNS:
        optimizer_name = check_optimizer_name(optimizer_candidate)
        for learning_rate_candidate in run_config.LEARNING_RATES:
            learning_rate = check_learning_rate(learning_rate_candidate)
            for loss_candidate in run_config.LOSS_FNS:
                loss_name = check_loss_name(loss_candidate)
                if loss_name == "Focal_Dice":
                    loss_parameter_name = "gamma"
                    loss_parameter_values = run_config.FOCAL_GAMMAS
                else:
                    loss_parameter_name = "class_weights"
                    loss_parameter_values = run_config.CLASS_WEIGHT_OPTIONS

                for loss_parameter_candidate in loss_parameter_values:
                    if loss_parameter_name == "gamma":
                        loss_parameter = check_gamma(loss_parameter_candidate)
                        gamma = loss_parameter
                        class_weights = check_class_weights(run_config.CLASS_WEIGHTS)
                        ce_weight = None
                        dice_weight = None
                    else:
                        loss_parameter = check_class_weights(loss_parameter_candidate)
                        gamma = None
                        class_weights = loss_parameter
                        ce_weight = check_loss_weight(run_config.CE_WEIGHT, "CE")
                        dice_weight = check_loss_weight(run_config.DICE_WEIGHT, "Dice")

                    L.seed_everything(config.SEED, workers=True)
                    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                    parameters = {
                        "image_size": config.IMAGE_SIZE,
                        "batch_size": config.BATCH_SIZE,
                        "num_classes": config.NUM_CLASSES,
                        "learning_rate": learning_rate,
                        "weight_decay": run_config.WEIGHT_DECAY,
                        "loss": loss_name,
                        "loss_parameter_name": loss_parameter_name,
                        "loss_parameter": loss_parameter,
                        "optimizer": optimizer_name,
                    }
                    if loss_name == "Focal_Dice":
                        parameters["class_weights"] = class_weights
                    else:
                        if ce_weight is not None:
                            parameters["ce_weight"] = ce_weight
                        if dice_weight is not None:
                            parameters["dice_weight"] = dice_weight

                    datamodule = MyDataModule(
                        data_dir=config.DATA_DIR,
                        multiclass=config.NUM_CLASSES > 2,
                        batch_size=config.BATCH_SIZE,
                        image_size=config.IMAGE_SIZE,
                        num_workers=config.NUM_WORKERS,
                        seed=config.SEED,
                    )

                    model_parameters = {
                        "input_channels": config.INPUT_CHANNELS,
                        "num_classes": config.NUM_CLASSES,
                        "weight_decay": run_config.WEIGHT_DECAY,
                        "loss": loss_name,
                        "opt": optimizer_name,
                    }
                    if learning_rate is not None:
                        model_parameters["lr"] = learning_rate
                    if class_weights is not None:
                        model_parameters["class_weights"] = class_weights
                    if loss_name == "Focal_Dice":
                        if gamma is not None:
                            model_parameters["gamma"] = gamma
                    else:
                        if ce_weight is not None:
                            model_parameters["ce_weight"] = ce_weight
                        if dice_weight is not None:
                            model_parameters["dice_weight"] = dice_weight
                    model_parameters["image_size"] = config.IMAGE_SIZE
                    model = LitSwinUNet(**model_parameters)

                    if run_config.TRANSFER_CHECKPOINT_PATH is not None:
                        checkpoint_path = str(Path(run_config.TRANSFER_CHECKPOINT_PATH))
                        load_weights_from_lightning_checkpoint(model, checkpoint_path)

                    trainer = L.Trainer(
                        accelerator=config.ACCELERATOR,
                        devices=config.DEVICES,
                        min_epochs=config.MIN_EPOCHS,
                        max_epochs=config.MAX_EPOCHS,
                        precision=config.PRECISION,
                        callbacks=build_callbacks(run_timestamp, parameters),
                        logger=build_logger(),
                        default_root_dir=str(config.ROOT_DIR),
                        gradient_clip_val=1.0,
                        gradient_clip_algorithm="norm",
                        log_every_n_steps=10,
                    )
                    trainer.fit(model, datamodule=datamodule)


    if config.FORCE_PROCESS_EXIT_AFTER_TRAINING:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)



if __name__ == "__main__":
    main()
