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

import comet_ml
import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger

import src.configs.config_main as config
from src.data.dataset import MyClassificationDataModule
from src.models.model_CLASS import LitClassifier
from src.models.train_utils import load_weights_from_lightning_checkpoint
from src.training.utils import (
    check_learning_rate,
    check_optimizer_name,
    check_class_weights,
)

parser = argparse.ArgumentParser(description="Train classifier using a named run configuration.")
parser.add_argument(
    "--config",
    default="run_config",
    help="Name of the run configuration module in src/configs (with or without .py).",
)
args = parser.parse_args()
config_name = args.config.removesuffix(".py")
run_config = importlib.import_module(f"src.configs.{config_name}")

MODEL_NAME = "classifier"
MONITOR_PARAM = "val_F1"



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
        filename=f"{MODEL_NAME}-{run_timestamp}-{{epoch:03d}}-{{val_F1:.4f}}",
        monitor=MONITOR_PARAM,
        mode=config.MONITOR_MODE,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_stop_callback = EarlyStopping(
        monitor=MONITOR_PARAM,
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
            L.seed_everything(config.SEED, workers=True)
            class_weights = check_class_weights(run_config.CLASS_WEIGHTS)

            run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            parameters = {
                "task": "classification",
                "image_size": config.IMAGE_SIZE,
                "batch_size": config.BATCH_SIZE,
                "input_channels": config.INPUT_CHANNELS,
                "learning_rate": learning_rate,
                "weight_decay": run_config.WEIGHT_DECAY,
                "optimizer": optimizer_name,
            }
            if class_weights is not None:
                parameters["class_weights"] = class_weights

            datamodule = MyClassificationDataModule(
                data_dir=config.DATA_DIR,
                batch_size=config.BATCH_SIZE,
                image_size=config.IMAGE_SIZE,
                num_workers=config.NUM_WORKERS,
                seed=config.SEED,
            )

            model_parameters = {
                "in_channels": config.INPUT_CHANNELS,
                "weight_decay": run_config.WEIGHT_DECAY,
                "opt": optimizer_name,
            }
            if learning_rate is not None:
                model_parameters["lr"] = learning_rate
            if class_weights is not None:
                model_parameters["class_weights"] = class_weights
            model = LitClassifier(**model_parameters)

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
