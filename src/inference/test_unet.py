import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightning as L

from src.inference.config_utils import load_inference_config
from src.data.dataset import VolumeSegmentationDataModule
from src.models.model_UNET import LitUNet
from src.inference.test_utils import VolumeSegmentationTester
from src.models.train_utils import load_weights_from_lightning_checkpoint


MODEL_NAME = "unet"


if __name__ == "__main__":
    run_config = load_inference_config()
    if run_config.MODEL_NAME != MODEL_NAME:
        raise ValueError(f"Config for {run_config.MODEL_NAME} cannot be used by {MODEL_NAME}.")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = run_config.CHECKPOINT_PATH
    L.seed_everything(run_config.SEED, workers=True)

    datamodule = VolumeSegmentationDataModule(
        data_dir=run_config.DATA_DIR,
        multiclass=run_config.NUM_CLASSES > 2,
        image_size=run_config.IMAGE_SIZE,
        num_workers=run_config.NUM_WORKERS,
        slice_axis=run_config.VOLUME_SLICE_AXIS,
    )

    model = LitUNet(
        input_channels=run_config.INPUT_CHANNELS,
        num_classes=run_config.NUM_CLASSES,
    )
    load_weights_from_lightning_checkpoint(model, str(checkpoint_path))

    volume_metrics_csv_path = run_config.METRICS_DIR / f"volume_metrics_{run_timestamp}.csv"
    summary_csv_path = run_config.SUMMARIES_DIR / f"summary_metrics_{run_timestamp}.csv"
    inference_output_dir = run_config.IMAGES_DIR

    tester = VolumeSegmentationTester(
        model=model,
        model_name=MODEL_NAME,
        checkpoint_path=checkpoint_path,
        output_csv_path=volume_metrics_csv_path,
        summary_csv_path=summary_csv_path,
        slice_batch_size=run_config.TEST_SLICE_BATCH_SIZE,
        save_inference_volumes=run_config.SAVE_INFERENCE_VOLUMES,
        inference_output_dir=inference_output_dir,
    )

    trainer = L.Trainer(
        accelerator=run_config.ACCELERATOR,
        devices=run_config.DEVICES,
        precision=run_config.PRECISION,
        logger=False,
        default_root_dir=str(run_config.ROOT_DIR),
    )
    trainer.test(model=tester, datamodule=datamodule)

