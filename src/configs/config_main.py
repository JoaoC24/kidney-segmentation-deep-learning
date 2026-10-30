import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


COMET_PROJECT_NAME = "TKV_seg"
COMET_WORKSPACE = "joaoc24"
COMET_API_KEY = os.getenv("COMET_API_KEY", "")
COMET_ONLINE = True


DATA_DIR = ROOT_DIR / "data" / "preprocessed_slices"
VOLUME_DATA_DIR = ROOT_DIR / "data" / "preprocessed_volumes"
NUM_WORKERS = 4
PERSISTENT_WORKERS = False


ACCELERATOR = "auto"
DEVICES = "auto"
PRECISION = "bf16-mixed"


INPUT_CHANNELS = int(1)
NUM_CLASSES = int(2)
IMAGE_SIZE = int(128)
BATCH_SIZE = int(32)
MAX_EPOCHS = int(250)
MIN_EPOCHS = int(1)
MODEL_SAVE_DIRECTORY = ROOT_DIR / "ckpts"
SEED = int(42)
MONITOR_PARAM = "val_dice"
MONITOR_MODE = "max"


EARLY_STOPPING_PATIENCE = int(15)
EARLY_STOPPING_DELTA = float(0.0)




FORCE_PROCESS_EXIT_AFTER_TRAINING = True
