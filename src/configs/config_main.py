import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Comet
COMET_PROJECT_NAME = "TKV_seg"
COMET_WORKSPACE = "joaoc24"
COMET_API_KEY = os.getenv("COMET_API_KEY", "")
COMET_ONLINE = True

# Dataset
DATA_DIR = ROOT_DIR / "data" / "preprocessed"
NUM_WORKERS = 4
PERSISTENT_WORKERS = False

# Compute
ACCELERATOR = "auto"
DEVICES = "auto"
PRECISION = "bf16-mixed"

# Training
INPUT_CHANNELS = int(1)
NUM_CLASSES = int(2)
IMAGE_SIZE = int(128)
BATCH_SIZE = int(32)
MAX_EPOCHS = int(250)
MIN_EPOCHS = int(1)
MODEL_SAVE_DIRECTORY = Path(ROOT_DIR / "ckpts" / "lr_experiment" / "1e-4")
SEED = int(42)
MONITOR_PARAM = "val_dice"
MONITOR_MODE = "max"

# Early stopping
EARLY_STOPPING_PATIENCE = int(15)
EARLY_STOPPING_DELTA = float(0.0)



# Process shutdown
FORCE_PROCESS_EXIT_AFTER_TRAINING = True
