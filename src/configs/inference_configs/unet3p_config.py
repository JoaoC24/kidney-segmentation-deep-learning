from src.configs.config_main import ROOT_DIR


MODEL_NAME = "unet3p"
DATA_DIR = ROOT_DIR / "data" / "preprocessed_volumes"
NUM_WORKERS = 4
ACCELERATOR = "auto"
DEVICES = "auto"
PRECISION = "bf16-mixed"
SEED = 42
INPUT_CHANNELS = 1
NUM_CLASSES = 2
IMAGE_SIZE = 128
VOLUME_SLICE_AXIS = 0
TEST_SLICE_BATCH_SIZE = 32
SAVE_INFERENCE_VOLUMES = True


CHECKPOINT_PATH = ROOT_DIR / "ckpts" / "best" / "unet3p.ckpt"

METRICS_DIR = ROOT_DIR / "results" / "metrics" / "unet3p"
SUMMARIES_DIR = ROOT_DIR / "results" / "summaries" / "unet3p"
IMAGES_DIR = ROOT_DIR / "results" / "images" / "unet3p"
