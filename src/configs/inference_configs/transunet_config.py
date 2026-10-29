from src.configs.config_main import ROOT_DIR


MODEL_NAME = "transunet_1L"
DATA_DIR = ROOT_DIR / "data" / "preprocessed"
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

# Edit this path to choose the checkpoint tested by test_transunet.py.
CHECKPOINT_PATH = ROOT_DIR / "ckpts" / "best" / "transunet_1L.ckpt"

METRICS_DIR = ROOT_DIR / "results" / "metrics" / "transunet"
SUMMARIES_DIR = ROOT_DIR / "results" / "summaries" / "transunet"
IMAGES_DIR = ROOT_DIR / "results" / "images" / "transunet"

TRANSUNET_VIT_DEPTH = 1
