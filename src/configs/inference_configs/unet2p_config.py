from src.configs.config_main import ROOT_DIR


MODEL_NAME = "unet2p"
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


CHECKPOINT_PATH = ROOT_DIR / "ckpts" / "best" / "unet2p.ckpt"

METRICS_DIR = ROOT_DIR / "results" / "metrics" / "unet2p"
SUMMARIES_DIR = ROOT_DIR / "results" / "summaries" / "unet2p"
IMAGES_DIR = ROOT_DIR / "results" / "images" / "unet2p"

UNET2P_DEPTH = 4
UNET2P_INFERENCE_PRUNE = UNET2P_DEPTH
UNET2P_INFERENCE_AVERAGE = False
UNET2P_DEEP_SUPERVISION_WEIGHTS = (1, 2, 4, 8)
