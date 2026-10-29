
# Run-specific optimizer settings
OPT_FN = "AdamW"
LEARNING_RATE = float(1e-4)
WEIGHT_DECAY = float(1e-4)

# Candidate values used by train_unet.py's grid search.
OPT_FNS = [OPT_FN]
LEARNING_RATES = [LEARNING_RATE]

# Run-specific loss settings
LOSS_FN = "BCE_Dice"
CLASS_WEIGHTS = (0.05, 1.0)
GAMMA = float(2.0)
CE_WEIGHT = float(1.0)
DICE_WEIGHT = float(1.0)

LOSS_FNS = [LOSS_FN]
CLASS_WEIGHT_OPTIONS = [CLASS_WEIGHTS]
FOCAL_GAMMAS = [GAMMA]

TRANSFER_CHECKPOINT_PATH = None

# TransUNet architecture setting
VIT_DEPTH = 6
VIT_DEPTHS = [VIT_DEPTH]
