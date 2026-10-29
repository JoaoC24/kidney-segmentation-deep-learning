# Kidney Segmentation with Deep Learning

An end-to-end 2D deep learning workflow for segmenting kidney structures from
3D medical image volumes. The project includes paired-volume preprocessing,
slice-based model training, volume-level inference, and experiment/result
logging.

## Models

- U-Net
- U-Net++
- U-Net 3+
- TransUNet
- Swin-UNet
- CNN image classifier

The segmentation models are implemented in PyTorch and trained with PyTorch
Lightning. Training and validation metrics are logged through Comet ML.

## Setup

Use Python 3.10 or newer. Install PyTorch for your CPU or CUDA environment,
then install the remaining project dependencies:

```bash
python -m venv .venv
# Activate the virtual environment for your shell, then:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Training uses Comet ML by default. Set `COMET_API_KEY` in the environment
before starting training. To run without online Comet logging, set
`COMET_ONLINE = False` in `src/configs/config_main.py`.

## Data layout and preparation

Place each image and its corresponding label in alphabetically matching order
under the raw input folders. Inputs may be NIfTI files or DICOM series.
For DICOM, keep each 3D series together in its own subfolder under the
corresponding `Images/` or `Labels/` directory.

```text
data/raw/
├── Images/
└── Labels/
```

Run these commands from the repository root, in order:

```bash
python -m src.data.preprocessing
python -m src.data.volume_orientation
python -m src.data.data_split
python -m src.data.volume_to_slices
```

The scripts perform the following steps:

1. Read paired image and label volumes in alphabetical order. Apply masking,
   N4 bias correction, percentile clipping, and foreground normalization to
   each image. Preserve label values and check that paired array shapes match.
   Save matching numbered volumes in `data/interim/Images/` and `Labels/`.
2. Move each pair's longest array axis to axis 0 in place, applying the same
   axis move to the image and label.
3. Split the volumes into train and test subsets using a seeded 80/20 split.
   The volume pairs are saved under `data/preprocessed_volumes/`.
4. Convert each volume into 2D slices along axis 0 and save paired slices under
   `data/preprocessed_slices/`. Slice filenames retain the volume filename and
   append a four-digit, one-based slice index.

The split fraction and seed can be changed with `--test-fraction` and `--seed`
when running `src.data.data_split`. Input and output directories can also be
overridden with each script's command-line options; use `--help` for details.

## Training

Training datamodules read the 2D slices from `data/preprocessed_slices/`.
Configuration defaults are in `src/configs/config_main.py`; model-specific
training settings are in `src/configs/run_config.py`.

For example, train U-Net with:

```bash
python src/training/train_unet.py --config run_config
```

Other training scripts are available for U-Net++, U-Net 3+, TransUNet,
Swin-UNet, and the classifier:

```text
src/training/train_unet2p.py
src/training/train_unet3p.py
src/training/train_transunet.py
src/training/train_swinunet.py
src/training/train_class.py
```

Training saves Lightning checkpoints directly under `ckpts/` and logs to
Comet ML. The early-stopping and checkpoint callbacks monitor validation Dice
by default. Inference configs point under `ckpts/best/`; move the checkpoint
you want to evaluate there and update its configured filename.

## Volume inference and evaluation

Inference datamodules read the 3D test volumes from
`data/preprocessed_volumes/test/`. Set the checkpoint path and other options
in the matching file under `src/configs/inference_configs/`, then run a model's
inference script from the repository root. For example:

```bash
python src/inference/test_unet.py --config unet_config
```

Inference writes per-volume and summary CSV metrics under `results/metrics/`
and `results/summaries/`. Predicted volumes are written under
`results/images/` when enabled in the model's inference config.

## Project structure

```text
src/
├── configs/       # Shared, training, and inference configuration
├── data/          # Preprocessing, splitting, orientation, and datasets
├── inference/     # Volume evaluation and result export
├── models/        # Lightning modules and model architectures
└── training/      # Model training entry points
data/              # Local input and generated datasets (not committed)
ckpts/             # Model checkpoints (not committed)
results/           # Evaluation outputs (not committed)
```

## Evaluation note

The train/test split is performed by volume before slicing, which keeps a
volume's slices within one of those two subsets. The current training
datamodule then creates validation by randomly splitting the training slices.
As a result, slices from the same volume can appear in both training and
validation. Use volume-grouped validation before interpreting validation
metrics as an independent estimate.

## Data and checkpoints

Medical image data, generated arrays, checkpoints, Comet archives, and result
files are excluded from Git. Add only data that you are permitted to share.
