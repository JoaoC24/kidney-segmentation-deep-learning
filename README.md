# Kidney Segmentation with Deep Learning

An end-to-end deep learning workflow for segmenting kidney structures from 3D medical image volumes. The project includes input volume preprocessing, slice-based model training, volume-level inference/evaluation, and experiment/result logging.

The best model achieved an average Dice score of 90.2% on the test data, and an inference time of 0.55 ms per MRI slice.
Result discussions are cut short given existing non-disclosure policies.

Below is presented an example of the segmentation output from all models for a given image.

![Segmentation result example](figs/img.jpg?raw=true "Title")

## Models (implemented based on:)

- U-Net
- U-Net++
- U-Net3+
- TransUNet
- Swin-UNet
- CNN image classifier

The segmentation models are implemented in PyTorch and trained with PyTorch Lightning. Training and validation metrics are logged through Comet ML.

## Setup

Requires Python 3.10 or newer, PyTorch, and other dependencies that can be installed by running

```bash
python -m pip install -r requirements.txt
```

inside the virtual environment.

Training uses Comet ML by default. The cloud logging is dependent on `COMET_API_KEY`, which was redacted from the code before the push. To run without online Comet logging, `COMET_ONLINE = False` in `src/configs/config_main.py`.

## Project structure

```text
src/
├── configs/       # Shared, training, and inference configs and parameters
├── data/          # Preprocessing, splitting, orientation, and datasets
├── inference/     # Model evaluation, volume inference and result export
├── models/        # Lightning modules and model architectures
└── training/      # Model training entry points
data/              # Local input and generated datasets (not committed)
ckpts/             # Model checkpoints (not committed)
results/           # Evaluation outputs (not committed)
```

## Data layout and preparation

Data files were not pushed to the repository. The raw data consisted in paired Image and Label files (either DICOM or NIfTI format) obtained from 3 publicly available datasets containing MRI scan results. The preprocessing pipeline converts the format to numpy files (.np) for optimization reasons, and runs a series of operations:

```text
-body mask extraction
-N4 bias field correction
-percentile clipping
-foreground intensity normalization
-standard volume reorientation
-volume slicing
```

The resulting dataset was splitted into training (+ validation) and testing with a 80/20 ratio.

The execution was achieved by running these commands from the repository root, in order:

```bash
python -m src.data.preprocessing
python -m src.data.volume_orientation
python -m src.data.data_split
python -m src.data.volume_to_slices
```

The split fraction and seed can be changed by the arguments `--test-fraction` and `--seed` when running `src.data.data_split`.

## Training

Training datamodules load the 2D slices from `data/preprocessed_slices/`. Configuration defaults are stored in `src/configs/config_main.py`; model-specific training settings are saved in additional config files that are passed as arguments when running the training scripts. These files were redactes from the repository, but contain hyperparameter search spaces customized for each model architecture.

For example, train U-Net with:

```bash
python src/training/train_unet.py --config hyperparameter_search_config
```

Other training scripts are available for U-Net++, U-Net 3+, TransUNet, Swin-UNet, and the classifier:

Training saves Lightning checkpoints directly under `ckpts/` and logs to Comet ML. Early-stopping and checkpoint callbacks were implemented to control training sequences.

## Volume inference and evaluation

Inference datamodules read the 3D test volumes from `data/preprocessed_volumes/test/`. Set the checkpoint path and other options in the matching file under `src/configs/inference_configs/`, then run a model's inference script from the repository root. For example:

```bash
python src/inference/test_unet.py --config unet_config
```

where unet_config.py contains the necessary model initialization parameters.

Inference writes per-volume and summary CSV metrics under `results/metrics/` and `results/summaries/`. Predicted volumes are saved under `results/images/` when enabled in the model's inference config.

## Note

Medical image data, generated arrays, checkpoints, Comet archives, and result files are ignored by Git.
