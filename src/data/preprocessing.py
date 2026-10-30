"""Preprocess paired 3D image and label volumes into NumPy arrays.

Run from the repository root with ``python -m src.data.preprocessing``.
Alphabetically paired volumes in ``data/raw/Images`` and ``data/raw/Labels``
are saved under matching indexed filenames in ``data/interim``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects


NIFTI_SUFFIXES = (".nii", ".nii.gz")
DICOM_SUFFIXES = {".dcm", ".dicom"}


def image_masking(image: np.ndarray) -> np.ndarray:
    """Build a foreground mask from a 3D image using Otsu and components."""
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {image.shape}.")
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("Image contains no finite voxel values.")


    threshold = threshold_otsu(image[finite])
    mask = (image > threshold) & finite
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask.astype(bool), min_size=1000)
    labels = label(mask)
    counts = np.bincount(labels.ravel())
    if counts.size <= 1 or counts[1:].max(initial=0) == 0:
        raise ValueError("Could not find a foreground component of at least 1000 voxels.")
    largest_label = int(np.argmax(counts[1:]) + 1)
    return labels == largest_label


def n4_bias_correction(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply SimpleITK N4 bias correction, estimating at half resolution."""
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.uint8)
    if image.ndim != 3 or mask.shape != image.shape:
        raise ValueError(f"N4 expects matching 3D image and mask, got {image.shape} and {mask.shape}.")
    if not mask.any():
        raise ValueError("N4 correction requires a non-empty mask.")

    raw_image = sitk.GetImageFromArray(image)
    mask_image = sitk.GetImageFromArray(mask)
    shrink_factor = 2
    input_image = sitk.Shrink(raw_image, [shrink_factor] * 3)
    input_mask = sitk.Shrink(mask_image, [shrink_factor] * 3)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetConvergenceThreshold(1e-6)
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrector.Execute(input_image, input_mask)

    log_bias_field = corrector.GetLogBiasFieldAsImage(raw_image)
    corrected = raw_image / sitk.Exp(log_bias_field)
    return sitk.GetArrayFromImage(corrected)


def percentile_clip(
    data: np.ndarray,
    p1: float = 1,
    p2: float = 99,
    adjust: bool = False,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Clip intensities using foreground percentiles when a mask is supplied."""
    data = np.asarray(data)
    if not 0 <= p1 < p2 <= 100:
        raise ValueError(f"Expected 0 <= p1 < p2 <= 100, got {p1} and {p2}.")
    selected = np.isfinite(data)
    if mask is not None:
        if np.shape(mask) != data.shape:
            raise ValueError("Mask and image must have the same shape.")
        selected &= np.asarray(mask, dtype=bool)
    values = data[selected]
    if values.size == 0:
        raise ValueError("No finite foreground values available for percentile clipping.")

    lower, upper = np.percentile(values, [p1, p2])
    clipped = np.clip(data, lower, upper)
    if adjust:
        clipped = clipped - lower
    return clipped


def image_norm(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score normalize foreground voxels and keep background at zero."""
    image = np.asarray(image, dtype=np.float32)
    foreground = np.asarray(mask, dtype=bool)
    if image.ndim != 3 or foreground.shape != image.shape:
        raise ValueError("Normalization expects matching 3D image and mask arrays.")
    values = image[foreground]
    if values.size == 0:
        raise ValueError("Cannot normalize an empty foreground mask.")
    mean = values.mean()
    std = values.std()
    normalized = np.zeros_like(image, dtype=np.float32)
    normalized[foreground] = (image[foreground] - mean) / (std + 1e-8)
    return normalized


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Mask, N4-correct, foreground-clip, and z-score a 3D volume."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {image.shape}.")
    if not np.isfinite(image).all():
        image = np.nan_to_num(image, copy=True)

    mask = image_masking(image)
    corrected = n4_bias_correction(image, mask)
    clipped = percentile_clip(corrected, p1=1, p2=99, adjust=False, mask=mask)
    return image_norm(clipped, mask)


def _read_nifti(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _discover_volumes(directory: Path) -> list[tuple[str, list[Path]]]:
    """Discover NIfTI files and DICOM series within one modality directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Required input directory not found: {directory}")

    volumes: list[tuple[str, list[Path]]] = []
    nifti_paths = [
        path for path in directory.rglob("*")
        if path.is_file() and path.name.lower().endswith(NIFTI_SUFFIXES)
    ]
    volumes.extend(
        (path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem, [path])
        for path in nifti_paths
    )

    claimed: set[Path] = set()
    folders = [directory, *(path for path in directory.rglob("*") if path.is_dir())]
    for folder in folders:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or []
        for series_id in series_ids:
            files = [Path(name) for name in sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(folder), series_id)]
            files = [path for path in files if path not in claimed]
            if files:
                volumes.append((files[0].parent.name or series_id, files))
                claimed.update(files)

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in DICOM_SUFFIXES and path not in claimed:
            volumes.append((path.stem, [path]))
    return sorted(volumes, key=lambda item: item[0].casefold())


def _read_volume(files: list[Path]) -> np.ndarray:
    if len(files) == 1 and files[0].name.lower().endswith(NIFTI_SUFFIXES):
        return _read_nifti(files[0])
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in files])
    return sitk.GetArrayFromImage(reader.Execute())


def preprocess_directory(raw_dir: Path, output_dir: Path) -> list[Path]:
    """Preprocess sorted image/label pairs and save matching ``.npy`` files."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    images = _discover_volumes(raw_dir / "Images")
    labels = _discover_volumes(raw_dir / "Labels")
    if len(images) != len(labels):
        raise ValueError(
            f"Found {len(images)} image volumes and {len(labels)} label volumes; "
            "alphabetical index pairing requires equal counts."
        )

    image_dir, label_dir = output_dir / "Images", output_dir / "Labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, ((image_name, image_files), (label_name, label_files)) in enumerate(zip(images, labels)):
        image = _read_volume(image_files)
        label_array = _read_volume(label_files)
        if image.shape != label_array.shape:
            raise ValueError(
                f"Paired volumes at index {index} have different shapes: "
                f"{image_name} {image.shape} vs {label_name} {label_array.shape}."
            )
        processed_image = preprocess_image(image)

        image_path = image_dir / f"{index:04d}.npy"
        label_path = label_dir / f"{index:04d}.npy"
        np.save(image_path, processed_image)
        np.save(label_path, label_array)
        saved.extend((image_path, label_path))
        print(f"Saved pair {index}: {image_name} + {label_name} -> {image_path.name}")
    return saved


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=repo_root / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "interim")
    args = parser.parse_args()
    preprocess_directory(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
