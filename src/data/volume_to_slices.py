"""Convert train/test NumPy volumes into paired 2D slice files.

Run from the repository root with ``python -m src.data.volume_to_slices``.
Slices are taken along axis 0, the first axis after volume orientation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def convert_volume_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    """Save each image and label volume as matching 2D slice files."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    saved: list[Path] = []

    for subset in ("train", "test"):
        subset_dir = input_dir / subset
        image_dir, label_dir = subset_dir / "Images", subset_dir / "Labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Expected Images/ and Labels/ under: {subset_dir}")

        image_paths = sorted(image_dir.glob("*.npy"), key=lambda path: path.name.casefold())
        label_paths = sorted(label_dir.glob("*.npy"), key=lambda path: path.name.casefold())
        if len(image_paths) != len(label_paths):
            raise ValueError(
                f"{subset} has {len(image_paths)} image volumes and "
                f"{len(label_paths)} label volumes."
            )

        destination_images = output_dir / subset / "Images"
        destination_labels = output_dir / subset / "Labels"
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)

        for index, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
            if image_path.stem != label_path.stem:
                raise ValueError(
                    f"{subset} pair {index} has different filenames: "
                    f"{image_path.name} != {label_path.name}."
                )
            image = np.load(image_path)
            label = np.load(label_path)
            if image.ndim != 3 or label.ndim != 3:
                raise ValueError(
                    f"{subset} pair {index} must contain 3D volumes, "
                    f"got shapes {image.shape} and {label.shape}."
                )
            if image.shape != label.shape:
                raise ValueError(
                    f"{subset} pair {index} has mismatched shapes: "
                    f"{image.shape} != {label.shape}."
                )

            for slice_index in range(image.shape[0]):
                filename = f"{image_path.stem}_{slice_index + 1:04d}.npy"
                image_output = destination_images / filename
                label_output = destination_labels / filename
                np.save(image_output, image[slice_index])
                np.save(label_output, label[slice_index])
                saved.extend((image_output, label_output))
            print(f"Converted {subset}/{image_path.name}: {image.shape[0]} slices")

    return saved


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=repo_root / "data" / "preprocessed_volumes"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "data" / "preprocessed_slices"
    )
    args = parser.parse_args()
    convert_volume_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
