"""Move each paired volume's longest axis to axis 0, in place.

Run from the repository root with ``python -m src.data.volume_orientation``.
Image and label files are paired by their alphabetically sorted index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def orient_volume(volume: np.ndarray) -> np.ndarray:
    """Return a 3D volume with its longest axis moved to the first position."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")
    longest_axis = int(np.argmax(volume.shape))
    return np.moveaxis(volume, longest_axis, 0)


def orient_directory(directory: Path) -> list[Path]:
    """Reorient image/label pairs in ``Images`` and ``Labels`` in place."""
    directory = Path(directory)
    image_dir, label_dir = directory / "Images", directory / "Labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected Images/ and Labels/ directories under: {directory}")
    image_paths = sorted(image_dir.glob("*.npy"), key=lambda path: path.name.casefold())
    label_paths = sorted(label_dir.glob("*.npy"), key=lambda path: path.name.casefold())
    if len(image_paths) != len(label_paths):
        raise ValueError(
            f"Found {len(image_paths)} image arrays and {len(label_paths)} label arrays; "
            "alphabetical index pairing requires equal counts."
        )

    updated: list[Path] = []
    for index, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        image = np.load(image_path)
        label = np.load(label_path)
        if image.ndim != 3 or label.ndim != 3:
            raise ValueError(
                f"Pair {index} must contain 3D arrays, got {image.shape} and {label.shape}."
            )
        if image.shape != label.shape:
            raise ValueError(
                f"Paired arrays at index {index} have different shapes: "
                f"{image_path.name} {image.shape} vs {label_path.name} {label.shape}."
            )
        longest_axis = int(np.argmax(image.shape))
        oriented_image = np.moveaxis(image, longest_axis, 0)
        oriented_label = np.moveaxis(label, longest_axis, 0)
        np.save(image_path, oriented_image)
        np.save(label_path, oriented_label)
        updated.extend((image_path, label_path))
        print(f"Updated pair {index}: {image.shape} -> {oriented_image.shape}")
    return updated


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "interim",
        help="Directory containing Images/ and Labels/ NumPy volumes.",
    )
    args = parser.parse_args()
    orient_directory(args.data_dir)


if __name__ == "__main__":
    main()
