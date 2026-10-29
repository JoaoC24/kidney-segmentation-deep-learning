"""Split paired NumPy volumes from data/interim into train and test folders.

Run from the repository root with ``python -m src.data.data_split``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def split_data(
    input_dir: Path,
    output_dir: Path,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Copy paired arrays into deterministic train/test subsets by file index."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}.")

    image_dir, label_dir = input_dir / "Images", input_dir / "Labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected Images/ and Labels/ directories under: {input_dir}")
    images = sorted(image_dir.glob("*.npy"), key=lambda path: path.name.casefold())
    labels = sorted(label_dir.glob("*.npy"), key=lambda path: path.name.casefold())
    if not images:
        raise FileNotFoundError(f"No .npy image files found in: {image_dir}")
    if len(images) != len(labels):
        raise ValueError(f"Found {len(images)} images and {len(labels)} labels.")

    pairs = list(zip(images, labels))
    for index, (image_path, label_path) in enumerate(pairs):
        if image_path.stem != label_path.stem:
            raise ValueError(
                f"Sorted pair {index} has different filenames: "
                f"{image_path.name} != {label_path.name}."
            )
    if len(pairs) < 2:
        raise ValueError("At least two image/label pairs are needed for a train/test split.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pairs))
    test_count = min(max(round(len(pairs) * test_fraction), 1), len(pairs) - 1)
    test_indices = set(indices[:test_count].tolist())
    saved_train: list[Path] = []
    saved_test: list[Path] = []

    for index, (image_path, label_path) in enumerate(pairs):
        subset = "test" if index in test_indices else "train"
        destination_image_dir = output_dir / subset / "Images"
        destination_label_dir = output_dir / subset / "Labels"
        destination_image_dir.mkdir(parents=True, exist_ok=True)
        destination_label_dir.mkdir(parents=True, exist_ok=True)
        destination_image = destination_image_dir / image_path.name
        destination_label = destination_label_dir / label_path.name
        shutil.copy2(image_path, destination_image)
        shutil.copy2(label_path, destination_label)
        destinations = saved_test if subset == "test" else saved_train
        destinations.extend((destination_image, destination_label))

    print(f"Split {len(pairs)} pairs into {len(pairs) - test_count} train and {test_count} test pairs.")
    return saved_train, saved_test


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=repo_root / "data" / "interim")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "preprocessed_volumes")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_data(args.input_dir, args.output_dir, args.test_fraction, args.seed)


if __name__ == "__main__":
    main()
