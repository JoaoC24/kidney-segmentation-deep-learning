from __future__ import annotations

from pathlib import Path
import lightning as L
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.nn.functional as F
from src.configs import config_main as config


def config_persistent_workers(num_workers: int) -> bool:
    return bool(config.PERSISTENT_WORKERS) and num_workers > 0


def _validate_image_label_pairs(image_paths: list[Path], label_paths: list[Path]) -> None:
    if len(image_paths) != len(label_paths):
        raise ValueError(
            f"Different number of images ({len(image_paths)}) and labels ({len(label_paths)})."
        )
    mismatched = [
        (image_path.name, label_path.name)
        for image_path, label_path in zip(image_paths, label_paths)
        if image_path.stem != label_path.stem
    ]
    if mismatched:
        image_name, label_name = mismatched[0]
        raise ValueError(f"Image/label file mismatch: {image_name} != {label_name}")


class ClassificationDataset(Dataset):
    def __init__(self, data_dir: str | Path, image_size: int = 128, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "Images"
        self.label_dir = self.data_dir / "Labels"
        self.image_size = (image_size, image_size)
        self.augment = augment
        
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Label folder not found: {self.label_dir}")
        
        self.image_paths = sorted(self.image_dir.glob("*.npy"))
        self.label_paths = sorted(self.label_dir.glob("*.npy"))

        if not self.image_paths:
            raise FileNotFoundError(f"No .npy files found in: {self.image_dir}")
        if not self.label_paths:
            raise FileNotFoundError(f"No .npy files found in: {self.label_dir}")

        _validate_image_label_pairs(self.image_paths, self.label_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = self._load_image(self.image_paths[index])
        label_array = self._load_label(self.label_paths[index])

        image = self._resize_image(image)
        
        label = (label_array > 0).any().astype(int)
        
        if self.augment and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[-1])
        
        return image, label
        
    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        array = np.load(path)
        if array.ndim != 2:
            raise ValueError(f"Loader: expected a 2D image array, got shape {array.shape}.")
        return torch.from_numpy(array).float().unsqueeze(0)        

    def _load_label(self, path: Path) -> torch.Tensor:
        array = np.load(path)
        if array.ndim != 2:
            raise ValueError(f"Loader: expected a 2D label array, got shape {array.shape}.")
        return array
    
    def _resize_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)
        return image.squeeze(0)

class MyClassificationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path = config.DATA_DIR,
        batch_size: int = 16,
        image_size: int = 128,
        num_workers: int = 4,
        val_fraction: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            train_dir = self.data_dir / "train"
            split_dataset = ClassificationDataset(
                data_dir=train_dir,
                image_size=self.image_size,
                augment=False,
            )

            val_size = int(len(split_dataset) * self.val_fraction)
            train_size = len(split_dataset) - val_size

            train_indices, val_indices = random_split(
                range(len(split_dataset)),
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            train_dataset = ClassificationDataset(
                data_dir=train_dir,
                image_size=self.image_size,
                augment=True,
            )
            val_dataset = ClassificationDataset(
                data_dir=train_dir,
                image_size=self.image_size,
                augment=False,
            )

            self.train_ds = Subset(train_dataset, train_indices.indices)
            self.val_ds = Subset(val_dataset, val_indices.indices)

        if stage in (None, "test", "predict"):
            self.test_dataset = ClassificationDataset(
                data_dir=self.data_dir / "test",
                image_size=self.image_size,
                augment=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,  # otimiza a transefrencia de dados CPU->GPU, mas aumenta a carga na RAM CPU
            persistent_workers=config_persistent_workers(self.num_workers)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,  # same as above
            persistent_workers=config_persistent_workers(self.num_workers)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=config_persistent_workers(self.num_workers),
        )


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: str | Path, image_size: int = 128, multiclass=False, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "Images"
        self.label_dir = self.data_dir / "Labels"
        self.image_size = (image_size, image_size)
        self.multiclass = multiclass
        self.augment = augment
        
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Label folder not found: {self.label_dir}")
        
        self.image_paths = sorted(self.image_dir.glob("*.npy"))
        self.label_paths = sorted(self.label_dir.glob("*.npy"))

        if not self.image_paths:
            raise FileNotFoundError(f"No .npy files found in: {self.image_dir}")
        if not self.label_paths:
            raise FileNotFoundError(f"No .npy files found in: {self.label_dir}")

        _validate_image_label_pairs(self.image_paths, self.label_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = self._load_image(self.image_paths[index])
        label = self._load_label(self.label_paths[index])

        image = self._resize_image(image)
        label = self._resize_label(label)
        
        if self.augment and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[-2]) # vertical
            label = torch.flip(label, dims=[-2])
            if not self.multiclass and torch.rand(()) < 0.5:
                image = torch.flip(image, dims=[-1]) # horizontal
                label = torch.flip(label, dims=[-1])
        
        return image, label
        
    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        array = np.load(path)
        if array.ndim != 2:
            raise ValueError(f"Loader: expected a 2D image array, got shape {array.shape}.")
        return torch.from_numpy(array).float().unsqueeze(0)        

    def _load_label(self, path: Path) -> torch.Tensor:
        array = np.load(path)
        if array.ndim != 2:
            raise ValueError(f"Loader: expected a 2D label array, got shape {array.shape}.")
        if not self.multiclass:
            array = (array > 0).astype(np.int64)
        return torch.from_numpy(array).long()
    
    def _resize_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)
        return image.squeeze(0)

    def _resize_label(self, label: torch.Tensor) -> torch.Tensor:
        label = label.unsqueeze(0).unsqueeze(0).float()
        label = F.interpolate(label, size=self.image_size, mode="nearest")
        return label.squeeze(0).squeeze(0).long() # without channel dimension


class MyDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path = config.DATA_DIR,
        multiclass: bool =False,
        batch_size: int = 16,
        image_size: int = 128,
        num_workers: int = 4,
        val_fraction: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.multiclass = multiclass
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            train_dir = self.data_dir / "train"
            split_dataset = SegmentationDataset(
                data_dir=train_dir,
                image_size=self.image_size,
                multiclass=self.multiclass,
                augment=False,
            )

            val_size = int(len(split_dataset) * self.val_fraction)
            train_size = len(split_dataset) - val_size
            train_indices, val_indices = random_split(
                range(len(split_dataset)),
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            train_dataset = SegmentationDataset(
                data_dir=train_dir,
                image_size=self.image_size,
                multiclass=self.multiclass,
                augment=True,
            )
            val_dataset = SegmentationDataset(
                data_dir=train_dir,
                image_size=self.image_size,
                multiclass=self.multiclass,
                augment=False,
            )
            self.train_ds = Subset(train_dataset, train_indices.indices)
            self.val_ds = Subset(val_dataset, val_indices.indices)

        if stage in (None, "test", "predict"):
            self.test_dataset = SegmentationDataset(
                data_dir=self.data_dir / "test",
                image_size=self.image_size,
                multiclass=self.multiclass,
                augment=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,  # otimiza a transefrencia de dados CPU->GPU, mas aumenta a carga na RAM CPU
            persistent_workers=config_persistent_workers(self.num_workers)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,  # same as above
            persistent_workers=config_persistent_workers(self.num_workers)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=config_persistent_workers(self.num_workers),
        )


class VolumeSegmentationDataset(Dataset):
    """Load paired 3D volumes and resize each slice for inference."""

    def __init__(
        self,
        data_dir: str | Path = config.VOLUME_DATA_DIR / "test",
        image_size: int = 128,
        multiclass: bool = False,
        slice_axis: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "Images"
        self.label_dir = self.data_dir / "Labels"
        self.image_size = (image_size, image_size)
        self.multiclass = multiclass
        self.slice_axis = slice_axis

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.image_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"Label folder not found: {self.label_dir}")
        self.image_paths = sorted(self.image_dir.glob("*.npy"))
        self.label_paths = sorted(self.label_dir.glob("*.npy"))
        if not self.image_paths:
            raise FileNotFoundError(f"No .npy files found in: {self.image_dir}")
        if not self.label_paths:
            raise FileNotFoundError(f"No .npy files found in: {self.label_dir}")
        _validate_image_label_pairs(self.image_paths, self.label_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path, label_path = self.image_paths[index], self.label_paths[index]
        image = self._move_slice_axis(self._load_volume(image_path).float())
        label = self._move_slice_axis(self._load_volume(label_path).long())
        if image.shape != label.shape:
            raise ValueError(
                f"Image/label shape mismatch for {image_path.name}: "
                f"{tuple(image.shape)} != {tuple(label.shape)}"
            )

        image = self._resize_image_volume(image)
        label = self._resize_label_volume(label)
        if not self.multiclass:
            label = (label > 0).long()
        return image, label, str(image_path), str(label_path)

    @staticmethod
    def _load_volume(path: Path) -> torch.Tensor:
        array = np.load(path)
        if array.ndim != 3:
            raise ValueError(f"Loader: expected a 3D volume array, got shape {array.shape}.")
        return torch.from_numpy(array)

    def _move_slice_axis(self, volume: torch.Tensor) -> torch.Tensor:
        if self.slice_axis < 0 or self.slice_axis >= volume.ndim:
            raise ValueError(f"slice_axis must be in [0, {volume.ndim - 1}], got {self.slice_axis}.")
        return volume if self.slice_axis == 0 else torch.movedim(volume, self.slice_axis, 0)

    def _resize_image_volume(self, volume: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            volume.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=False
        )

    def _resize_label_volume(self, volume: torch.Tensor) -> torch.Tensor:
        volume = F.interpolate(volume.unsqueeze(1).float(), size=self.image_size, mode="nearest")
        return volume.squeeze(1).long()


class VolumeSegmentationDataModule(L.LightningDataModule):
    """Inference datamodule reading 3D test volumes from VOLUME_DATA_DIR/test."""

    def __init__(
        self,
        data_dir: str | Path = config.VOLUME_DATA_DIR,
        multiclass: bool = False,
        image_size: int = 128,
        num_workers: int = 4,
        slice_axis: int = 0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.multiclass = multiclass
        self.image_size = image_size
        self.num_workers = num_workers
        self.slice_axis = slice_axis

    def setup(self, stage=None):
        if stage in (None, "test", "predict"):
            self.test_dataset = VolumeSegmentationDataset(
                data_dir=self.data_dir / "test",
                image_size=self.image_size,
                multiclass=self.multiclass,
                slice_axis=self.slice_axis,
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=config_persistent_workers(self.num_workers),
        )
