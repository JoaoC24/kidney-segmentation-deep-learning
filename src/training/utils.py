import importlib.util
import lightning as L
import time
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks import Callback
from pathlib import Path


def load_config(config_path: Path):
    spec = importlib.util.spec_from_file_location(
        "config", config_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config: {config_path}")

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def check_optimizer_name(value: object) -> str | None:
    if not isinstance(value, str) or value not in {"AdamW", "SGD"}:
        print(f"Warning: invalid optimizer name {value!r}, using the model default (AdamW).")
        return None
    return value


def check_learning_rate(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        print(f"Warning: invalid learning rate {value!r}; using the model default.")
        return None
    return value


def check_loss_name(value: object) -> str:
    if not isinstance(value, str) or value not in {"Focal_Dice", "BCE_Dice"}:
        print(f"Warning: invalid loss name {value!r}, defaulting to BCE_Dice.")
        return "BCE_Dice"
    return value


def check_gamma(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        print(f"Warning: invalid gamma {value!r}; using the model default.")
        return None
    return value


def check_class_weights(value: object) -> tuple[float, float] | None:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(isinstance(weight, bool) or not isinstance(weight, (int, float)) for weight in value)
    ):
        print(f"Warning: invalid class weights {value!r}; using the model default.")
        return None
    return float(value[0]), float(value[1])


def check_loss_weight(value: object, name: str) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        print(f"Warning: invalid {name} weight {value!r}; using the model default.")
        return None
    return value


def check_vit_depth(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        print(f"Warning: invalid ViT depth {value!r}; using the model default.")
        return None
    return value
