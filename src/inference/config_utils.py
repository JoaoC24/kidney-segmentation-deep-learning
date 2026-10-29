import argparse
import importlib
import keyword
import re


def load_inference_config(argv: list[str] | None = None):
    """Import the inference config named on the command line."""
    parser = argparse.ArgumentParser(description="Run model testing with a named configuration.")
    parser.add_argument(
        "--config",
        required=True,
        help="Inference config filename in src/configs/inference_configs, with or without .py.",
    )
    config_name = parser.parse_args(argv).config.removesuffix(".py")
    if not re.fullmatch(r"[A-Za-z_]\w*", config_name) or keyword.iskeyword(config_name):
        parser.error("--config must be an inference config filename under src/configs/inference_configs.")
    return importlib.import_module(f"src.configs.inference_configs.{config_name}")
