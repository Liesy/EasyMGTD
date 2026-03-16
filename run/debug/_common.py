"""
Shared utilities for single-method debug scripts.

This module extracts common logic from benchmark.py so that each debug script
stays minimal and focused on its own detection method.

Functions:
    init_env        - Set GPU, seed, and load config from config.json.
    load_demo_data  - Load dataset and truncate to train_size / test_size.
    print_results   - Pretty-print a list of DetectOutput objects.
    resolve_model_path - Resolve $ENV_VAR references in model paths.
"""

import os
import sys
import json
import torch

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `easymgtd` can be imported
# regardless of the current working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from easymgtd.utils import setup_seed
from easymgtd.loading.dataloader import load


# ---------------------------------------------------------------------------
# Default path to config.json (relative to the project root)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "run", "config.json")


def init_env(gpu_id: str = "0", config_path: str = None) -> dict:
    """
    Initialize the runtime environment for a debug run.

    Steps:
        1. Set CUDA_VISIBLE_DEVICES to `gpu_id`.
        2. Load config.json from `config_path` (defaults to run/config.json).
        3. Set the random seed from the global config section.

    Args:
        gpu_id: GPU device index string, e.g. "0" or "0,1".
        config_path: Absolute or relative path to config.json.
                     Defaults to <project_root>/run/config.json.

    Returns:
        The parsed configuration dict.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Apply global seed
    global_cfg = config.get("global", {})
    seed = global_cfg.get("seed", 3407)
    setup_seed(seed)

    return config


def resolve_model_path(path: str) -> str:
    """
    Resolve a model path that may contain $ENV_VAR references.

    If `path` starts with '$', treat the remainder as an environment variable
    name and return its value (falling back to the original string).

    Args:
        path: A model path string, possibly starting with '$'.

    Returns:
        The resolved path string.
    """
    if path and isinstance(path, str) and path.startswith("$"):
        env_var = path[1:]
        return os.environ.get(env_var, path)
    return path


def load_demo_data(config: dict, topic: str, detectLLM: str) -> dict:
    """
    Load a dataset and truncate it to the configured train/test sizes.

    Reads `train_size`, `test_size`, and `dataset` from config["global"],
    loads via easymgtd.loading.dataloader.load(), then truncates both
    the train and test splits.

    Args:
        config: Full configuration dict (from config.json).
        topic: Topic or category name, e.g. "STEM" or "Art".
        detectLLM: Name of the LLM whose generated text is detected.

    Returns:
        A dict with structure:
            {"train": {"text": [...], "label": [...]},
             "test":  {"text": [...], "label": [...]}}
    """
    global_cfg = config.get("global", {})
    train_size = global_cfg.get("train_size", 1000)
    test_size = global_cfg.get("test_size", 2000)
    dataset_name = global_cfg.get("dataset", "AITextDetect")

    data = load(dataset_name, detectLLM=detectLLM, category=topic)

    # Truncate to requested sizes
    demo = {
        "train": {
            "text": data["train"]["text"][:train_size],
            "label": data["train"]["label"][:train_size],
        },
        "test": {
            "text": data["test"]["text"][:test_size],
            "label": data["test"]["label"][:test_size],
        },
    }
    return demo


def print_results(results: list) -> None:
    """
    Pretty-print a list of DetectOutput objects to stdout.

    For each result with a `.train` or `.test` Metric, prints the
    corresponding metrics in a readable format.

    Args:
        results: A list of DetectOutput objects returned by experiment.launch().
    """
    for i, r in enumerate(results):
        print(f"========== Result {i} ==========")
        if r.name:
            print(f"  Name: {r.name}")
        if r.train:
            print(
                f"  Train: acc={r.train.acc:.4f}  prec={r.train.precision:.4f}  "
                f"rec={r.train.recall:.4f}  f1={r.train.f1:.4f}  auc={r.train.auc:.4f}"
            )
        if r.test:
            print(
                f"  Test:  acc={r.test.acc:.4f}  prec={r.test.precision:.4f}  "
                f"rec={r.test.recall:.4f}  f1={r.test.f1:.4f}  auc={r.test.auc:.4f}"
            )
    print("=" * 40)
