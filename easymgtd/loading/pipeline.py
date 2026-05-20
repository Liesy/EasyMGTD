"""
Data processing pipeline for the EasyMGTD data loading system.

Provides the universal pipeline that transforms schema-conforming samples
into experiment-ready data dicts with train/test splits.

Key functions:
    - build_experiment_data: shuffle → split → process → cache → return
    - process_spaces: canonical text normalization
    - load_cache / save_cache: JSON cache management
"""

import os
import json
import random
import tqdm

from .schemas import (
    BinarySample,
    MultiClassSample,
    AttributionSample,
    FineGrainedSample,
)


def process_spaces(text: str) -> str:
    """
    Normalize punctuation spacing in text.

    This is the single canonical implementation, replacing duplicated
    versions previously in dataloader.py and dataloader_attribution.py.

    Args:
        text: Raw text string.

    Returns:
        Text with normalized spacing around punctuation.
    """
    return (
        text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ;", ";")
        .replace(" '", "'")
        .replace(" ' ", "'")
        .replace(" :", ":")
        .replace("<newline>", "\n")
        .replace("`` ", '"')
        .replace(" ''", '"')
        .replace("''", '"')
        .replace(".. ", "... ")
        .replace(" )", ")")
        .replace("( ", "(")
        .replace(" n't", "n't")
        .replace(" i ", " I ")
        .replace(" i'", " I'")
        .replace("\\'", "'")
        .replace("\n ", "\n")
        .strip()
    )


def load_cache(cache_path: str) -> dict | None:
    """
    Check if a cache file exists and load it.

    Also prints dataset statistics (train/test split, label distribution).

    Args:
        cache_path: Path to the JSON cache file.

    Returns:
        Loaded data dict if cache exists, None otherwise.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Using cached data: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Print statistics
        for split_name in ["train", "test"]:
            if split_name in data and "label" in data[split_name]:
                labels = data[split_name]["label"]
                total = len(labels)
                if total > 0:
                    machine = sum(1 for l in labels if l > 0)
                    human = total - machine
                    print(f"  {split_name}: total={total}, human={human}, machine={machine}")

        return data
    return None


def save_cache(data: dict, cache_path: str) -> None:
    """
    Save experiment data to a JSON cache file.

    Creates parent directories if they don't exist.

    Args:
        data: Experiment data dict to save.
        cache_path: Path to the JSON cache file.
    """
    if cache_path and not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving experiment data to {cache_path}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)


def build_experiment_data(
    samples: list,
    *,
    seed: int = 0,
    split_ratio: float = 0.8,
    cache_path: str = None,
) -> dict:
    """
    Universal pipeline: cache check → shuffle → split → process_spaces → save → return.

    Converts a list of sample dataclass instances into the standard experiment
    data dict format expected by BaseExperiment.load_data().

    Handles different sample types:
    - BinarySample / MultiClassSample / AttributionSample:
      Output: {"train": {"text": [], "label": []}, "test": {...}}
    - FineGrainedSample:
      Output: {"train": {"text": [], "label": [], "sentences": [], "sentence_labels": []}, "test": {...}}

    Args:
        samples: List of sample dataclass instances.
        seed: Random seed for shuffling.
        split_ratio: Fraction of data for training (default 0.8).
        cache_path: Optional path for JSON cache file.

    Returns:
        dict compatible with BaseExperiment.load_data().
    """
    # Check cache first
    cached = load_cache(cache_path)
    if cached is not None:
        return cached

    # Shuffle
    random.seed(seed)
    random.shuffle(samples)

    total = len(samples)
    split_idx = int(total * split_ratio)

    # Detect sample type for appropriate output structure
    is_fine_grained = total > 0 and isinstance(samples[0], FineGrainedSample)

    if is_fine_grained:
        data = {
            "train": {"text": [], "label": [], "sentences": [], "sentence_labels": []},
            "test": {"text": [], "label": [], "sentences": [], "sentence_labels": []},
        }
    else:
        data = {
            "train": {"text": [], "label": []},
            "test": {"text": [], "label": []},
        }

    for i, sample in enumerate(tqdm.tqdm(samples, desc="Building experiment data")):
        partition = "train" if i < split_idx else "test"
        data[partition]["text"].append(process_spaces(sample.text))
        data[partition]["label"].append(sample.label)

        if is_fine_grained:
            data[partition]["sentences"].append(sample.sentences)
            data[partition]["sentence_labels"].append(sample.sentence_labels)

        # Preserve optional metadata (e.g., 'category' from MultiClassSample)
        if isinstance(sample, MultiClassSample) and sample.category is not None:
            if "category" not in data[partition]:
                data[partition]["category"] = []
            data[partition]["category"].append(sample.category)

    # Save cache
    save_cache(data, cache_path)

    return data
