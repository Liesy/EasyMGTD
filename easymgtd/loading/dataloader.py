"""
Legacy compatibility wrapper for the EasyMGTD data loading system.

All dataset loading is now handled by DatasetRegistry.
This module provides backward-compatible function signatures
that delegate to the new registry-based system.

New code should use DatasetRegistry directly:
    from easymgtd.loading import DatasetRegistry
    data = DatasetRegistry.load("AITextDetect", targetLLM="gpt35", category="Art")
"""

from .registry import DatasetRegistry
from .constants import (
    DATASETS,
    MODELS,
    CATEGORIES,
    TOPICS,
    TOPIC_MAPPING,
    LABEL_MAPPING,
    DATASET_AITextDetect,
    DATASET_DIR_OTHERS,
    SAVED_DATA_DIR,
)


def load(name, targetLLM, category="Art", seed=0, repo=None, data_path=None):
    """
    Load a dataset by name.

    Args:
        name: Dataset name (e.g., "TruthfulQA", "AITextDetect").
        targetLLM: Target LLM name whose generated text is being detected.
        category: Category or topic name.
        seed: Random seed.
        repo: Dataset repository path (for AITextDetect).
        data_path: Override data file path.
    """
    kwargs = {"targetLLM": targetLLM, "category": category}
    if repo is not None:
        kwargs["repo"] = repo
    if data_path is not None:
        kwargs["path"] = data_path

    return DatasetRegistry.load(name, seed=seed, **kwargs)


def load_topic_data(targetLLM, topic, seed=0, repo=None):
    """Load AITextDetect data at topic level."""
    kwargs = {"targetLLM": targetLLM, "category": topic}
    if repo is not None:
        kwargs["repo"] = repo
    return DatasetRegistry.load("AITextDetect", seed=seed, **kwargs)


def load_subject_data(targetLLM, category, seed=0, repo=None):
    """Load AITextDetect data at subject level."""
    kwargs = {"targetLLM": targetLLM, "category": category}
    if repo is not None:
        kwargs["repo"] = repo
    return DatasetRegistry.load("AITextDetect", seed=seed, **kwargs)


def load_attribution(category="Art", seed=0, repo=None):
    """Load AITextDetect attribution data at subject level."""
    kwargs = {"category": category}
    if repo is not None:
        kwargs["repo"] = repo
    return DatasetRegistry.load("AITextDetect_Attribution", seed=seed, **kwargs)


def load_attribution_topic(topic, seed=0, repo=None):
    """Load AITextDetect attribution data at topic level."""
    kwargs = {"topic": topic}
    if repo is not None:
        kwargs["repo"] = repo
    return DatasetRegistry.load("AITextDetect_Attribution_Topic", seed=seed, **kwargs)


def load_incremental(order, category="Art", seed=0, repo=None):
    """Load AITextDetect incremental data at subject level."""
    kwargs = {"order": order, "category": category}
    if repo is not None:
        kwargs["repo"] = repo
    return DatasetRegistry.load("AITextDetect_Incremental", seed=seed, **kwargs)


def load_incremental_topic(order, topic, seed=0, repo=None):
    """Load AITextDetect incremental data at topic level."""
    kwargs = {"order": order, "topic": topic}
    if repo is not None:
        kwargs["repo"] = repo
    return DatasetRegistry.load("AITextDetect_Incremental_Topic", seed=seed, **kwargs)


# Re-export process_spaces from the new canonical location
from .pipeline import process_spaces


def download_data(model_name, category, repo=None):
    """Download raw AITextDetect data for a model/category."""
    if repo is None:
        repo = DATASET_AITextDetect
    from .transforms.aitextdetect_binary import load_aitextdetect_split

    return load_aitextdetect_split(repo, name=model_name, split=category)
