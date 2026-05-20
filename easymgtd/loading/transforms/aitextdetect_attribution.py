"""
Transform for AITextDetect attribution (source model identification).

Handles multi-model attribution where the task is to identify which model
generated a given text. Supports both subject-level and topic-level loading.

Output: list[AttributionSample]

Migrated from dataloader.py:
    - prepare_attribution() (L471-519)
    - prepare_attribution_topic() (L522-593)
    - load_attribution() (L613-627)
    - load_attribution_topic() (L596-610)
"""

import os
import json
import random
from concurrent.futures import ThreadPoolExecutor

from ..registry import DatasetRegistry, DatasetTransform
from ..schemas import AttributionSample
from ..constants import (
    DATASET_AITextDetect,
    SAVED_DATA_DIR,
    MODELS,
    CATEGORIES,
    TOPIC_MAPPING,
)
from ..pipeline import load_cache, save_cache, build_experiment_data
from .aitextdetect_binary import load_aitextdetect_split
from easymgtd.utils import setup_seed


def _download_data(model_name, category, repo):
    """Helper to download data for a single model (for concurrent loading)."""
    return load_aitextdetect_split(repo, name=model_name, split=category)


@DatasetRegistry.register("AITextDetect_Attribution")
class AITextDetectAttributionTransform(DatasetTransform):
    """
    Transform for AITextDetect attribution at subject level.

    Loads human data + all model data for a single category/subject,
    balances across models, and assigns per-model labels.

    Label mapping: Human=0, Moonshot=1, gpt35=2, Mixtral=3, Llama3=4, gpt-4omini=5
    """

    output_schema = AttributionSample

    # Label assignment for attribution
    _label_mapping = {
        "Human": 0,
        "Moonshot": 1,
        "gpt35": 2,
        "Mixtral": 3,
        "Llama3": 4,
        "gpt-4omini": 5,
    }

    def transform(self, raw_data: list[dict], **kwargs) -> list[AttributionSample]:
        """
        Build attribution samples for a single category.

        Args:
            raw_data: Ignored (data loaded internally).
            **kwargs:
                category (str): Subject name (e.g., "Art").
                repo (str, optional): Dataset directory path.
                seed (int, optional): Random seed.

        Returns:
            List of AttributionSample instances.
        """
        category = kwargs.get("category", "Art")
        repo = kwargs.get("repo", DATASET_AITextDetect)
        seed = kwargs.get("seed", 0)

        setup_seed(seed)

        # Load human data
        human_data = load_aitextdetect_split(repo, name="Human", split=category)

        # Load all model data concurrently
        model_data = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_download_data, m, category, repo): m for m in MODELS
            }
            for future in futures:
                model_name = futures[future]
                try:
                    model_data[model_name] = future.result()
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")

        # Balance: find minimum length across all models
        min_len = min(len(model_data[m]) for m in MODELS)

        # Balance model data
        for m in MODELS:
            model_data[m] = model_data[m].shuffle().select(range(min_len))

        # Build samples
        samples = []
        for m in MODELS:
            for d in model_data[m]:
                samples.append(
                    AttributionSample(text=d["text"], label=self._label_mapping[m])
                )

        # Add human samples
        human_sample = human_data.shuffle().select(range(min_len))
        for d in human_sample:
            samples.append(
                AttributionSample(text=d["text"], label=self._label_mapping["Human"])
            )

        return samples


@DatasetRegistry.register("AITextDetect_Attribution_Topic")
class AITextDetectAttributionTopicTransform(DatasetTransform):
    """
    Transform for AITextDetect attribution at topic level.

    Loads and balances data across all subjects within a topic,
    across all models plus human data.

    Label mapping: human=0, Moonshot=1, gpt35=2, Mixtral=3, Llama3=4, gpt-4omini=5
    """

    output_schema = AttributionSample

    _label_mapping = {
        "human": 0,
        "Moonshot": 1,
        "gpt35": 2,
        "Mixtral": 3,
        "Llama3": 4,
        "gpt-4omini": 5,
    }

    def transform(self, raw_data: list[dict], **kwargs) -> list[AttributionSample]:
        """
        Build attribution samples for a topic (multiple subjects).

        Args:
            raw_data: Ignored.
            **kwargs:
                topic (str): Topic name ("STEM", "Humanities", "Social_sciences").
                repo (str, optional): Dataset directory path.
                seed (int, optional): Random seed.

        Returns:
            List of AttributionSample instances.
        """
        topic = kwargs.get("topic")
        if topic is None:
            raise ValueError("Attribution topic transform requires 'topic' parameter")

        repo = kwargs.get("repo", DATASET_AITextDetect)
        seed = kwargs.get("seed", 0)

        setup_seed(seed)

        # Load data per model per subject
        all_data = {"human": []}
        for model in MODELS:
            all_data[model] = []

        for model in MODELS:
            for subject in CATEGORIES:
                if TOPIC_MAPPING[subject] != topic:
                    continue
                mgt_data = load_aitextdetect_split(repo, name=model, split=subject)
                all_data[model].append(mgt_data)

        for subject in CATEGORIES:
            if TOPIC_MAPPING[subject] != topic:
                continue
            human_data = load_aitextdetect_split(repo, name="Human", split=subject)
            all_data["human"].append(human_data)

        # Balance within each model (across subjects)
        min_len_dict = {}
        for model in MODELS + ["human"]:
            min_len = min(len(d) for d in all_data[model])
            min_len_dict[model] = min_len

        for model in MODELS + ["human"]:
            for i in range(len(all_data[model])):
                all_data[model][i] = (
                    all_data[model][i].shuffle().select(range(min_len_dict[model]))
                )

        # Balance across models
        min_total = min(
            sum(len(d) for d in all_data[model]) for model in MODELS + ["human"]
        )

        # Build samples with round-robin across subjects
        samples = []
        for model in MODELS + ["human"]:
            cnt = 0
            idx = 0
            while cnt < min_total:
                for i in range(len(all_data[model])):
                    if idx < len(all_data[model][i]):
                        samples.append(
                            AttributionSample(
                                text=all_data[model][i][idx]["text"],
                                label=self._label_mapping[model],
                            )
                        )
                        cnt += 1
                idx += 1

        return samples
