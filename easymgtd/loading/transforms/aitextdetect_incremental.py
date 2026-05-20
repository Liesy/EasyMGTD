"""
Transform for AITextDetect incremental learning data.

Produces multi-stage train/test data where each stage introduces new LLM models.
Used by IncrementalExperiment, IncrementalThresholdExperiment, and FewShotExperiment.

Output format: {"train": [stage_dict, ...], "test": [stage_dict, ...]}
where each stage_dict = {"text": [...], "label": [...]}

Migrated from dataloader.py:
    - prepare_incremental() (L630-752)
    - prepare_incremental_topic() (L779-908)
    - load_incremental() (L755-776)
    - load_incremental_topic() (L921-943)
"""

import os
import json
import random
from concurrent.futures import ThreadPoolExecutor

from ..registry import DatasetRegistry, IncrementalTransform
from ..constants import (
    DATASET_AITextDetect,
    SAVED_DATA_DIR,
    CATEGORIES,
    TOPIC_MAPPING,
    LABEL_MAPPING,
)
from .aitextdetect_binary import load_aitextdetect_split
from easymgtd.utils import setup_seed


def _download_data(model_name, category, repo):
    """Helper to download data for a single model."""
    return load_aitextdetect_split(repo, name=model_name, split=category)


def _prepare_incremental_subject(
    order: list, category: str, seed: int, repo: str
) -> dict:
    """
    Prepare incremental data for a single subject/category.

    Args:
        order: List of lists defining model introduction order.
               Example: [['Moonshot'], ['gpt35', 'Llama3']]
        category: Subject name (e.g., "Art").
        seed: Random seed.
        repo: Dataset directory path.

    Returns:
        dict with "train" and "test" keys, each a list of stage dicts.
    """
    setup_seed(seed)

    # Load human data
    human_data = load_aitextdetect_split(repo, name="Human", split=category)

    # Load model data concurrently
    model_data = {}
    all_models = [model for group in order for model in group]
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_download_data, m, category, repo): m for m in all_models
        }
        for future in futures:
            model_name = futures[future]
            try:
                model_data[model_name] = future.result()
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")

    # Determine data length based on first round
    if len(order[0]) == 1:
        first_round_len = len(model_data[order[0][0]])
    else:
        first_round_len = min(len(model_data[model]) for model in order[0])

    # Limit human sample
    human_sample = human_data.shuffle().select(range(first_round_len))

    # Balance model data
    for m in model_data:
        if len(model_data[m]) > first_round_len:
            model_data[m] = model_data[m].shuffle().select(range(first_round_len))

    # Train/test split
    split = 0.8
    train_data = {}
    test_data = {}
    for m in all_models:
        available_len = len(model_data[m])
        current_len = min(available_len, first_round_len)
        train_data[m] = model_data[m].select(range(int(current_len * split)))
        test_data[m] = model_data[m].select(
            range(int(current_len * split), current_len)
        )

    train_data["Human"] = human_sample.select(range(int(first_round_len * split)))
    test_data["Human"] = human_sample.select(
        range(int(first_round_len * split), first_round_len)
    )

    # Build incremental stages
    data = {"train": [], "test": []}
    for i, group in enumerate(order):
        # Train stage
        temp_train = {"text": [], "label": []}
        if i == 0:
            for d in train_data["Human"]:
                temp_train["text"].append(d["text"])
                temp_train["label"].append(0)

        for j, model_name in enumerate(group):
            model_len = (
                min(len(train_data[model_name]), first_round_len)
                if i > 0
                else len(train_data[model_name])
            )
            model_train = train_data[model_name].select(range(model_len))
            for d in model_train:
                temp_train["text"].append(d["text"])
                temp_train["label"].append(j + 1 + sum(len(g) for g in order[:i]))

        if len(group) > 1 or i == 0:
            combined = list(zip(temp_train["text"], temp_train["label"]))
            random.shuffle(combined)
            temp_train["text"], temp_train["label"] = zip(*combined)
            temp_train["text"] = list(temp_train["text"])
            temp_train["label"] = list(temp_train["label"])

        data["train"].append(temp_train)

        # Test stage
        temp_test = {"text": [], "label": []}
        if i == 0:
            for d in test_data["Human"]:
                temp_test["text"].append(d["text"])
                temp_test["label"].append(0)

        for j, model_name in enumerate(group):
            model_len = (
                min(len(test_data[model_name]), first_round_len)
                if i > 0
                else len(test_data[model_name])
            )
            model_test = test_data[model_name].select(range(model_len))
            for d in model_test:
                temp_test["text"].append(d["text"])
                temp_test["label"].append(j + 1 + sum(len(g) for g in order[:i]))

        if i > 0:
            prev_test = data["test"][i - 1]
            temp_test["text"].extend(prev_test["text"])
            temp_test["label"].extend(prev_test["label"])

        data["test"].append(temp_test)

    return data


def _prepare_incremental_topic(
    order: list, topic: str, seed: int, repo: str
) -> dict:
    """
    Prepare incremental data for a topic (multiple subjects).

    Args:
        order: List of lists defining model introduction order.
        topic: Topic name ("STEM", "Humanities", "Social_sciences").
        seed: Random seed.
        repo: Dataset directory path.

    Returns:
        dict with "train" and "test" keys, each a list of stage dicts.
    """
    setup_seed(seed)

    all_models = [model for group in order for model in group]

    # Load all data per model per subject
    all_data = {"human": []}
    for model in all_models:
        all_data[model] = []

    # Load human data per subject
    for subject in CATEGORIES:
        if TOPIC_MAPPING[subject] == topic:
            human_data = load_aitextdetect_split(repo, name="Human", split=subject)
            all_data["human"].append(human_data)

    # Load model data per subject
    for model in all_models:
        for subject in CATEGORIES:
            if TOPIC_MAPPING[subject] == topic:
                mgt_data = load_aitextdetect_split(repo, name=model, split=subject)
                all_data[model].append(mgt_data)

    # Balance within each model across subjects
    for key in all_models + ["human"]:
        min_len = min(len(d) for d in all_data[key])
        for i in range(len(all_data[key])):
            all_data[key][i] = all_data[key][i].shuffle().select(range(min_len))
        # Flatten to list of dicts
        all_data[key] = [d for sublist in all_data[key] for d in sublist]
        random.shuffle(all_data[key])

    # Determine data length based on first round
    if len(order[0]) == 1:
        first_round_len = len(all_data[order[0][0]])
    else:
        first_round_len = min(len(all_data[model]) for model in order[0])

    # Limit human and model data
    random.shuffle(all_data["human"])
    human_sample = random.sample(all_data["human"], first_round_len)

    for m in all_data:
        if len(all_data[m]) > first_round_len:
            random.shuffle(all_data[m])
            all_data[m] = all_data[m][:first_round_len]

    # Train/test split
    split = 0.8
    train_data = {}
    test_data = {}
    for m in all_models:
        available_len = len(all_data[m])
        current_len = min(available_len, first_round_len)
        train_data[m] = all_data[m][: int(current_len * split)]
        test_data[m] = all_data[m][int(current_len * split) : current_len]

    train_data["Human"] = human_sample[: int(first_round_len * split)]
    test_data["Human"] = human_sample[int(first_round_len * split) : first_round_len]

    # Build incremental stages
    data = {"train": [], "test": []}
    for i, group in enumerate(order):
        # Train stage
        temp_train = {"text": [], "label": []}
        if i == 0:
            for d in train_data["Human"]:
                temp_train["text"].append(d["text"])
                temp_train["label"].append(0)

        for j, model_name in enumerate(group):
            model_len = (
                min(len(train_data[model_name]), first_round_len)
                if i > 0
                else len(train_data[model_name])
            )
            model_train = random.sample(train_data[model_name], model_len)
            for d in model_train:
                temp_train["text"].append(d["text"])
                temp_train["label"].append(j + 1 + sum(len(g) for g in order[:i]))

        if len(group) > 1 or i == 0:
            combined = list(zip(temp_train["text"], temp_train["label"]))
            random.shuffle(combined)
            temp_train["text"], temp_train["label"] = zip(*combined)
            temp_train["text"] = list(temp_train["text"])
            temp_train["label"] = list(temp_train["label"])

        data["train"].append(temp_train)

        # Test stage
        temp_test = {"text": [], "label": []}
        if i == 0:
            for d in test_data["Human"]:
                temp_test["text"].append(d["text"])
                temp_test["label"].append(0)

        for j, model_name in enumerate(group):
            model_len = (
                min(len(test_data[model_name]), first_round_len)
                if i > 0
                else len(test_data[model_name])
            )
            model_test = random.sample(test_data[model_name], model_len)
            for d in model_test:
                temp_test["text"].append(d["text"])
                temp_test["label"].append(j + 1 + sum(len(g) for g in order[:i]))

        if i > 0:
            prev_test = data["test"][i - 1]
            temp_test["text"].extend(prev_test["text"])
            temp_test["label"].extend(prev_test["label"])

        data["test"].append(temp_test)

    return data


# ==============================================================================
# Registered transforms
# ==============================================================================


@DatasetRegistry.register("AITextDetect_Incremental")
class AITextDetectIncrementalTransform(IncrementalTransform):
    """
    Incremental transform for AITextDetect at subject/category level.

    Produces multi-stage data with JSON caching.
    """

    def build(self, **kwargs) -> dict:
        """
        Build incremental data for a single category.

        Args:
            **kwargs:
                order (list[list[str]]): Model introduction order.
                    Example: [['Moonshot'], ['gpt35', 'Llama3']]
                category (str): Subject name (e.g., "Art").
                seed (int, optional): Random seed. Default 0.
                repo (str, optional): Dataset directory path.

        Returns:
            dict with "train" and "test" lists of stage dicts.
        """
        order = kwargs.get("order")
        if order is None:
            raise ValueError("Incremental transform requires 'order' parameter")

        category = kwargs.get("category", "Art")
        seed = kwargs.get("seed", 0)
        repo = kwargs.get("repo", DATASET_AITextDetect)

        # Build cache path
        seq = ""
        for model_group in order:
            for model in model_group:
                model_id = LABEL_MAPPING[model]
                seq += str(model_id)
            seq += "_"
        seq = seq[:-1]
        cache_path = os.path.join(
            SAVED_DATA_DIR, str(seed), f"{category}_incremental_{seq}.json"
        )

        # Check cache
        if os.path.exists(cache_path):
            print(f"Using cached data: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Build data
        data = _prepare_incremental_subject(order, category, seed=seed, repo=repo)

        # Save cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving experiment data to {cache_path}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        return data


@DatasetRegistry.register("AITextDetect_Incremental_Topic")
class AITextDetectIncrementalTopicTransform(IncrementalTransform):
    """
    Incremental transform for AITextDetect at topic level.

    Produces multi-stage data across all subjects within a topic.
    """

    def build(self, **kwargs) -> dict:
        """
        Build incremental data for a topic.

        Args:
            **kwargs:
                order (list[list[str]]): Model introduction order.
                topic (str): Topic name ("STEM", "Humanities", "Social_sciences").
                seed (int, optional): Random seed. Default 0.
                repo (str, optional): Dataset directory path.

        Returns:
            dict with "train" and "test" lists of stage dicts.
        """
        order = kwargs.get("order")
        if order is None:
            raise ValueError("Incremental topic transform requires 'order' parameter")

        topic = kwargs.get("topic")
        if topic is None:
            raise ValueError("Incremental topic transform requires 'topic' parameter")

        seed = kwargs.get("seed", 0)
        repo = kwargs.get("repo", DATASET_AITextDetect)

        # Build cache path
        seq = ""
        for model_group in order:
            for model in model_group:
                model_id = LABEL_MAPPING[model]
                seq += str(model_id)
            seq += "_"
        seq = seq[:-1]
        cache_path = os.path.join(
            SAVED_DATA_DIR, str(seed), f"{topic}_incremental_{seq}.json"
        )

        # Check cache
        if os.path.exists(cache_path):
            print(f"Using cached data: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Build data
        data = _prepare_incremental_topic(order, topic, seed=seed, repo=repo)

        # Save cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"Saving experiment data to {cache_path}")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        return data
