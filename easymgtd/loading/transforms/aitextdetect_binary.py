"""
Transform for AITextDetect dataset (binary classification).

Handles the complex AITextDetect loading logic:
    - Reads human data from Human/{category}/ directory (multiple JSON sources)
    - Reads machine data from {targetLLM}_new/{category}_task3.json
    - Supports both subject-level and topic-level loading
    - Balances human and machine data (50:50)
    - Text truncation via tiktoken (max 2048 tokens)

Output: list[BinarySample]

Migrated from dataloader.py:
    - _load_aitextdetect() (L288-385)
    - load_subject_data() (L388-417)
    - load_topic_data() (L420-460)
"""

import os
import json
import random

import tiktoken
import pandas as pd
from datasets import load_dataset, Dataset

from ..registry import DatasetRegistry, DatasetTransform
from ..schemas import BinarySample
from ..constants import (
    DATASET_AITextDetect,
    SAVED_DATA_DIR,
    CATEGORIES,
    TOPIC_MAPPING,
    AITEXTDETECT_SOURCE_DICT,
)
from ..pipeline import load_cache, save_cache
from easymgtd.utils import setup_seed


# ==============================================================================
# Internal data loading helper (from original _load_aitextdetect)
# ==============================================================================

# Shared tiktoken encoding instance
_encoding = None


def _get_encoding():
    """Lazily initialize the tiktoken encoding."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def _truncate_text(text: str, max_tokens: int = 2048) -> str:
    """
    Truncate text to max_tokens using tiktoken, ending at a sentence boundary.

    Args:
        text: Input text.
        max_tokens: Maximum number of tokens.

    Returns:
        Truncated text (or original if within limit).
    """
    encoding = _get_encoding()
    tokens = encoding.encode(text, allowed_special={"<|endoftext|>"})
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(tokens)
        # Try to end at a sentence boundary
        last_period_idx = truncated_text.rfind("。")
        if last_period_idx == -1:
            last_period_idx = truncated_text.rfind(".")
        if last_period_idx != -1:
            truncated_text = truncated_text[: last_period_idx + 1]
        return truncated_text
    return text


def load_aitextdetect_split(
    repo: str, name: str, split: str
) -> Dataset:
    """
    Load a split of the AITextDetect dataset from local JSON files.

    This replaces the original _load_aitextdetect() function, handling
    both Human data (multi-source) and model-generated data.

    Args:
        repo: Base directory path for the dataset.
        name: "Human" or model name (e.g., "gpt35").
        split: Category/subject name (e.g., "Art", "Physics").

    Returns:
        HuggingFace Dataset with 'text' column.

    Raises:
        FileNotFoundError: If no data files are found.
    """
    base_dir = repo
    if repo.endswith(".py"):
        base_dir = os.path.dirname(repo)
    if not os.path.isdir(base_dir):
        # Fallback to HuggingFace remote loading
        return load_dataset(repo, trust_remote_code=True, name=name, split=split)

    # Identify files to load
    filepaths = []
    if name == "Human":
        dest_dir = os.path.join(base_dir, "Human", split)
        sources = AITEXTDETECT_SOURCE_DICT.get(split, [])
        for src in sources:
            fn = os.path.join(dest_dir, f"{split}_{src}_new.json")
            if os.path.exists(fn):
                filepaths.append(fn)
    else:
        fn = os.path.join(base_dir, f"{name}_new", f"{split}_task3.json")
        if os.path.exists(fn):
            filepaths.append(fn)

    if not filepaths:
        raise FileNotFoundError(
            f"Could not find local data for AITextDetect: {name} - {split} in {base_dir}"
        )

    # Read and process all files
    records = []
    global_key = 0
    for file in filepaths:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            if not row["text"].strip():
                continue
            text = _truncate_text(row["text"], max_tokens=2048)

            meta = None
            if name == "Human":
                meta = {
                    k: row["meta"][k]
                    for k in ["data_source", "category", "other"]
                    if k in row.get("meta", {})
                }
                if meta.get("other") and "level" in meta["other"]:
                    meta["other"] = None

            records.append(
                {
                    "id": row.get("id", global_key),
                    "text": text,
                    "file": os.path.basename(file),
                    "meta": meta,
                }
            )
            global_key += 1

    return Dataset.from_pandas(pd.DataFrame(records))


# ==============================================================================
# Transform implementation
# ==============================================================================


@DatasetRegistry.register("AITextDetect")
class AITextDetectBinaryTransform(DatasetTransform):
    """
    Transform for AITextDetect dataset (binary classification).

    Supports two modes via the 'category' parameter:
    - Subject mode: category in CATEGORIES (e.g., "Art", "Physics")
      -> loads one subject's human + machine data
    - Topic mode: category in TOPICS (e.g., "STEM", "Humanities")
      -> loads all subjects under that topic, balanced

    The transform handles its own data loading internally since
    AITextDetect data comes from multiple JSON files in a directory
    structure, not a single file.
    """

    output_schema = BinarySample

    def transform(self, raw_data: list[dict], **kwargs) -> list[BinarySample]:
        """
        Transform AITextDetect data into BinarySamples.

        This method dispatches to subject-level or topic-level loading
        based on the category parameter.

        Args:
            raw_data: Ignored (data is loaded internally from repo directory).
            **kwargs:
                targetLLM (str): Target LLM model name.
                category (str): Subject or topic name.
                repo (str, optional): Path to dataset directory.
                seed (int, optional): Random seed.

        Returns:
            List of BinarySample instances.
        """
        targetLLM = kwargs.get("targetLLM")
        if targetLLM is None:
            raise ValueError("AITextDetect transform requires 'targetLLM' parameter")

        category = kwargs.get("category", "Art")
        repo = kwargs.get("repo", DATASET_AITextDetect)
        seed = kwargs.get("seed", 0)

        if category in CATEGORIES:
            return self._load_subject(targetLLM, category, seed, repo)
        elif category in ("STEM", "Humanities", "Social_sciences"):
            return self._load_topic(targetLLM, category, seed, repo)
        else:
            raise ValueError(f"Unknown category: {category}")

    def _load_subject(
        self, targetLLM: str, category: str, seed: int, repo: str
    ) -> list[BinarySample]:
        """
        Load subject-level data (single category).

        Balances human and machine data to 50:50.
        """
        print(f"Loading human data for {category}")
        human_data = load_aitextdetect_split(repo, name="Human", split=category)

        print(f"Loading machine data for {targetLLM}/{category}")
        mgt_data = load_aitextdetect_split(repo, name=targetLLM, split=category)

        print("Data loaded")

        # Balance: use the smaller of the two
        smaller_len = min(len(human_data), len(mgt_data))
        human_data = human_data.shuffle(seed)

        samples = []
        for i in range(smaller_len):
            samples.append(BinarySample(text=mgt_data[i]["text"], label=1))
            samples.append(BinarySample(text=human_data[i]["text"], label=0))

        return samples

    def _load_topic(
        self, targetLLM: str, topic: str, seed: int, repo: str
    ) -> list[BinarySample]:
        """
        Load topic-level data (multiple subjects under one topic).

        Balances across all subjects within the topic.
        """
        setup_seed(seed)

        all_human = []
        all_mgt = []

        for subject in CATEGORIES:
            if TOPIC_MAPPING[subject] == topic:
                human_data = load_aitextdetect_split(
                    repo, name="Human", split=subject
                )
                mgt_data = load_aitextdetect_split(
                    repo, name=targetLLM, split=subject
                )
                all_human.append(human_data)
                all_mgt.append(mgt_data)

        # Find minimum length across all subjects (balance between subjects)
        min_len = min(
            min(len(h) for h in all_human),
            min(len(m) for m in all_mgt),
        )

        # Balance and collect samples
        samples = []
        for i in range(len(all_human)):
            human_subset = all_human[i].shuffle().select(range(min_len))
            mgt_subset = all_mgt[i].shuffle().select(range(min_len))

            for j in range(min_len):
                samples.append(BinarySample(text=mgt_subset[j]["text"], label=1))
                samples.append(BinarySample(text=human_subset[j]["text"], label=0))

        random.shuffle(samples)
        return samples
