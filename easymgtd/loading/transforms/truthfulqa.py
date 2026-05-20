"""
Transform for TruthfulQA dataset.

Raw data format (CSV):
    - Question: str
    - Best Answer: str (human answer)
    - {targetLLM}_answer: str (model-generated answer)
    - Category: str

Output: list[BinarySample]

Migrated from dataloader.py load_TruthfulQA() (L171-192).
"""

import os

from ..registry import DatasetRegistry, DatasetTransform
from ..schemas import BinarySample
from ..readers import read_csv
from ..constants import DATASET_DIR_OTHERS


@DatasetRegistry.register("TruthfulQA")
class TruthfulQATransform(DatasetTransform):
    """
    Transform for TruthfulQA dataset (binary classification).

    Reads a CSV with human answers and LLM-generated answers,
    filters by text length, and produces BinarySample pairs.
    """

    output_schema = BinarySample

    def transform(self, raw_data: list[dict], **kwargs) -> list[BinarySample]:
        """
        Transform TruthfulQA raw records into BinarySamples.

        Args:
            raw_data: List of dicts from CSV. If empty, reads from default path.
            **kwargs:
                targetLLM (str): Name of the target LLM (used to select the
                    answer column, e.g., "gpt35" -> "gpt35_answer").
                path (str, optional): Override path to CSV file.

        Returns:
            List of BinarySample instances.
        """
        targetLLM = kwargs.get("targetLLM")
        if targetLLM is None:
            raise ValueError("TruthfulQA transform requires 'targetLLM' parameter")

        # Load raw data if not provided
        if not raw_data:
            path = kwargs.get("path") or os.path.join(
                DATASET_DIR_OTHERS, "TruthfulQA_LLMs.csv"
            )
            raw_data = read_csv(path)

        answer_key = f"{targetLLM}_answer"
        samples = []

        for row in raw_data:
            human_text = str(row.get("Best Answer", ""))
            machine_text = str(row.get(answer_key, ""))

            # Filter: both texts must have >1 word, machine text <2000 chars
            if (
                len(human_text.split()) > 1
                and len(machine_text.split()) > 1
                and len(machine_text) < 2000
            ):
                # Ensure human text ends with a period
                if not human_text.endswith("."):
                    human_text += "."

                samples.append(BinarySample(text=human_text, label=0))
                samples.append(BinarySample(text=machine_text, label=1))

        return samples
