"""
Transform for NarrativeQA dataset.

Raw data format (CSV):
    - Question: str
    - answers: str (semicolon-separated, first one used as human answer)
    - {targetLLM}_answer: str (model-generated answer)

Output: list[BinarySample]

Migrated from dataloader.py load_NarrativeQA() (L213-234).
"""

import os

from ..registry import DatasetRegistry, DatasetTransform
from ..schemas import BinarySample
from ..readers import read_csv
from ..constants import DATASET_DIR_OTHERS


@DatasetRegistry.register("NarrativeQA")
class NarrativeQATransform(DatasetTransform):
    """
    Transform for NarrativeQA dataset (binary classification).

    Uses the first semicolon-separated answer as the human answer.
    Filters by word count (1 < words < 150).
    """

    output_schema = BinarySample

    def transform(self, raw_data: list[dict], **kwargs) -> list[BinarySample]:
        """
        Transform NarrativeQA raw records into BinarySamples.

        Args:
            raw_data: List of dicts from CSV. If empty, reads from default path.
            **kwargs:
                targetLLM (str): Name of the target LLM.
                path (str, optional): Override path to CSV file.

        Returns:
            List of BinarySample instances.
        """
        targetLLM = kwargs.get("targetLLM")
        if targetLLM is None:
            raise ValueError("NarrativeQA transform requires 'targetLLM' parameter")

        if not raw_data:
            path = kwargs.get("path") or os.path.join(
                DATASET_DIR_OTHERS, "NarrativeQA_LLMs.csv"
            )
            raw_data = read_csv(path)

        answer_key = f"{targetLLM}_answer"
        samples = []

        for row in raw_data:
            # Human answer: first semicolon-separated segment
            answers_raw = str(row.get("answers", ""))
            human_text = answers_raw.split(";")[0]
            machine_text = str(row.get(answer_key, ""))

            human_words = len(human_text.split())
            machine_words = len(machine_text.split())

            # Filter: both texts must have 1 < words < 150
            if (
                human_words > 1
                and machine_words > 1
                and human_words < 150
                and machine_words < 150
            ):
                # Ensure human text ends with a period
                if not human_text.endswith("."):
                    human_text += "."

                samples.append(BinarySample(text=human_text, label=0))
                samples.append(BinarySample(text=machine_text, label=1))

        return samples
