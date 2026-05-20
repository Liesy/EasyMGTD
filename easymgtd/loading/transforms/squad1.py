"""
Transform for SQuAD1 dataset.

Raw data format (CSV):
    - Question: str
    - answers: str (JSON-encoded dict with "text" key containing human answers)
    - {targetLLM}_answer: str (model-generated answer)

Output: list[BinarySample]

Migrated from dataloader.py load_SQuAD1() (L195-210).
"""

import os

from ..registry import DatasetRegistry, DatasetTransform
from ..schemas import BinarySample
from ..readers import read_csv
from ..constants import DATASET_DIR_OTHERS


@DatasetRegistry.register("SQuAD1")
class SQuAD1Transform(DatasetTransform):
    """
    Transform for SQuAD1 dataset (binary classification).

    Reads a CSV with human answers (encoded as JSON in 'answers' column)
    and LLM-generated answers.
    """

    output_schema = BinarySample

    def transform(self, raw_data: list[dict], **kwargs) -> list[BinarySample]:
        """
        Transform SQuAD1 raw records into BinarySamples.

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
            raise ValueError("SQuAD1 transform requires 'targetLLM' parameter")

        if not raw_data:
            path = kwargs.get("path") or os.path.join(
                DATASET_DIR_OTHERS, "SQuAD1_LLMs.csv"
            )
            raw_data = read_csv(path)

        answer_key = f"{targetLLM}_answer"
        samples = []

        for row in raw_data:
            # Human answer is JSON-encoded: {"text": ["answer"], ...}
            answers_raw = row.get("answers", "")
            try:
                human_text = str(eval(answers_raw)["text"][0])
            except (SyntaxError, KeyError, IndexError, TypeError):
                continue

            machine_text = str(row.get(answer_key, ""))

            if len(human_text.split()) > 1 and len(machine_text.split()) > 1:
                # Ensure human text ends with a period
                if not human_text.endswith("."):
                    human_text += "."

                samples.append(BinarySample(text=human_text, label=0))
                samples.append(BinarySample(text=machine_text, label=1))

        return samples
