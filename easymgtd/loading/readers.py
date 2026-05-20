"""
Multi-format file readers for the EasyMGTD data loading system.

Supports reading data from:
    - JSON files (single array or object)
    - JSONL files (one JSON object per line)
    - CSV files
    - Parquet files
    - HuggingFace Datasets (local or remote)

All readers return list[dict], providing a unified interface
for downstream Transform classes.
"""

import os
import json
from typing import Optional


def read_file(path: str, format: str = "auto") -> list[dict]:
    """
    Read data from a file in various formats.

    Args:
        path: Path to the data file.
        format: File format. One of "json", "jsonl", "csv", "parquet", "auto".
                If "auto", the format is inferred from the file extension.

    Returns:
        List of dictionaries, one per record.

    Raises:
        ValueError: If format cannot be inferred or is unsupported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    if format == "auto":
        format = _infer_format(path)

    reader_map = {
        "json": read_json,
        "jsonl": read_jsonl,
        "csv": read_csv,
        "parquet": read_parquet,
    }

    if format not in reader_map:
        raise ValueError(
            f"Unsupported format: {format}. "
            f"Supported: {list(reader_map.keys())} and 'huggingface'."
        )

    return reader_map[format](path)


def _infer_format(path: str) -> str:
    """
    Infer file format from extension.

    Args:
        path: File path.

    Returns:
        Format string: "json", "jsonl", "csv", or "parquet".

    Raises:
        ValueError: If extension is unrecognized.
    """
    ext = os.path.splitext(path)[1].lower()
    ext_map = {
        ".json": "json",
        ".jsonl": "jsonl",
        ".csv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
    }
    if ext not in ext_map:
        raise ValueError(
            f"Cannot infer format from extension '{ext}'. "
            f"Supported extensions: {list(ext_map.keys())}. "
            f"Please specify format explicitly."
        )
    return ext_map[ext]


def read_json(path: str) -> list[dict]:
    """
    Read a JSON file.

    Handles two formats:
    - Array of objects: [{"text": "...", "label": 0}, ...]
    - Single object with list values: {"text": ["...", ...], "label": [0, ...]}
      (columnar format, converted to row-based)

    Args:
        path: Path to the JSON file.

    Returns:
        List of dictionaries.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Columnar format: convert to row-based
        keys = list(data.keys())
        if not keys:
            return []
        n = len(data[keys[0]])
        return [{k: data[k][i] for k in keys} for i in range(n)]
    else:
        raise ValueError(f"Unexpected JSON root type: {type(data)}")


def read_jsonl(path: str) -> list[dict]:
    """
    Read a JSONL file (one JSON object per line).

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dictionaries.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e
    return records


def read_csv(path: str) -> list[dict]:
    """
    Read a CSV file using pandas.

    Args:
        path: Path to the CSV file.

    Returns:
        List of dictionaries (one per row).
    """
    import pandas as pd

    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def read_parquet(path: str) -> list[dict]:
    """
    Read a Parquet file using pandas.

    Args:
        path: Path to the Parquet file.

    Returns:
        List of dictionaries (one per row).
    """
    import pandas as pd

    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def read_huggingface(
    repo_or_path: str,
    name: Optional[str] = None,
    split: Optional[str] = None,
    **kwargs,
) -> list[dict]:
    """
    Read data from a HuggingFace Dataset (local or remote).

    Args:
        repo_or_path: HuggingFace repository ID or local path.
        name: Dataset configuration name (if applicable).
        split: Dataset split to load (e.g., "train", "test").
        **kwargs: Additional arguments passed to datasets.load_dataset().

    Returns:
        List of dictionaries (one per record).
    """
    from datasets import load_dataset

    load_kwargs = {}
    if name is not None:
        load_kwargs["name"] = name
    if split is not None:
        load_kwargs["split"] = split
    load_kwargs.update(kwargs)

    dataset = load_dataset(repo_or_path, **load_kwargs)

    # If no split was specified, load_dataset returns a DatasetDict
    # In that case, try to get a reasonable default
    from datasets import DatasetDict

    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            # Use the first available split
            first_key = list(dataset.keys())[0]
            dataset = dataset[first_key]

    return [dict(row) for row in dataset]
