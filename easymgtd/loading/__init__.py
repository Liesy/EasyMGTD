"""
EasyMGTD data loading module.

Provides:
    - DatasetRegistry: central registry for dataset transforms
    - DatasetTransform / IncrementalTransform: base classes for new datasets
    - Schema classes: BinarySample, MultiClassSample, AttributionSample, etc.
    - Readers: multi-format file readers (JSON, JSONL, CSV, Parquet, HuggingFace)
    - Legacy compatibility: load_incremental, load_incremental_topic, etc.
"""

from .model_loader import load_pretrained, load_pretrained_mask, load_pretrained_supervise
from .registry import DatasetRegistry, DatasetTransform, IncrementalTransform
from .schemas import (
    BinarySample,
    MultiClassSample,
    AttributionSample,
    FineGrainedSample,
    ExperimentData,
    IncrementalData,
)

# Import transforms to trigger registration with DatasetRegistry
from . import transforms

# Legacy compatibility exports
from .dataloader import load_incremental, load_incremental_topic
