"""
Data schemas for the EasyMGTD data loading system.

Defines sample-level schemas (output units of Transform) and
dataset-level schemas (output of Pipeline consumed by Experiments).

5 Schema Types:
    - BinarySample: human vs. machine binary classification
    - MultiClassSample: multi-class classification with optional metadata
    - AttributionSample: source model attribution
    - FineGrainedSample: sentence-level annotation within a document
    - IncrementalData: multi-stage incremental learning data
"""

from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Sample-level schemas (Transform output units)
# ============================================================


@dataclass
class BinarySample:
    """
    Single sample for binary (human vs. machine) detection.

    Attributes:
        text: The text content.
        label: 0 for human-written, 1 for machine-generated.
    """

    text: str
    label: int  # 0=human, 1=machine

    def __post_init__(self):
        if self.label not in (0, 1):
            raise ValueError(f"BinarySample label must be 0 or 1, got {self.label}")


@dataclass
class MultiClassSample:
    """
    Single sample for multi-class classification.

    Attributes:
        text: The text content.
        label: Integer class label (0..N).
        category: Optional metadata (e.g., topic, subject).
    """

    text: str
    label: int  # 0..N
    category: Optional[str] = None


@dataclass
class AttributionSample:
    """
    Single sample for source model attribution.

    Attributes:
        text: The text content.
        label: 0 for Human, 1..N for model-specific labels.
    """

    text: str
    label: int  # 0=Human, 1..N=model-specific


@dataclass
class FineGrainedSample:
    """
    Document with sentence-level annotations.

    A full text is composed of multiple sentences, each with its own label.
    This supports fine-grained detection where different parts of a document
    may have different origins (human vs. machine).

    Attributes:
        text: Full document text.
        sentences: Individual sentences that compose the document.
        sentence_labels: Per-sentence label (e.g., 0=human, 1=machine).
        label: Optional document-level label.
    """

    text: str
    sentences: list[str]
    sentence_labels: list[int]
    label: Optional[int] = None

    def __post_init__(self):
        if len(self.sentences) != len(self.sentence_labels):
            raise ValueError(
                f"sentences ({len(self.sentences)}) and "
                f"sentence_labels ({len(self.sentence_labels)}) must have same length"
            )


# ============================================================
# Dataset-level schemas (Pipeline output, consumed by Experiments)
# ============================================================


@dataclass
class SplitData:
    """
    One split (train or test) of standard experiment data.

    Attributes:
        text: List of text samples.
        label: List of integer labels corresponding to each text.
    """

    text: list[str] = field(default_factory=list)
    label: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dict format expected by BaseExperiment.load_data()."""
        return {"text": self.text, "label": self.label}


@dataclass
class ExperimentData:
    """
    Standard experiment data with train/test split.

    This is the primary output format consumed by BaseExperiment.load_data().

    Attributes:
        train: Training split data.
        test: Test split data.
    """

    train: SplitData = field(default_factory=SplitData)
    test: SplitData = field(default_factory=SplitData)

    def to_dict(self) -> dict:
        """Convert to dict format: {"train": {"text": [], "label": []}, "test": {...}}."""
        return {"train": self.train.to_dict(), "test": self.test.to_dict()}


@dataclass
class FineGrainedSplitData:
    """
    One split of fine-grained experiment data with sentence-level annotations.

    Attributes:
        text: List of full document texts.
        label: List of document-level labels.
        sentences: List of sentence lists (one per document).
        sentence_labels: List of per-sentence label lists (one per document).
    """

    text: list[str] = field(default_factory=list)
    label: list[int] = field(default_factory=list)
    sentences: list[list[str]] = field(default_factory=list)
    sentence_labels: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dict format with sentence-level fields."""
        return {
            "text": self.text,
            "label": self.label,
            "sentences": self.sentences,
            "sentence_labels": self.sentence_labels,
        }


@dataclass
class FineGrainedExperimentData:
    """
    Fine-grained experiment data with train/test split and sentence-level annotations.

    Attributes:
        train: Training split data with sentence-level annotations.
        test: Test split data with sentence-level annotations.
    """

    train: FineGrainedSplitData = field(default_factory=FineGrainedSplitData)
    test: FineGrainedSplitData = field(default_factory=FineGrainedSplitData)

    def to_dict(self) -> dict:
        """Convert to dict format."""
        return {"train": self.train.to_dict(), "test": self.test.to_dict()}


@dataclass
class StageData:
    """
    One stage of incremental learning data.

    Attributes:
        text: List of text samples for this stage.
        label: List of labels for this stage.
    """

    text: list[str] = field(default_factory=list)
    label: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dict format."""
        return {"text": self.text, "label": self.label}


@dataclass
class IncrementalData:
    """
    Multi-stage incremental learning data.

    Each stage contains a train/test split. Test sets accumulate
    data from previous stages.

    Attributes:
        train: List of per-stage training data.
        test: List of per-stage test data.
    """

    train: list[StageData] = field(default_factory=list)
    test: list[StageData] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dict format: {"train": [stage_dict, ...], "test": [...]}."""
        return {
            "train": [s.to_dict() for s in self.train],
            "test": [s.to_dict() for s in self.test],
        }
