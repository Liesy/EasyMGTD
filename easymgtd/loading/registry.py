"""
Transform base classes and DatasetRegistry for the EasyMGTD data loading system.

Provides:
    - DatasetTransform: base class for standard dataset transforms
    - IncrementalTransform: base class for incremental/multi-stage transforms
    - DatasetRegistry: central registry with @register decorator and load() entry point
"""

from abc import ABC, abstractmethod
from typing import Type, Union


class DatasetTransform(ABC):
    """
    Base class for standard dataset transforms.

    Subclasses must:
    1. Set `output_schema` to the sample dataclass (BinarySample, etc.)
    2. Implement `transform()` to convert raw data to schema-conforming samples

    Example:
        @DatasetRegistry.register("MyDataset")
        class MyTransform(DatasetTransform):
            output_schema = BinarySample

            def transform(self, raw_data, *, targetLLM, **kwargs):
                samples = []
                for row in raw_data:
                    samples.append(BinarySample(text=row["human"], label=0))
                    samples.append(BinarySample(text=row[targetLLM], label=1))
                return samples
    """

    output_schema = None  # Must be set by subclasses

    @abstractmethod
    def transform(self, raw_data: list[dict], **kwargs) -> list:
        """
        Transform raw records into schema-conforming samples.

        Args:
            raw_data: List of raw record dicts (from readers.py).
                      May be empty if the transform has custom loading logic.
            **kwargs: Dataset-specific parameters (targetLLM, category, etc.)

        Returns:
            List of sample dataclass instances matching output_schema.
        """
        raise NotImplementedError


class IncrementalTransform(ABC):
    """
    Base class for incremental/multi-stage dataset transforms.

    These transforms produce IncrementalData (multi-stage train/test)
    and bypass the standard split pipeline.

    Subclasses must implement `build()` to construct the complete
    multi-stage data structure.
    """

    @abstractmethod
    def build(self, **kwargs) -> dict:
        """
        Build the complete incremental data structure.

        Returns:
            dict compatible with IncrementalExperiment.load_data(),
            i.e., {"train": [stage_dict, ...], "test": [stage_dict, ...]}.
        """
        raise NotImplementedError


class DatasetRegistry:
    """
    Central registry for all dataset transforms.

    Usage:
        # Register a transform:
        @DatasetRegistry.register("MyDataset")
        class MyTransform(DatasetTransform):
            ...

        # Load data:
        data = DatasetRegistry.load("MyDataset", targetLLM="gpt35", seed=3407)

        # List available datasets:
        DatasetRegistry.list_datasets()
    """

    _registry: dict[str, Type[Union[DatasetTransform, IncrementalTransform]]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a dataset transform.

        Args:
            name: Unique name for the dataset.

        Returns:
            Decorator function.

        Raises:
            ValueError: If name is already registered.
        """

        def decorator(transform_cls):
            if name in cls._registry:
                raise ValueError(
                    f"Dataset '{name}' is already registered by "
                    f"{cls._registry[name].__name__}. "
                    f"Cannot register {transform_cls.__name__}."
                )
            cls._registry[name] = transform_cls
            return transform_cls

        return decorator

    @classmethod
    def list_datasets(cls) -> list[str]:
        """List all registered dataset names."""
        return list(cls._registry.keys())

    @classmethod
    def get_transform(cls, name: str) -> Union[DatasetTransform, IncrementalTransform]:
        """
        Get a transform instance by dataset name.

        Args:
            name: Registered dataset name.

        Returns:
            An instance of the registered transform class.

        Raises:
            ValueError: If name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown dataset: '{name}'. "
                f"Available datasets: {cls.list_datasets()}"
            )
        return cls._registry[name]()

    @classmethod
    def load(
        cls,
        name: str,
        *,
        seed: int = 0,
        split_ratio: float = 0.8,
        cache_path: str = None,
        **kwargs,
    ) -> dict:
        """
        Load a dataset by name, applying the registered transform
        and the appropriate pipeline.

        For standard transforms (DatasetTransform):
            1. Read raw data from file (if path is provided)
            2. Apply transform to get schema-conforming samples
            3. Run build_experiment_data() for split/shuffle/cache

        For incremental transforms (IncrementalTransform):
            1. Delegate directly to build()

        Args:
            name: Registered dataset name.
            seed: Random seed for shuffling and splitting.
            split_ratio: Fraction of data for training (rest goes to test).
            cache_path: Optional path for JSON cache.
            **kwargs: Passed to transform.transform() or transform.build().
                      Common kwargs: targetLLM, category, path, repo, etc.

        Returns:
            dict compatible with BaseExperiment.load_data() or
            IncrementalExperiment.load_data().
        """
        transform = cls.get_transform(name)

        if isinstance(transform, IncrementalTransform):
            return transform.build(seed=seed, **kwargs)

        # Standard transform path
        from .pipeline import build_experiment_data

        # Extract path if provided, read raw data
        path = kwargs.pop("path", None)
        raw_data = []
        if path:
            from .readers import read_file

            fmt = kwargs.pop("format", "auto")
            raw_data = read_file(path, format=fmt)

        # Pass seed to transform kwargs (some transforms use it internally)
        kwargs["seed"] = seed
        samples = transform.transform(raw_data, **kwargs)

        return build_experiment_data(
            samples,
            seed=seed,
            split_ratio=split_ratio,
            cache_path=cache_path,
        )
