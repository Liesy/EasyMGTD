# Common utilities for experiment sub-modules.
# Provides shared helpers to eliminate code duplication across experiment classes.

import numpy as np
from dataclasses import dataclass, fields

from ..auto import BaseExperiment, DetectOutput


# ---------------------------------------------------------------------------
# BaseConfig: shared config base class with a generic update method
# ---------------------------------------------------------------------------


@dataclass
class BaseConfig:
    """
    Abstract base class for experiment configuration dataclasses.
    Provides a generic `update` method that sets attributes from a
    keyword-argument dictionary, only for fields defined in the dataclass.
    """

    def update(self, kargs):
        """
        Update configuration fields from a dictionary.

        Args:
            kargs (dict): Key-value pairs where keys correspond to
                          field names of this dataclass. Only matching
                          fields are updated; unknown keys are ignored.
        """
        for field in fields(self):
            if field.name in kargs:
                setattr(self, field.name, kargs[field.name])


# ---------------------------------------------------------------------------
# init_detectors: unified detector initialization pattern
# ---------------------------------------------------------------------------


def init_detectors(detector, detector_class):
    """
    Normalize detector input to a list and validate.

    If a single detector instance is passed, wraps it in a list.
    If a list (or other iterable) is passed, uses it as-is.
    Raises ValueError if the result is empty.

    Args:
        detector: A single detector instance or a list of detectors.
        detector_class: The expected class for isinstance check.

    Returns:
        list: A non-empty list of detector instances.

    Raises:
        ValueError: If the resulting detector list is empty.
    """
    detectors = [detector] if isinstance(detector, detector_class) else detector
    if not detectors:
        raise ValueError("You should pass a list of detector to an experiment")
    return detectors


# ---------------------------------------------------------------------------
# load_incremental_data: shared load_data for incremental-style experiments
# ---------------------------------------------------------------------------


def load_incremental_data(experiment, data):
    """
    Load incremental/staged data format into an experiment instance.

    Uses the last stage of the data as the standard train/test sets.
    Used by IncrementalExperiment, IncrementalThresholdExperiment,
    and FewShotExperiment.

    Args:
        experiment (BaseExperiment): The experiment instance to populate.
        data (dict): Dictionary with 'train' and 'test' keys, each containing
                     a list of stage data dicts with 'text' and 'label' keys.
    """
    experiment.loaded = True
    experiment.data = data
    experiment.train_text = data["train"][-1]["text"]
    experiment.train_label = data["train"][-1]["label"]
    experiment.test_text = data["test"][-1]["text"]
    experiment.test_label = data["test"][-1]["label"]


# ---------------------------------------------------------------------------
# build_supervised_output: shared output construction for supervised experiments
# ---------------------------------------------------------------------------


def build_supervised_output(experiment, num_labels, pair=None, intermedia=None):
    """
    Construct prediction output from either a (text, label) pair or
    intermediate results. Used by IncrementalExperiment and FewShotExperiment.

    For binary classifiers (num_labels == 2):
        Prediction = 1 if softmax probability >= 0.5 else 0.
    For multi-class:
        Prediction = first column of output (argmax).

    Args:
        experiment (BaseExperiment): The experiment instance (for data_prepare).
        num_labels (int): Number of output labels from the detector model.
        pair (tuple, optional): (text_list, label_list) to run detection on.
            When provided, calls detector.detect on the text.
            The first element of the pair must be detectable text, and
            detection is triggered by the caller before passing.
        intermedia (tuple, optional): Pre-computed (predictions, labels).

    Returns:
        tuple: (labels, predicted_labels, prediction_scores)

    Raises:
        ValueError: If neither pair nor intermedia is provided.
    """
    if not pair and not intermedia:
        raise ValueError(
            "At least one text or intermedia should be given for prediction"
        )
    if not intermedia:
        # pair is (predictions_raw, labels) - caller already ran detect()
        inter_preds, inter_labels = experiment.data_prepare(pair[0], pair[1])
    else:
        inter_preds, inter_labels = experiment.data_prepare(*intermedia)

    print("Run classification for results")
    if num_labels == 2:
        y_inter_preds = np.where(inter_preds[:, 0] >= 0.5, 1, 0)
        inter_preds = [x for x in inter_preds.flatten().tolist()]
    else:
        y_inter_preds = inter_preds[:, 0]

    return inter_labels, y_inter_preds, inter_preds


# ---------------------------------------------------------------------------
# launch_with_dual_predictions: shared launch logic for threshold/perturb
# ---------------------------------------------------------------------------


def launch_with_dual_predictions(experiment, **config):
    """
    Launch method for experiments that may produce dual predictions
    (threshold + logistic regression). Used by ThresholdExperiment
    and PerturbExperiment.

    Detects the prediction format:
    - Dual: train_pred/test_pred are tuples of two result sets ->
      produces two DetectOutput objects (name='threshold' and 'logistic').
    - Single: train_pred/test_pred are single result tuples ->
      produces one DetectOutput (name='logistic').

    Args:
        experiment (BaseExperiment): The experiment instance.
        **config: Configuration passed to experiment.predict().

    Returns:
        list[DetectOutput]: Evaluation results.

    Raises:
        RuntimeError: If data has not been loaded yet.
    """
    if not experiment.loaded:
        raise RuntimeError("You should load the data first, call load_data.")
    print("Calculate result for each data point")
    predict_list = experiment.predict(**config)
    final_output = []
    for detector_predict in predict_list:
        train_pred = detector_predict["train_pred"]
        test_pred = detector_predict["test_pred"]

        # Dual-prediction format: each is a tuple of two result sets
        is_dual = (
            isinstance(train_pred, tuple)
            and len(train_pred) == 2
            and isinstance(train_pred[0], tuple)
        )

        if is_dual:
            # Two classification methods: threshold + logistic
            train_metric1 = experiment.cal_metrics(*train_pred[0])
            test_metric1 = experiment.cal_metrics(*test_pred[0])
            train_metric2 = experiment.cal_metrics(*train_pred[1])
            test_metric2 = experiment.cal_metrics(*test_pred[1])
            final_output.append(
                DetectOutput(name="threshold", train=train_metric1, test=test_metric1)
            )
            final_output.append(
                DetectOutput(name="logistic", train=train_metric2, test=test_metric2)
            )
        else:
            # Single classification method (logistic regression only)
            train_metric = experiment.cal_metrics(*train_pred)
            test_metric = experiment.cal_metrics(*test_pred)
            final_output.append(
                DetectOutput(name="logistic", train=train_metric, test=test_metric)
            )
    return final_output
