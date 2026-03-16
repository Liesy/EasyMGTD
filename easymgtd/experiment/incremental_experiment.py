# Supervised incremental learning detection experiment.
# Supports staged training with optional fine-tuning and eval modes.

import numpy as np
from dataclasses import dataclass

from ..auto import BaseExperiment
from ..methods import IncrementalDetector
from ._base import (
    BaseConfig,
    init_detectors,
    load_incremental_data,
    build_supervised_output,
)


@dataclass
class IncrementalConfig(BaseConfig):
    """
    Configuration for incremental learning experiments.

    Attributes:
        need_finetune (bool): Whether to fine-tune the model. Default False.
        name (str): Experiment name identifier. Default ''.
        need_save (bool): Whether to save the fine-tuned model. Default True.
        batch_size (int): Training batch size. Default 16.
        pos_bit (int): Positive class label index. Default 1.
        epochs (int): Number of training epochs. Default 1.
        save_path (str): Directory for saving models. Default 'finetuned/'.
        gradient_accumulation_steps (int): Gradient accumulation steps. Default 1.
        lr (float): Learning rate. Default 5e-6.
        lr_factor (int): Learning rate scaling factor. Default 5.
    """

    need_finetune: bool = False
    name: str = ""
    need_save: bool = True
    batch_size: int = 16
    pos_bit: int = 1
    epochs: int = 1
    save_path: str = "finetuned/"
    gradient_accumulation_steps: int = 1
    lr: float = 5e-6
    lr_factor: int = 5


class IncrementalExperiment(BaseExperiment):
    """
    Experiment for supervised incremental learning detection.

    Supported detectors: incremental.

    Workflow:
    1. Optionally fine-tune the detector incrementally on staged data.
    2. Evaluate using softmax threshold (binary) or argmax (multi-class).
    3. Supports eval-only mode for testing without fine-tuning.
    """

    _ALLOWED_detector = ["incremental"]

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = init_detectors(detector, IncrementalDetector)
        self.supervise_config = IncrementalConfig()

    def load_data(self, data):
        """
        Load incremental data, using the last stage as train/test sets.

        Args:
            data (dict): Dictionary with 'train' and 'test' keys, each
                         containing a list of stage data dicts with
                         'text' and 'label' keys.
        """
        load_incremental_data(self, data)

    def return_output(self, detector, pair=None, intermedia=None):
        """
        Construct prediction output using the shared build_supervised_output
        helper. Passes num_labels from detector.model.pretrained.

        Args:
            detector: The detector instance.
            pair (tuple, optional): (raw_detect_output, label_list).
                If provided, detector.detect() is called internally.
            intermedia (tuple, optional): Pre-computed (predictions, labels).

        Returns:
            tuple: (labels, predicted_labels, prediction_scores)
        """
        # For IncrementalDetector, num_labels is accessed via model.pretrained
        num_labels = detector.model.pretrained.num_labels
        if pair:
            # Run detection first, then pass raw output to build_supervised_output
            raw_output = detector.detect(pair[0], disable_tqdm=True)
            return build_supervised_output(self, num_labels, pair=(raw_output, pair[1]))
        return build_supervised_output(self, num_labels, intermedia=intermedia)

    def predict(self, **kargs):
        """
        Run incremental detection and evaluation for all detectors.

        Supports two modes:
        - eval=True:  Evaluate intermediate results (if fine-tuned) and test set.
        - eval=False: Evaluate train and test sets.

        Returns:
            list[dict]: Each dict has combination of 'intermedia_pred',
                        'train_pred', and/or 'test_pred' keys.
        """
        predict_list = []
        disable_tqdm = kargs.get("disable_tqdm", False)
        for detector in self.detector:
            print(f"Running prediction of detector {detector.name}")
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, "is not for it")
                continue
            self.supervise_config.update(kargs)
            intermedia = None
            if self.supervise_config.need_finetune:
                intermedia = detector.finetune(self.data, self.supervise_config)
                print("Fine-tune finished")
            print(detector.model.use_bic)
            is_eval = kargs.get("eval", False)
            if is_eval:
                print("Predict testing data")
                if intermedia:
                    predict_list.append(
                        {
                            "intermedia_pred": self.return_output(
                                detector, intermedia=intermedia
                            )
                        }
                    )
                predict_list.append(
                    {
                        "test_pred": self.return_output(
                            detector, pair=(self.test_text, self.test_label)
                        )
                    }
                )

            else:
                predict_list.append(
                    {
                        "train_pred": self.return_output(
                            detector, pair=(self.trian_text, self.train_label)
                        )
                    }
                )
                predict_list.append(
                    {
                        "test_pred": self.return_output(
                            detector, pair=(self.test_text, self.test_label)
                        )
                    }
                )
        return predict_list
