# Few-shot learning detection experiment.
# Supports few-shot adaptation of detectors with eval modes.

import numpy as np
from dataclasses import dataclass

from ..auto import BaseExperiment
from ..methods import FewShotDetector
from ._base import (
    BaseConfig,
    init_detectors,
    load_incremental_data,
    build_supervised_output,
)


@dataclass
class FewShotConfig(BaseConfig):
    """
    Configuration for few-shot learning experiments.

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
        kshot (int): Number of examples per class for few-shot. Default 5.
        classifier (str): Classifier type to use. Default 'SVM'.
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
    kshot: int = 5
    classifier: str = "SVM"


class FewShotExperiment(BaseExperiment):
    """
    Experiment for few-shot learning detection.

    Supported detectors: baseline, generate, rn.

    Workflow:
    1. Run few-shot fine-tuning/adaptation via detector.finetune().
    2. Evaluate using softmax threshold (binary) or argmax (multi-class).
    3. Supports eval-only mode for testing without training set evaluation.
    """

    _ALLOWED_detector = ["baseline", "generate", "rn"]

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        # BUG FIX: original code incorrectly used IncrementalDetector for isinstance check
        self.detector = init_detectors(detector, FewShotDetector)
        self.supervise_config = FewShotConfig()
        self.kshot = None

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
        helper. Passes num_labels from detector.model directly.

        Args:
            detector: The detector instance.
            pair (tuple, optional): (raw_detect_output, label_list).
                If provided, detector.detect() is called internally.
            intermedia (tuple, optional): Pre-computed (predictions, labels).

        Returns:
            tuple: (labels, predicted_labels, prediction_scores)
        """
        # For FewShotDetector, num_labels is accessed directly via model
        num_labels = detector.model.num_labels
        if pair:
            # Run detection first, then pass raw output to build_supervised_output
            raw_output = detector.detect(pair[0], disable_tqdm=True)
            return build_supervised_output(self, num_labels, pair=(raw_output, pair[1]))
        return build_supervised_output(self, num_labels, intermedia=intermedia)

    def predict(self, **kargs):
        """
        Run few-shot detection and evaluation for all detectors.

        Supports two modes:
        - eval=True:  Only evaluate test set.
        - eval=False: Evaluate both train and test sets.

        Returns:
            list[dict]: Each dict has 'test_pred' (and optionally 'train_pred').
        """
        predict_list = []
        disable_tqdm = kargs.get("disable_tqdm", False)
        for detector in self.detector:
            print(f"Running prediction of detector {detector.name}")
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, "is not for it")
                continue
            self.supervise_config.update(kargs)
            detector.finetune(self.data, self.supervise_config)
            print("Fine-tune finished")
            is_eval = kargs.get("eval", False)

            if is_eval:
                print("Predict testing data")
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
