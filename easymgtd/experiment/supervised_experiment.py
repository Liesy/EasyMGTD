# Supervised fine-tuning detection experiment.
# Supports supervised detectors (OpenAI-D, ConDA, ChatGPT-D, LM-D, RADAR)
# with optional fine-tuning and binary/multi-class evaluation.

import numpy as np
from dataclasses import dataclass

from ..auto import BaseExperiment
from ..methods import SupervisedDetector
from ._base import BaseConfig, init_detectors


@dataclass
class SupervisedConfig(BaseConfig):
    """
    Configuration for supervised fine-tuning experiments.

    Attributes:
        need_finetune (bool): Whether to fine-tune the model. Default False.
        name (str): Experiment name identifier. Default ''.
        need_save (bool): Whether to save the fine-tuned model. Default True.
        batch_size (int): Training batch size. Default 16.
        pos_bit (int): Positive class label index. Default 1.
        epochs (int): Number of training epochs. Default 3.
        save_path (str): Directory for saving fine-tuned models. Default 'finetuned/'.
        gradient_accumulation_steps (int): Gradient accumulation steps. Default 1.
        lr (float): Learning rate. Default 5e-6.
        logging_steps (int): Steps between logging events. Default 30.
        weight_decay (float): Weight decay for optimizer. Default 0.01.
        save_total_limit (int): Maximum number of checkpoints to keep. Default 2.
        swanlab_project (str): SwanLab project name for logging. Default 'EasyMGTD'.
    """
    need_finetune: bool = False
    name: str = ""
    need_save: bool = True
    batch_size: int = 16
    pos_bit: int = 1
    epochs: int = 3
    save_path: str = "finetuned/"
    gradient_accumulation_steps: int = 1
    lr: float = 5e-6
    logging_steps: int = 30
    weight_decay: float = 0.01
    save_total_limit: int = 2
    swanlab_project: str = "EasyMGTD"


class SupervisedExperiment(BaseExperiment):
    """
    Experiment for supervised fine-tuning detectors.

    Supported detectors: OpenAI-D, ConDA, ChatGPT-D, LM-D, RADAR.

    Workflow:
    1. Optionally fine-tune the detector model on training data.
    2. Run inference on train/test sets.
    3. Classify using softmax threshold (binary) or argmax (multi-class).
    """

    _ALLOWED_detector = ["OpenAI-D", "ConDA", "ChatGPT-D", "LM-D", "RADAR"]

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = init_detectors(detector, SupervisedDetector)
        self.supervise_config = SupervisedConfig()

    def predict(self, **kargs):
        """
        Run supervised detection and evaluation for all detectors.

        Supports two modes controlled by the 'eval' keyword argument:
        - eval=True:  Only evaluate on the test set (for transfer/zero-shot eval).
        - eval=False: Evaluate on both train and test sets (default).

        For binary classifiers (num_labels == 2):
            Prediction = 1 if softmax probability >= 0.5 else 0.
        For multi-class:
            Prediction = argmax of output logits.

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
            if self.supervise_config.need_finetune:
                data_train = {"text": self.train_text, "label": self.train_label}
                detector.finetune(data_train, self.supervise_config)
                print("Fine-tune finished")

            is_eval = kargs.get("eval", False)
            if is_eval:
                # Eval-only mode: only evaluate test set
                print("Predict testing data")
                test_preds, test_labels = self.data_prepare(
                    detector.detect(self.test_text, disable_tqdm=disable_tqdm),
                    self.test_label,
                )
                print("Run classification for results")
                if detector.model.config.num_labels == 2:
                    y_test_pred = np.where(test_preds[:, 0] >= 0.5, 1, 0)
                    test_preds = [x for x in test_preds.flatten().tolist()]
                else:
                    y_test_pred = test_preds[:, 0]
                test_result = test_labels, y_test_pred, test_preds
                predict_list.append({"test_pred": test_result})

            else:
                # Full mode: evaluate both train and test sets
                print("Predict training data")
                train_preds, train_labels = self.data_prepare(
                    detector.detect(self.train_text, disable_tqdm=disable_tqdm),
                    self.train_label,
                )
                print("Predict testing data")
                test_preds, test_labels = self.data_prepare(
                    detector.detect(self.test_text, disable_tqdm=disable_tqdm),
                    self.test_label,
                )
                print("Run classification for results")

                if detector.model.config.num_labels == 2:
                    y_train_pred = np.where(train_preds[:, 0] >= 0.5, 1, 0)
                    y_test_pred = np.where(test_preds[:, 0] >= 0.5, 1, 0)
                    train_preds = [x for x in train_preds.flatten().tolist()]
                    test_preds = [x for x in test_preds.flatten().tolist()]
                else:
                    y_train_pred = train_preds[:, 0]
                    y_test_pred = test_preds[:, 0]

                train_result = train_labels, y_train_pred, train_preds
                test_result = test_labels, y_test_pred, test_preds
                predict_list.append(
                    {"train_pred": train_result, "test_pred": test_result}
                )
        return predict_list
