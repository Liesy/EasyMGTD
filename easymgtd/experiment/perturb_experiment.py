# Perturbation-based zero-shot detection experiment.
# Uses perturbation-based detectors (detectGPT, NPR, fast-detectGPT, DNA-GPT)
# and classifies via threshold search and/or logistic regression.

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression

from ..auto import BaseExperiment
from ..methods import PerturbBasedDetector
from ._base import BaseConfig, init_detectors, launch_with_dual_predictions


@dataclass
class PerturbConfig(BaseConfig):
    """
    Configuration for perturbation-based detection experiments.

    Attributes:
        span_length (int): Length of masked spans. Default 2.
        buffer_size (int): Buffer size around masked spans. Default 1.
        mask_top_p (float): Top-p sampling threshold for mask filling. Default 1.0.
        pct_words_masked (float): Fraction of words to mask. Default 0.3.
        DEVICE (int): GPU device index. Default 0.
        random_fills (bool): Whether to use random fills instead of model fills. Default False.
        random_fills_tokens (bool): Whether to use random token fills. Default False.
        n_perturbation_rounds (int): Number of perturbation rounds. Default 1.
        n_perturbations (int): Number of perturbations per round. Default 10.
        criterion_score (str): Scoring criterion, e.g. 'z' for z-score. Default 'z'.
        seed (int): Random seed for reproducibility. Default 0.
    """
    span_length: int = 2
    buffer_size: int = 1
    mask_top_p: float = 1
    pct_words_masked: float = 0.3
    DEVICE: int = 0
    random_fills: bool = False
    random_fills_tokens: bool = False
    n_perturbation_rounds: int = 1
    n_perturbations: int = 10
    criterion_score: str = "z"
    seed: int = 0


class PerturbExperiment(BaseExperiment):
    """
    Experiment for perturbation-based zero-shot detectors.

    Supported detectors: detectGPT, NPR, fast-detectGPT, DNA-GPT.

    Classification strategies:
    - Threshold: finds optimal threshold (higher score = more machine-like).
    - Logistic regression: trains a LogisticRegression on detection scores.
    - For all supported detectors, both strategies are applied.
    """

    _ALLOWED_detector = ["detectGPT", "NPR", "fast-detectGPT", "DNA-GPT"]

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = init_detectors(detector, PerturbBasedDetector)
        self.perturb_config = PerturbConfig()

    def launch(self, **config):
        """
        Execute the full experiment pipeline: predict -> classify -> evaluate.

        Delegates to the shared launch_with_dual_predictions helper.

        Returns:
            list[DetectOutput]: Evaluation results.
        """
        return launch_with_dual_predictions(self, **config)

    def predict(self, **kargs):
        """
        Run perturbation-based detection and classification for all detectors.

        For each detector:
        1. Apply perturbation config, then run detect on train/test data.
        2. For NPR/fast-detectGPT/DNA-GPT/detectGPT:
           - Find optimal threshold (higher score = machine-generated).
           - Train logistic regression as second classifier.
           - Return both result sets.
        3. Fallback: logistic regression only.

        Returns:
            list[dict]: Each dict has 'train_pred' and 'test_pred' keys.
        """
        predict_list = []
        for detector in self.detector:
            print(f"Running prediction of detector {detector.name}")
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, "is not for it")
                continue

            self.perturb_config.update(kargs)
            print("Predict training data")
            x_train, y_train = self.data_prepare(
                detector.detect(self.train_text, self.train_label, self.perturb_config),
                self.train_label,
            )
            print("Predict testing data")
            x_test, y_test = self.data_prepare(
                detector.detect(self.test_text, self.test_label, self.perturb_config),
                self.test_label,
            )
            print("Run classification for results")

            # All supported perturbation detectors use threshold + logistic
            if detector.name in ["NPR", "fast-detectGPT", "DNA-GPT", "detectGPT"]:
                print("Using threshold criterion")
                detector.find_threshold(x_train, y_train)
                y_train_preds = [x > detector.threshold for x in x_train]
                y_test_preds = [x > detector.threshold for x in x_test]
                train_result1 = y_train, y_train_preds, x_train
                test_result1 = y_test, y_test_preds, x_test

                # Also train logistic regression
                print("Using logistic regression")
                clf = LogisticRegression(random_state=0).fit(x_train, y_train)
                train_result2 = self.run_clf(clf, x_train, y_train)
                test_result2 = self.run_clf(clf, x_test, y_test)

                predict_list.append(
                    {
                        "train_pred": (train_result1, train_result2),
                        "test_pred": (test_result1, test_result2),
                    }
                )

            else:
                clf = LogisticRegression(random_state=0).fit(x_train, y_train)
                train_result = self.run_clf(clf, x_train, y_train)
                test_result = self.run_clf(clf, x_test, y_test)

                predict_list.append(
                    {"train_pred": train_result, "test_pred": test_result}
                )

        return predict_list
