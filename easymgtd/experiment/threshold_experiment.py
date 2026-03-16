# Threshold-based zero-shot detection experiment.
# Uses metric-based detectors (ll, rank, LRR, rank_GLTR, entropy, Binoculars)
# and classifies via optimal threshold search and/or logistic regression.

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..auto import BaseExperiment
from ..methods import MetricBasedDetector
from ._base import init_detectors, launch_with_dual_predictions


class ThresholdExperiment(BaseExperiment):
    """
    Experiment for metric-based zero-shot detectors.

    Supported detectors: ll, rank, LRR, rank_GLTR, entropy, Binoculars.

    Classification strategies:
    - Threshold: finds optimal threshold on training scores.
    - Logistic regression: trains a LogisticRegression on training scores.
    - For threshold-capable detectors, both strategies are applied and
      two DetectOutput objects are returned (threshold + logistic).
    - For rank_GLTR, only logistic regression is used.
    """

    _ALLOWED_detector = [
        "ll",
        "rank",
        "LRR",
        "rank_GLTR",
        "entropy",
        "Binoculars",
        "tdt",
    ]

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = init_detectors(detector, MetricBasedDetector)

    def launch(self, **config):
        """
        Execute the full experiment pipeline: predict -> classify -> evaluate.

        Delegates to the shared launch_with_dual_predictions helper,
        which handles both dual-prediction (threshold + logistic) and
        single-prediction (logistic only) formats.

        Returns:
            list[DetectOutput]: A list of DetectOutput objects, each containing
                                train/test Metric results.
        """
        return launch_with_dual_predictions(self, **config)

    def predict(self, **config):
        """
        Run detection and classification for all detectors.

        For each detector:
        1. Extract detection scores on train/test data.
        2. For threshold-capable detectors (Binoculars, rank, ll, LRR, entropy):
           - Find optimal threshold and produce threshold-based predictions.
           - Train logistic regression and produce LR-based predictions.
           - Return both as a tuple of two result sets.
        3. For rank_GLTR: only logistic regression.

        Returns:
            list[dict]: Each dict has 'train_pred' and 'test_pred' keys.
        """
        predict_list = []
        for detector in self.detector:
            print(f"Running prediction of detector {detector.name}")
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, "is not for this experiment")
                continue
            if detector.name in ["rank_GLTR"]:
                # rank_GLTR returns multi-dimensional output, no data_prepare needed
                print("Predict training data")
                x_train, y_train = detector.detect(self.train_text), self.train_label
                x_train = np.array(x_train)
                y_train = np.array(y_train)
                print("Predict testing data")
                x_test, y_test = detector.detect(self.test_text), self.test_label
                x_test = np.array(x_test)
                y_test = np.array(y_test)
            else:
                # Scalar score detectors: reshape via data_prepare
                print("Predict training data")
                x_train, y_train = self.data_prepare(
                    detector.detect(self.train_text), self.train_label
                )
                print("Predict testing data")
                x_test, y_test = self.data_prepare(
                    detector.detect(self.test_text), self.test_label
                )

            print("Run classification for results")

            # Detectors that support threshold-based classification
            if detector.name in ["Binoculars", "rank", "ll", "LRR", "entropy", "tdt"]:
                print("Using threshold criterion")
                detector.find_threshold(x_train, y_train)
                # Direction distinction:
                # rank/LRR/entropy/Binoculars/tdt: lower score = more machine-like
                # ll: higher score = more machine-like
                if detector.name in ["rank", "LRR", "entropy", "Binoculars", "tdt"]:
                    y_train_preds = [x < detector.threshold for x in x_train]
                    y_test_preds = [x < detector.threshold for x in x_test]
                    train_result1 = (
                        y_train,
                        y_train_preds,
                        -1 * x_train,
                    )  # human has higher score
                    test_result1 = y_test, y_test_preds, -1 * x_test

                elif detector.name in ["ll"]:
                    y_train_preds = [x > detector.threshold for x in x_train]
                    y_test_preds = [x > detector.threshold for x in x_test]
                    train_result1 = y_train, y_train_preds, x_train
                    test_result1 = y_test, y_test_preds, x_test

                # Also train logistic regression as second classification method
                print("Using logistic regression")
                clf = LogisticRegression(random_state=0).fit(
                    np.clip(x_train, -1e10, 1e10), y_train
                )
                train_result2 = self.run_clf(clf, x_train, y_train)
                test_result2 = self.run_clf(clf, x_test, y_test)

                predict_list.append(
                    {
                        "train_pred": (train_result1, train_result2),
                        "test_pred": (test_result1, test_result2),
                    }
                )
            else:
                # rank_GLTR: logistic regression only
                clf = LogisticRegression(random_state=0).fit(x_train, y_train)
                train_result = self.run_clf(clf, x_train, y_train)
                test_result = self.run_clf(clf, x_test, y_test)

                predict_list.append(
                    {"train_pred": train_result, "test_pred": test_result}
                )

        return predict_list
