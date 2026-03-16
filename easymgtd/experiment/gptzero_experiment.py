# GPTZero API detection experiment.
# Handles GPTZero-based detection without any fine-tuning.

from ..auto import BaseExperiment
from ..methods import GPTZeroDetector
from ._base import init_detectors


class GPTZeroExperiment(BaseExperiment):
    """
    Experiment for GPTZero API-based detection.

    Supported detectors: GPTZero.

    Workflow:
    1. Run inference via GPTZero API on train and test data.
    2. Classify using 0.5 threshold on returned logits.

    Note: No fine-tuning is involved in this experiment.
    """

    _ALLOWED_detector = ["GPTZero"]

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        # BUG FIX: original code incorrectly used DemasqDetector for isinstance check
        self.detector = init_detectors(detector, GPTZeroDetector)

    def predict(self, **kargs):
        """
        Run GPTZero detection and classification for all detectors.

        For each detector:
        1. Run inference on train and test data.
        2. Apply 0.5 threshold: logit > 0.5 -> positive class.

        Returns:
            list[dict]: Each dict has 'train_pred' and 'test_pred' keys.
        """
        predict_list = []
        for detector in self.detector:
            print(f"Running prediction of detector {detector.name}")
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, "is not for _ALLOWED_detector")
                continue

            logits = detector.detect(self.train_text)
            preds = [1 if logit > 0.5 else 0 for logit in logits]
            logits_t = detector.detect(self.test_text)
            preds_t = [1 if logit > 0.5 else 0 for logit in logits_t]
            predict_list.append(
                {
                    "train_pred": (self.train_label, preds, logits),
                    "test_pred": (self.test_label, preds_t, logits_t),
                }
            )
        return predict_list
