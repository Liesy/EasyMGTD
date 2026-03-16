# Demasq-specific detection experiment.
# Dedicated experiment for the Demasq detector with optional fine-tuning.

from dataclasses import dataclass

from ..auto import BaseExperiment
from ..methods import DemasqDetector
from ._base import BaseConfig, init_detectors


@dataclass
class DemasqConfig(BaseConfig):
    """
    Configuration for Demasq detection experiments.

    Attributes:
        need_finetune (bool): Whether to fine-tune the model. Default True.
        need_save (bool): Whether to save model weights. Default True.
        batch_size (int): Training batch size. Default 1.
        save_path (str): Directory for saving model weights. Default 'model_weight/'.
        epoch (int): Number of training epochs. Default 12.
    """
    need_finetune: bool = True
    need_save: bool = True
    batch_size: int = 1
    save_path: str = "model_weight/"
    epoch: int = 12


class DemasqExperiment(BaseExperiment):
    """
    Experiment for the Demasq detector.

    Supported detectors: demasq.

    Workflow:
    1. Optionally fine-tune the Demasq model on training data.
    2. Run inference to get logits.
    3. Classify using 0.5 threshold on logits.
    """

    _ALLOWED_detector = ["demasq"]

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = init_detectors(detector, DemasqDetector)
        self.config = DemasqConfig()

    def predict(self, **kargs):
        """
        Run Demasq detection and classification for all detectors.

        For each detector:
        1. If need_finetune=True, fine-tune on training data.
        2. Run inference on train and test data to get logits.
        3. Apply 0.5 threshold: logit > 0.5 -> positive class.

        Returns:
            list[dict]: Each dict has 'train_pred' and 'test_pred' keys.
        """
        predict_list = []
        for detector in self.detector:
            print(f"Running prediction of detector {detector.name}")
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, "is not for it")
                continue
            self.config.update(kargs)
            if self.config.need_finetune:
                data_train = {"text": self.train_text, "label": self.train_label}
                detector.finetune(data_train, self.config)

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
