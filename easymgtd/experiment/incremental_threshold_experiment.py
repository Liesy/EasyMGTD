# Incremental threshold-based detection experiment.
# Combines zero-shot metric detectors with incremental learning via
# logistic regression classifiers and exemplar-based replay.

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from ..auto import BaseExperiment
from ..methods import IncrementalDetector
from ..loading.model_loader import load_pretrained_supervise
from ._base import init_detectors, load_incremental_data


class IncrementalThresholdExperiment(BaseExperiment):
    """
    Experiment for zero-shot detection in an incremental learning setting.

    Uses metric-based detectors combined with a logistic regression classifier
    that is incrementally expanded as new classes are introduced.

    Supported detectors: ll, rank, LRR, rank_GLTR, entropy, Binoculars, fast-detectGPT.

    Key features:
    - Exemplar construction: uses RoBERTa [CLS] embeddings to select
      representative samples closest to class centroids.
    - Classifier expansion: dynamically adds new output dimensions to
      the logistic regression classifier for new classes.
    - Stage-by-stage training: processes data in stages, merging with
      exemplars from previous stages.
    """

    _ALLOWED_detector = [
        "ll",
        "rank",
        "LRR",
        "rank_GLTR",
        "entropy",
        "Binoculars",
        "fast-detectGPT",
    ]

    def __init__(self, detector, **kargs) -> None:
        super().__init__()
        self.detector = init_detectors(detector, IncrementalDetector)
        self.model, self.tokenizer = load_pretrained_supervise(
            "/data1/models/roberta-base", kargs
        )
        self.cache_size = kargs.get("cache_size", 0)
        print(self.cache_size)

    def load_data(self, data):
        """
        Load incremental data, using the last stage as train/test sets.

        Args:
            data (dict): Dictionary with 'train' and 'test' keys, each
                         containing a list of stage data dicts.
        """
        load_incremental_data(self, data)

    def get_dataset(self, stage_data, exampler=None, return_exampler=False):
        """
        Merge current stage data with exemplars from previous stages.

        Args:
            stage_data (dict): Current stage data with 'text' and 'label'.
            exampler (dict, optional): Previous stage exemplars.
            return_exampler (bool): Whether to construct and return new exemplars.

        Returns:
            tuple: (merged_data, exemplar_dict_or_None)
        """
        if exampler:
            stage_data["text"] = list(stage_data["text"]) + list(exampler["text"])
            stage_data["label"] = list(stage_data["label"]) + list(exampler["label"])
        if return_exampler and self.cache_size != 0:
            print("construct the exampler for current class")
            exampler_idx = self.construct_exampler(
                stage_data, cache_size=self.cache_size
            )
            exampler = {
                "text": np.array(stage_data["text"])[exampler_idx],
                "label": np.array(stage_data["label"])[exampler_idx],
            }
            print(f"Get exampler of {len(exampler_idx)} training data")
        return stage_data, exampler

    def construct_exampler(self, stage_data, cache_size=100):
        """
        Construct exemplar set by selecting samples closest to class centroids.

        Process:
        1. Extract [CLS] embeddings from RoBERTa for each text sample.
        2. Compute class-wise mean embeddings (centroids).
        3. For each class, rank samples by cosine distance to centroid.
        4. Select top `cache_size` closest samples per class.

        Args:
            stage_data (dict): Data with 'text' and 'label' keys.
            cache_size (int): Number of exemplar samples to keep per class.

        Returns:
            list[int]: Indices of selected exemplar samples.
        """
        features = []
        labels = []
        print(len(stage_data["text"]), len(stage_data["label"]))
        for data in tqdm(stage_data["text"]):
            encoding = self.tokenizer(
                data, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(**encoding.to("cuda"), output_hidden_states=True)
                cls_embedding = outputs.hidden_states[-1][:, 0, :]
            features.append(cls_embedding.cpu().squeeze().numpy())
        labels = stage_data["label"]
        features = np.array(features)
        from sklearn.metrics.pairwise import cosine_distances

        # Step 2: Compute class centroids
        class_means = {}
        for label in np.unique(labels):
            class_features = features[labels == label]
            class_mean = np.mean(class_features, axis=0)
            class_means[label] = class_mean

        # Step 3: Compute distances to the class mean for each sample
        class_top_100 = []

        for label in np.unique(labels):
            # Get the indices and embeddings of samples in the current class
            class_indices = np.where(labels == label)[0]
            class_embeddings = features[class_indices]

            # Calculate distance from each sample to the class mean
            distances = []
            for i, embedding in zip(class_indices, class_embeddings):
                distance = cosine_distances([embedding], [class_means[label]])[0][0]
                distances.append((i, distance))

            # Sort by distance and select top-N closest samples
            distances.sort(key=lambda x: x[1])
            top_100_for_class = distances[:cache_size]

            # Store the top-N samples for the current class
            class_top_100.extend([i[0] for i in top_100_for_class])
        return class_top_100

    def increment_classes(self, detector, new_classes):
        """
        Expand the logistic regression classifier to accommodate new classes.

        Adds zero-initialized weight rows and intercept entries for the
        new classes, preserving existing learned parameters.

        Args:
            detector: Detector instance with a `classifier` attribute
                      (LogisticRegression).
            new_classes (int): Number of new classes to add.
        """
        clf = detector.classifier
        old_coef = clf.coef_
        old_intercept = clf.intercept_

        new_coef = np.zeros((new_classes, old_coef.shape[1]))
        new_intercept = np.zeros(new_classes)

        # Combine old and new parameters
        expanded_coef = np.vstack([old_coef, new_coef])
        expanded_intercept = np.hstack([old_intercept, new_intercept])

        # Set new parameters in a fresh Logistic Regression model
        clf.classes_ = np.arange(clf.classes_.size + new_classes)
        clf.coef_ = expanded_coef
        clf.intercept_ = expanded_intercept
        detector.classifier = clf

    def predict(self, **kargs):
        """
        Run incremental threshold-based detection across all stages.

        For each detector and each training stage:
        1. Merge current stage data with exemplars.
        2. If not the first stage, expand classifier dimensions.
        3. Extract detection scores (multi-dim for rank_GLTR, scalar for others).
        4. Train logistic regression classifier on current stage.
        5. Evaluate on corresponding test set.

        Returns:
            list[dict]: Each dict has 'test_pred' key with evaluation results.
        """
        predict_list = []
        disable_tqdm = kargs.get("disable_tqdm", False)
        for detector in self.detector:
            print(f"Running prediction of detector {detector.name}")
            if detector.name not in self._ALLOWED_detector:
                print(detector.name, "is not for it")
                continue
            stages = self.data["train"]
            eval_set = self.data["test"]
            exampler = None
            for idx, stage_data in enumerate(stages):
                train_dataset, exampler = self.get_dataset(
                    stage_data, exampler=exampler, return_exampler=True
                )
                test_dataset, _ = self.get_dataset(
                    eval_set[idx], exampler=None, return_exampler=False
                )
                if idx != 0:
                    unique_elements = set(stage_data["label"])
                    num_newclass = len(unique_elements)
                    self.increment_classes(detector, num_newclass)
                    print(detector.classifier.coef_.shape)

                if detector.name in ["rank_GLTR"]:
                    # rank_GLTR returns multi-dimensional output
                    print("Predict training data")
                    x_train, y_train = (
                        detector.detect(train_dataset["text"]),
                        train_dataset["label"],
                    )
                    x_train = np.array(x_train)
                    y_train = np.array(y_train)
                    print("Predict testing data")
                    x_test, y_test = (
                        detector.detect(test_dataset["text"]),
                        test_dataset["label"],
                    )
                    x_test = np.array(x_test)
                    y_test = np.array(y_test)
                else:
                    # Scalar score detectors
                    print("Predict training data")
                    print((train_dataset["text"][0]))
                    x_train, y_train = self.data_prepare(
                        detector.detect(train_dataset["text"]), train_dataset["label"]
                    )
                    print("Predict testing data")
                    x_test, y_test = self.data_prepare(
                        detector.detect(test_dataset["text"]), test_dataset["label"]
                    )
                detector.classifier.fit(x_train, y_train)
                test_result = self.run_clf(detector.classifier, x_test, y_test)
                predict_list.append({"test_pred": test_result})

        return predict_list
