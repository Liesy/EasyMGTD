# Perturbation-based detection methods.
#
# This module contains detectors that use perturbation strategies to
# distinguish machine-generated text from human-written text.
#
# Detectors:
# - PerturbBasedDetector: base class for mask-filling detectors (DetectGPT, NPR)
# - DetectGPTDetector: uses log-likelihood curvature via text perturbation
# - NPRDetector: uses normalized perturbed rank via text perturbation
# - FastDetectGPTDetector: uses sampling discrepancy via logits perturbation
# - DNAGPTDetector: uses truncate-and-regen divergence
#
# Perturbation strategies are decoupled into the perturbators subpackage.
# Each detector holds a perturbator instance that can be replaced via the
# `perturbator` constructor argument, as long as the type constraint is met:
# - DetectGPT/NPR/DNA-GPT accept TextPerturbator
# - Fast-DetectGPT accepts LogitsPerturbator

from ..auto import BaseDetector
from ..methods import LLDetector, RankDetector
from ..loading import load_pretrained, load_pretrained_mask
from .perturbators import (
    TextPerturbator,
    LogitsPerturbator,
    T5SpanPerturbator,
    LogProbSamplingPerturbator,
    TruncateRegenPerturbator,
)
import numpy as np
from sklearn.metrics import f1_score, roc_curve
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression


class PerturbBasedDetector(BaseDetector):
    """Base class for text-perturbation-based detectors (DetectGPT, NPR).

    Holds a scoring model and a TextPerturbator instance for generating
    perturbed text variants. The perturbator can be replaced by passing
    a custom TextPerturbator via the `perturbator` keyword argument.

    If no perturbator is provided, a T5SpanPerturbator is created as
    the default using the `mask_model_name_or_path` argument.

    Args:
        name: Detector name identifier.
        **kargs:
            model_name_or_path (str): Path to the scoring model (required).
            perturbator (TextPerturbator): Custom perturbator instance (optional).
            mask_model_name_or_path (str): Path to T5 mask model (used if
                perturbator is not provided).
            ceil_pct (bool): Whether to ceil the masking percentage. Default False.
    """

    def __init__(self, name, **kargs) -> None:
        self.name = name
        model_name_or_path = kargs.get("model_name_or_path", None)
        perturbator = kargs.get("perturbator", None)
        mask_model_name_or_path = kargs.get("mask_model_name_or_path", None)

        if not model_name_or_path:
            raise ValueError("You should pass model_name_or_path, but None is given.")

        # Scoring model (always required)
        self.model, self.tokenizer = load_pretrained(model_name_or_path)

        # Perturbator: user-provided or default T5
        if perturbator is not None:
            if not isinstance(perturbator, TextPerturbator):
                raise TypeError(
                    f"perturbator must be a TextPerturbator instance, "
                    f"got {type(perturbator).__name__}"
                )
            self.perturbator = perturbator
        elif mask_model_name_or_path:
            mask_model, mask_tokenizer = load_pretrained_mask(mask_model_name_or_path)
            self.perturbator = T5SpanPerturbator(
                mask_model, mask_tokenizer, self.tokenizer
            )
        else:
            raise ValueError(
                "Must provide either a perturbator instance or mask_model_name_or_path."
            )

        self.ceil_pct = kargs.get("ceil_pct", False)
        self.classifier = LogisticRegression()

    def perturb_once(self, texts, perturb_config, chunk_size=20):
        """Perturb texts in chunks using the perturbator.

        Args:
            texts: List of text strings to perturb.
            perturb_config: Configuration object for perturbation parameters.
            chunk_size: Number of texts to process per chunk. Default 20.

        Returns:
            list[str]: Perturbed text strings.
        """
        outputs = []
        for i in tqdm(range(0, len(texts), chunk_size)):
            outputs.extend(
                self.perturbator.perturb(
                    texts[i : i + chunk_size],
                    perturb_config,
                    ceil_pct=self.ceil_pct,
                )
            )
        return outputs

    def perturb(self, text, label, n_perturbations, perturb_config):
        """Generate multiple perturbations for each text and organize results.

        Each text is duplicated n_perturbations times, perturbed independently,
        and optionally re-perturbed for n_perturbation_rounds.

        Args:
            text: List of original text strings.
            label: Corresponding labels.
            n_perturbations: Number of perturbations per text.
            perturb_config: Configuration (must have n_perturbation_rounds).

        Returns:
            dict: {'text': [...], 'label': [...], 'perturbed_text': [...], 'len': int}
        """
        p_text = self.perturb_once(
            [x for x in text for _ in range(n_perturbations)], perturb_config
        )

        for _ in range(perturb_config.n_perturbation_rounds - 1):
            try:
                p_text = self.perturb_once(p_text, perturb_config)
            except AssertionError:
                break

        assert (
            len(p_text) == len(text) * n_perturbations
        ), f"Expected {len(text) * n_perturbations} perturbed samples, got {len(p_text)}"
        data = {"text": [], "label": [], "perturbed_text": []}

        for idx in range(len(text)):
            data["text"].append(text[idx])
            data["label"].append(label[idx])
            data["perturbed_text"].extend(
                p_text[idx * n_perturbations : (idx + 1) * n_perturbations]
            )
        data["len"] = len(text)
        return data


class DetectGPTDetector(PerturbBasedDetector, LLDetector):
    """DetectGPT detector using log-likelihood curvature estimation.

    Perturbs text using a TextPerturbator (default: T5SpanPerturbator)
    and compares the log-likelihood of the original vs perturbed texts.

    Supports two scoring criteria:
    - 'd': raw difference (ll_original - ll_perturbed_mean)
    - 'z': z-score normalization by perturbed standard deviation
    """

    def __init__(self, name, **kargs) -> None:
        PerturbBasedDetector.__init__(self, name, **kargs)
        LLDetector.__init__(self, name, model=self.model, tokenizer=self.tokenizer)
        self.threshold = None

    def find_threshold(self, train_scores, train_labels):
        """Find the optimal threshold for f1 score on training data.

        Args:
            train_scores: Array of detection scores.
            train_labels: Ground-truth labels.

        Returns:
            float: Best threshold value.
        """
        print(f"Finding best threshold for f1 score...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_f1 = 0
        for threshold in thresholds:
            # machine's score is larger, human's score is smaller
            predictions = train_scores > threshold
            f1 = f1_score(train_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold = best_threshold
        return best_threshold

    def detect(self, text, label, config):
        """Run DetectGPT detection pipeline.

        1. Perturb all texts
        2. Compute log-likelihood for originals and perturbations
        3. Compute curvature score (d or z criterion)

        Args:
            text: List of text strings.
            label: Corresponding labels.
            config: PerturbConfig object.

        Returns:
            numpy.ndarray: Detection scores per text.
        """
        perturb_config = config
        print("Running perturb on the given texts")
        data = self.perturb(text, label, perturb_config.n_perturbations, perturb_config)
        print("Perturb finished.")
        p_ll_origin = LLDetector.detect(self, data["text"])
        p_ll_origin = np.array(p_ll_origin)
        p_ll = LLDetector.detect(self, data["perturbed_text"])
        perturbed_ll_mean = []
        perturbed_ll_std = []
        for batch in DataLoader(p_ll, batch_size=perturb_config.n_perturbations):
            batch = batch.numpy()
            perturbed_ll_mean.append(np.mean(batch))
            perturbed_ll_std.append(np.std(batch) if len(batch) > 1 else 1)
        assert len(p_ll_origin) == len(perturbed_ll_mean)
        if perturb_config.criterion_score == "d":
            predictions = p_ll_origin - perturbed_ll_mean
        elif perturb_config.criterion_score == "z":
            perturbed_ll_std = [std if std > 0 else 1 for std in perturbed_ll_std]
            predictions = (p_ll_origin - perturbed_ll_mean) / perturbed_ll_std
        return predictions


# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
class FastDetectGPTDetector(BaseDetector):
    """Fast-DetectGPT detector using sampling discrepancy.

    Uses a scoring model and a reference model to compute discrepancy
    between the expected and observed log-likelihoods via logits-level
    perturbation. The perturbation computation is delegated to a
    LogitsPerturbator (default: LogProbSamplingPerturbator).

    The perturbator can be replaced by passing a custom LogitsPerturbator
    via the `perturbator` keyword argument.

    Args:
        name: Detector name identifier.
        **kargs:
            scoring_model_name_or_path (str): Path to scoring model (required).
            reference_model_name_or_path (str): Path to reference model (optional,
                defaults to scoring model).
            perturbator (LogitsPerturbator): Custom perturbator (optional).
            discrepancy_analytic (bool): Use analytic mode. Default False.
    """

    def __init__(self, name, **kargs) -> None:
        self.name = name
        scoring_model_name_or_path = kargs.get("scoring_model_name_or_path", None)
        reference_model_name_or_path = kargs.get("reference_model_name_or_path", None)
        perturbator = kargs.get("perturbator", None)

        if not scoring_model_name_or_path:
            raise ValueError(
                "You should pass the scoring_model_name_or_path, but None is given."
            )

        # Scoring model
        self.scoring_model, self.scoring_tokenizer = load_pretrained(
            scoring_model_name_or_path
        )
        self.classifier = LogisticRegression()
        self.scoring_model.eval()

        # Reference model (optional, defaults to scoring model)
        if reference_model_name_or_path:
            if reference_model_name_or_path != scoring_model_name_or_path:
                self.reference_model, self.reference_tokenizer = load_pretrained(
                    reference_model_name_or_path
                )
                self.reference_model.eval()

        # Perturbator: user-provided or default LogProbSampling
        if perturbator is not None:
            if not isinstance(perturbator, LogitsPerturbator):
                raise TypeError(
                    f"perturbator must be a LogitsPerturbator instance, "
                    f"got {type(perturbator).__name__}"
                )
            self.perturbator = perturbator
        else:
            analytic = kargs.get("discrepancy_analytic", False)
            self.perturbator = LogProbSamplingPerturbator(analytic=analytic)

    def find_threshold(self, train_scores, train_labels):
        """Find the optimal threshold for f1 score on training data.

        Args:
            train_scores: Array of detection scores.
            train_labels: Ground-truth labels.

        Returns:
            float: Best threshold value.
        """
        print(f"Finding best threshold for f1 score...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_f1 = 0
        for threshold in thresholds:
            # machine's score is larger, human's score is smaller
            predictions = train_scores > threshold
            f1 = f1_score(train_labels, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold = best_threshold
        return best_threshold

    def detect(self, text, label=None, config=None):
        """Run Fast-DetectGPT detection pipeline.

        For each text:
        1. Tokenize and get scoring model logits
        2. Get reference model logits (or reuse scoring logits)
        3. Compute discrepancy score via perturbator

        Args:
            text: List of text strings.
            label: Corresponding labels (unused, kept for interface compat).
            config: Configuration (unused, kept for interface compat).

        Returns:
            list[float]: Detection scores per text.
        """
        seed = 3407
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        predictions = []

        for idx in tqdm(range(len(text)), desc="Detecting"):
            tokenized = self.scoring_tokenizer(
                text[idx],
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
            ).to(self.scoring_model.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                if not hasattr(
                    self, "reference_model"
                ):  # reference model == scoring model
                    logits_ref = logits_score
                else:
                    tokenized = self.reference_tokenizer(
                        text[idx],
                        return_tensors="pt",
                        padding=True,
                        return_token_type_ids=False,
                        truncation=True,
                    ).to(self.reference_model.device)
                    # Compare on CPU to avoid cross-device tensor comparison error
                    assert torch.all(
                        tokenized.input_ids[:, 1:].cpu() == labels.cpu()
                    ), "Tokenizer is mismatch."
                    logits_ref = self.reference_model(**tokenized).logits[:, :-1]

                # Ensure all tensors are on the same device for perturbator computation
                # (scoring and reference models may reside on different GPUs)
                compute_device = logits_score.device
                logits_ref = logits_ref.to(compute_device)
                labels = labels.to(compute_device)

                # Delegate discrepancy computation to the perturbator
                crit = self.perturbator.perturb(logits_ref, logits_score, labels)
                predictions.append(crit)

        return predictions


class NPRDetector(PerturbBasedDetector, RankDetector):
    """NPR (Normalized Perturbed Rank) detector.

    Perturbs text using a TextPerturbator (default: T5SpanPerturbator)
    and compares token ranks before and after perturbation.
    """

    def __init__(self, name, **kargs) -> None:
        PerturbBasedDetector.__init__(self, name, **kargs)
        RankDetector.__init__(self, name, model=self.model, tokenizer=self.tokenizer)

    def detect(self, text, label, config):
        """Run NPR detection pipeline.

        1. Perturb all texts
        2. Compute log-rank for originals and perturbations
        3. Compute normalized ratio (perturbed_rank / original_rank)

        Args:
            text: List of text strings.
            label: Corresponding labels.
            config: PerturbConfig object.

        Returns:
            numpy.ndarray: Detection scores per text.
        """
        perturb_config = config
        print("Running perturb on the given texts")
        data = self.perturb(text, label, perturb_config.n_perturbations, perturb_config)
        print("Perturb finished.")

        p_rank_origin = RankDetector.detect(self, data["text"], log=True)
        p_rank_origin = np.array(p_rank_origin)

        p_rank = RankDetector.detect(self, data["perturbed_text"], log=True)
        perturbed_rank_mean = []
        for batch in DataLoader(p_rank, batch_size=perturb_config.n_perturbations):
            batch = batch.numpy()
            perturbed_rank_mean.append(np.mean(batch))
        assert len(p_rank_origin) == len(perturbed_rank_mean)
        predictions = perturbed_rank_mean / p_rank_origin

        return predictions


# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
class DNAGPTDetector(BaseDetector):
    """DNA-GPT detector using truncate-and-regen divergence.

    Truncates text and regenerates the truncated portion using a causal LM,
    then compares the log-probability divergence between the original and
    regenerated text.

    The perturbation strategy is delegated to a TextPerturbator
    (default: TruncateRegenPerturbator). Can be replaced by passing a
    custom TextPerturbator via the `perturbator` keyword argument.

    Args:
        name: Detector name identifier.
        **kargs:
            perturbator (TextPerturbator): Custom perturbator (optional).
            base_model_name_or_path (str): Path to causal LM (used if
                perturbator is not provided). Default 'gpt2'.
            batch_size (int): Batch size for generation. Default 5.
            regen_number (int): Number of regen samples per text. Default 5.
            temperature (float): Sampling temperature. Default 1.0.
            truncate_ratio (float): Truncation ratio. Default 0.5.
            mode (str): Threshold mode ('accuracy' or 'low-fpr'). Default 'accuracy'.
    """

    def __init__(self, name, **kargs) -> None:
        self.name = "DNA-GPT"
        perturbator = kargs.get("perturbator", None)

        if perturbator is not None:
            if not isinstance(perturbator, TextPerturbator):
                raise TypeError(
                    f"perturbator must be a TextPerturbator instance, "
                    f"got {type(perturbator).__name__}"
                )
            self.perturbator = perturbator
            # When using a custom perturbator, we still need a base model
            # for log-probability computation. User must provide it.
            base_model_name_or_path = kargs.get("base_model_name_or_path", "gpt2")
            self.base_model, self.base_tokenizer = load_pretrained(
                model_name_or_path=base_model_name_or_path
            )
            self.base_model.eval()
        else:
            base_model_name_or_path = kargs.get("base_model_name_or_path", "gpt2")
            self.base_model, self.base_tokenizer = load_pretrained(
                model_name_or_path=base_model_name_or_path
            )
            self.base_model.eval()
            self.perturbator = TruncateRegenPerturbator(
                self.base_model,
                self.base_tokenizer,
                batch_size=kargs.get("batch_size", 5),
                regen_number=kargs.get("regen_number", 5),
                temperature=kargs.get("temperature", 1.0),
                truncate_ratio=kargs.get("truncate_ratio", 0.5),
            )

        self.mode = kargs.get("mode", "accuracy")

    def get_likelihood(self, logits, labels, pad_index):
        """Compute masked mean log-likelihood.

        Args:
            logits: Model logits, shape [batch, seq_len, vocab_size].
            labels: Token ids, shape [batch, seq_len].
            pad_index: Pad token id to exclude from averaging.

        Returns:
            torch.Tensor: Mean log-likelihood per sample.
        """
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        mask = labels != pad_index
        log_likelihood = (log_likelihood * mask).sum(dim=1) / mask.sum(dim=1)
        return log_likelihood.squeeze(-1)

    def get_log_prob(self, text):
        """Compute log-probability of a single text.

        Args:
            text: Input text string.

        Returns:
            torch.Tensor: Log-probability tensor.
        """
        tokenized = self.base_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self.base_model.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.base_model(**tokenized).logits[:, :-1]
            return self.get_likelihood(
                logits_score, labels, self.base_tokenizer.pad_token_id
            )

    def get_log_probs(self, texts):
        """Compute log-probabilities for a batch of texts.

        Args:
            texts: List of text strings.

        Returns:
            torch.Tensor: Concatenated log-probabilities.
        """
        batch_size = (
            self.perturbator.batch_size
            if hasattr(self.perturbator, "batch_size")
            else 5
        )
        batch_lprobs = []
        for batch in range(len(texts) // batch_size):
            tokenized = self.base_tokenizer(
                texts[batch * batch_size : (batch + 1) * batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.base_model.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = self.base_model(**tokenized).logits[:, :-1]
                lprobs = self.get_likelihood(
                    logits_score, labels, self.base_tokenizer.pad_token_id
                )
                batch_lprobs.append(lprobs)
        return torch.cat(batch_lprobs, dim=0)

    def get_regen_samples(self, text):
        """Generate regenerated samples for a single text via the perturbator.

        Args:
            text: Input text string.

        Returns:
            list[str]: Regenerated text samples.
        """
        regen_number = (
            self.perturbator.regen_number
            if hasattr(self.perturbator, "regen_number")
            else 5
        )
        data = [text] * regen_number
        return self.perturbator.perturb(data)

    def get_dna_gpt_score(self, text):
        """Compute DNA-GPT divergence score for a single text.

        Compares original log-probability against mean log-probability
        of regenerated samples.

        Args:
            text: Input text string.

        Returns:
            float: Divergence score. Higher = more likely machine-generated.
        """
        lprob = self.get_log_prob(text)
        regens = self.get_regen_samples(text)
        lprob_regens = self.get_log_probs(regens)
        wscore = lprob[0] - lprob_regens.mean()
        return wscore.item()

    def find_threshold(self, train_scores, train_labels):
        """Find the optimal threshold for the configured mode.

        Args:
            train_scores: Array of detection scores.
            train_labels: Ground-truth labels.

        Returns:
            tuple: (best_threshold, best_accuracy)
        """
        print(f"Finding best threshold for {self.mode}...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_accuracy = 0
        if self.mode == "low-fpr":
            scores = train_scores
            fpr, tpr, roc_thresholds = roc_curve(train_labels, scores)
            target_fpr = 0.01
            idx = np.where(fpr <= target_fpr)[0][-1]
            self.threshold = roc_thresholds[idx]

        elif self.mode == "accuracy":
            for t in thresholds:
                predictions = train_scores > t
                accuracy = f1_score(train_labels, predictions)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = t

            self.threshold = best_threshold
        return best_threshold, best_accuracy

    def change_mode(self, mode):
        """Change the threshold finding mode.

        Args:
            mode: 'accuracy' or 'low-fpr'.

        Raises:
            ValueError: If mode is not supported.
        """
        if mode not in ["low-fpr", "accuracy"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    def detect(self, text, label, config):
        """Run DNA-GPT detection pipeline.

        Args:
            text: List of text strings.
            label: Corresponding labels (unused, kept for interface compat).
            config: Configuration (unused, kept for interface compat).

        Returns:
            list[float]: Detection scores per text.
        """
        predictions = []
        for idx in tqdm(range(len(text)), desc="Detecting"):
            dna_gpt_score = self.get_dna_gpt_score(text[idx])
            predictions.append(dna_gpt_score)
        return predictions
