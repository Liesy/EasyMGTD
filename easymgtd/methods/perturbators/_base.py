# Abstract base classes for perturbation strategies.
#
# BasePerturbator: top-level abstract base.
# TextPerturbator: text-in, text-out perturbation (mask-filling, truncate-regen).
# LogitsPerturbator: logits-in, score-out perturbation (log-prob sampling).
#
# TextPerturbator and LogitsPerturbator subclasses can be injected into
# detectors via the `perturbator` constructor argument to replace the
# default perturbation strategy.

from abc import ABC, abstractmethod
import torch


class BasePerturbator(ABC):
    """Abstract base for all perturbation strategies.

    All perturbation implementations must inherit from one of
    the two concrete sub-bases: TextPerturbator or LogitsPerturbator.
    """

    @abstractmethod
    def perturb(self, *args, **kwargs):
        """Execute the perturbation strategy.

        Signature varies by sub-base:
        - TextPerturbator:   perturb(texts, config) -> list[str]
        - LogitsPerturbator: perturb(logits_ref, logits_score, labels) -> float
        """
        raise NotImplementedError


class TextPerturbator(BasePerturbator):
    """Abstract base for text-level perturbation (text in, text out).

    Subclasses of TextPerturbator can be injected into detectors that
    perform text-level perturbation (DetectGPT, NPR, DNA-GPT) to replace
    the default perturbation strategy.

    Users who want to implement a custom text perturbation strategy
    (e.g. using BERT, GLM, or prompt-based LLM rewriting) should
    subclass TextPerturbator and implement the `perturb` method.
    """

    @abstractmethod
    def perturb(self, texts: list[str], config, **kwargs) -> list[str]:
        """Perturb a batch of texts.

        Args:
            texts: Input text strings to perturb.
            config: Perturbation configuration object (e.g. PerturbConfig)
                    providing parameters like span_length, pct_words_masked, etc.
            **kwargs: Additional strategy-specific parameters.

        Returns:
            list[str]: Perturbed text strings. Must have the same length
                       as the input `texts` list.
        """
        raise NotImplementedError


class LogitsPerturbator(BasePerturbator):
    """Abstract base for logits-level perturbation (logits in, score out).

    Subclasses of LogitsPerturbator can be injected into detectors that
    operate on model logits (Fast-DetectGPT) to replace the default
    sampling discrepancy computation.

    This is a pure computation layer: it does NOT hold any model references.
    The detector is responsible for running forward passes on scoring and
    reference models, then passing the resulting logits to this perturbator.

    Users who want to implement a custom logits-level perturbation strategy
    should subclass LogitsPerturbator and implement the `perturb` method.
    """

    @abstractmethod
    def perturb(
        self,
        logits_ref: torch.Tensor,
        logits_score: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute a perturbation-based discrepancy score.

        Args:
            logits_ref: Reference model logits, shape [1, seq_len, vocab_size].
            logits_score: Scoring model logits, shape [1, seq_len, vocab_size].
            labels: Ground-truth next-token ids, shape [1, seq_len].

        Returns:
            float: A scalar discrepancy score. Higher values typically
                   indicate machine-generated text.
        """
        raise NotImplementedError
