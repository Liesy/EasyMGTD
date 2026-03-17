# Log-probability based sampling perturbation strategy.
#
# Migrated from easymgtd/methods/perturb.py FastDetectGPTDetector.
# Implements discrepancy computation between reference and scoring model
# logits via conditional sampling (Fast-DetectGPT style).
#
# This is a pure computation layer: it does NOT hold any model references.
# The detector is responsible for running model forward passes and passing
# the resulting logits tensors to this perturbator.

import torch

from ._base import LogitsPerturbator


class LogProbSamplingPerturbator(LogitsPerturbator):
    """Log-probability sampling discrepancy computation (Fast-DetectGPT style).

    Given reference-model logits and scoring-model logits for the same text,
    computes a discrepancy score by comparing the log-likelihood of the
    original tokens against sampled alternatives.

    Supports two modes:
    - Empirical sampling ('discrepancy'): draws 10000 samples from the
      reference distribution and estimates mean/std.
    - Analytic ('discrepancy_analytic'): computes mean/variance analytically
      by enumerating over the vocabulary.

    Args:
        analytic: If True, use analytic discrepancy (default False).
    """

    def __init__(self, analytic=False):
        self.analytic = analytic

    def perturb(self, logits_ref, logits_score, labels):
        """Compute sampling discrepancy between reference and scoring logits.

        Args:
            logits_ref: Reference model logits, shape [1, seq_len, vocab_size].
            logits_score: Scoring model logits, shape [1, seq_len, vocab_size].
            labels: Ground-truth next-token ids, shape [1, seq_len].

        Returns:
            float: Discrepancy score. Higher = more likely machine-generated.
        """
        if self.analytic:
            return self._discrepancy_analytic(logits_ref, logits_score, labels)
        else:
            return self._discrepancy_sampling(logits_ref, logits_score, labels)

    def _get_samples(self, logits, labels):
        """Sample alternative token sequences from the reference distribution.

        Args:
            logits: Reference model logits, shape [1, seq_len, vocab_size].
            labels: Original token ids, shape [1, seq_len].

        Returns:
            torch.Tensor: Sampled token ids, shape [1, seq_len, nsamples].
        """
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        nsamples = 10000
        lprobs = torch.log_softmax(logits, dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs)
        samples = distrib.sample([nsamples]).permute([1, 2, 0])
        return samples

    def _get_likelihood(self, logits, labels):
        """Compute mean log-likelihood of given labels under the logits.

        Args:
            logits: Model logits, shape [1, seq_len, vocab_size].
            labels: Token ids to evaluate, shape [1, seq_len] or [1, seq_len, N].

        Returns:
            torch.Tensor: Mean log-likelihood, shape [1] or [1, N].
        """
        assert logits.shape[0] == 1
        assert labels.shape[0] == 1
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs = torch.log_softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=labels)
        return log_likelihood.mean(dim=1)

    def _discrepancy_sampling(self, logits_ref, logits_score, labels):
        """Empirical sampling discrepancy (original Fast-DetectGPT method).

        Draws samples from the reference distribution and compares the
        log-likelihood of original tokens vs sampled alternatives.

        Copyright (c) Guangsheng Bao.

        Args:
            logits_ref: Reference model logits, shape [1, seq_len, vocab_size].
            logits_score: Scoring model logits, shape [1, seq_len, vocab_size].
            labels: Ground-truth token ids, shape [1, seq_len].

        Returns:
            float: Discrepancy score.
        """
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1

        # Handle vocabulary size mismatch between models
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        samples = self._get_samples(logits_ref, labels)
        log_likelihood_x = self._get_likelihood(logits_score, labels)
        log_likelihood_x_tilde = self._get_likelihood(logits_score, samples)
        miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
        sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
        discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
        return discrepancy.item()

    def _discrepancy_analytic(self, logits_ref, logits_score, labels):
        """Analytic discrepancy computation (closed-form Fast-DetectGPT).

        Computes mean and variance analytically by enumerating over the
        entire vocabulary, avoiding the need for sampling.

        Copyright (c) Guangsheng Bao.

        Args:
            logits_ref: Reference model logits, shape [1, seq_len, vocab_size].
            logits_score: Scoring model logits, shape [1, seq_len, vocab_size].
            labels: Ground-truth token ids, shape [1, seq_len].

        Returns:
            float: Discrepancy score.
        """
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1

        # Handle vocabulary size mismatch between models
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = (
            labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        )
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)

        # Log-likelihood of the original next-token prediction
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)

        # Expected log-likelihood under the reference distribution
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)

        # Variance of log-likelihood under the reference distribution
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(
            mean_ref
        )

        # Normalized discrepancy
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(
            dim=-1
        ).sqrt()
        discrepancy = discrepancy.mean()
        return discrepancy.item()
