import numpy as np
import torch
import pywt
from scipy.stats import gaussian_kde
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..auto import BaseDetector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve


def transform_discrete_sequence(discrepancy_sequence, bandwidth="scott"):
    if len(discrepancy_sequence) == 0:
        return np.zeros(1000)
    if len(discrepancy_sequence) == 1:
        t = np.linspace(0, 1, 1000)
        signal = np.exp(-0.5 * ((t - 0.5) / 0.1) ** 2) * discrepancy_sequence[0]
        return signal
    kde = gaussian_kde(discrepancy_sequence, bw_method=bandwidth)
    t = np.linspace(0, len(discrepancy_sequence), 1000)
    continuous_signal = kde(t)
    return continuous_signal


def get_wavelet_features(continuous_signal, wavelet="cmor1.5-1.0", max_scales=12):
    if np.all(continuous_signal == 0) or len(continuous_signal) == 0:
        return [0.0, 0.0, 0.0]
    scales = np.arange(1, max_scales + 1)
    try:
        coeffs, freqs = pywt.cwt(continuous_signal, scales, wavelet)
    except Exception as e:
        print(f"Wavelet transform error: {e}")
        return [0.0, 0.0, 0.0]
    energy = np.sqrt(np.sum(np.abs(coeffs) ** 2, axis=1))
    signal_strength = np.std(continuous_signal)
    morph_energy = np.mean(energy[0:4])
    syn_energy = np.mean(energy[4:8])
    disc_energy = np.mean(energy[8:12])
    eps = 1e-6
    features = [
        float(np.log(morph_energy * signal_strength + eps)),
        float(np.log(syn_energy * signal_strength + eps)),
        float(np.log(disc_energy * signal_strength + eps)),
    ]
    return features


def get_t_discrepancy_analytic(
    logits_ref,
    logits_score,
    labels,
    nu=5,
    extract_wavelet_features=False,
    return_details=False,
):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(
        mean_ref
    )

    var_ref = torch.clamp(var_ref, min=1e-6)
    scale = torch.sqrt(var_ref * nu / (nu - 2))

    token_discrepancies_raw = (log_likelihood - mean_ref).squeeze(0)
    token_var = var_ref.squeeze(0)

    t_discrepancy_scalar = (
        log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)
    ) / scale.sum(dim=-1)
    t_discrepancy_scalar = t_discrepancy_scalar.mean().item()

    token_discrepancies = (
        (token_discrepancies_raw / torch.sqrt(token_var)).detach().cpu().numpy()
    )
    token_discrepancies = np.nan_to_num(
        token_discrepancies, nan=0.0, posinf=10.0, neginf=-10.0
    )
    continuous_signal = transform_discrete_sequence(token_discrepancies)
    scales = np.arange(1, 13)
    wavelet_coeffs, _ = pywt.cwt(continuous_signal, scales, "cmor1.5-1.0")

    if extract_wavelet_features:
        wavelet_features = get_wavelet_features(continuous_signal)
        scale_factor = abs(t_discrepancy_scalar) / (
            np.mean(np.abs(wavelet_features)) + 1e-6
        )
        wavelet_features = [f * scale_factor for f in wavelet_features]
        if return_details:
            details = {
                "token_discrepancies": token_discrepancies,
                "continuous_signal": continuous_signal,
                "wavelet_coeffs": wavelet_coeffs,
                "wavelet_features": wavelet_features,
                "t_discrepancy_scalar": t_discrepancy_scalar,
            }
            return wavelet_features, details
        return wavelet_features
    if return_details:
        details = {
            "token_discrepancies": token_discrepancies,
            "continuous_signal": continuous_signal,
            "wavelet_coeffs": wavelet_coeffs,
            "wavelet_features": [
                t_discrepancy_scalar,
                t_discrepancy_scalar,
                t_discrepancy_scalar,
            ],
            "t_discrepancy_scalar": t_discrepancy_scalar,
        }
        return t_discrepancy_scalar, details
    return t_discrepancy_scalar


class TDTDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name)
        ref_model_name_or_path = kargs.get(
            "reference_model_name_or_path", "tiiuae/falcon-7b"
        )
        scoring_model_name_or_path = kargs.get(
            "scoring_model_name_or_path", "tiiuae/falcon-7b-instruct"
        )

        self.DEVICE_1 = (
            torch.device("cuda:0")
            if torch.cuda.device_count() > 0
            else torch.device("cpu")
        )
        self.DEVICE_2 = (
            torch.device("cuda:1") if torch.cuda.device_count() > 1 else self.DEVICE_1
        )

        self.scoring_tokenizer = AutoTokenizer.from_pretrained(
            scoring_model_name_or_path
        )
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            scoring_model_name_or_path,
            device_map=self.DEVICE_2,
            torch_dtype=torch.bfloat16,
        )

        if ref_model_name_or_path != scoring_model_name_or_path:
            self.reference_tokenizer = AutoTokenizer.from_pretrained(
                ref_model_name_or_path
            )
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                ref_model_name_or_path,
                device_map=self.DEVICE_1,
                torch_dtype=torch.bfloat16,
            )
        else:
            self.reference_tokenizer = self.scoring_tokenizer
            self.reference_model = self.scoring_model

        if not self.scoring_tokenizer.pad_token:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
        if not self.reference_tokenizer.pad_token:
            self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token

        self.scoring_model.eval()
        self.reference_model.eval()

        self.max_length = kargs.get("max_length", 512)
        self.extract_wavelet_features = kargs.get("extract_wavelet_features", False)
        self.criterion_fn = get_t_discrepancy_analytic

    def compute_crit(self, text, return_details=False):
        tokenized_score = self.scoring_tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.scoring_model.device)
        labels = tokenized_score.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized_score).logits[:, :-1]
            if self.reference_model is self.scoring_model:
                logits_ref = logits_score
            else:
                tokenized_ref = self.reference_tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).to(self.reference_model.device)
                logits_ref = self.reference_model(**tokenized_ref).logits[:, :-1]

            result = self.criterion_fn(
                logits_ref,
                logits_score,
                labels,
                extract_wavelet_features=self.extract_wavelet_features,
                return_details=return_details,
            )
        return result

    def detect(self, text, **kargs):
        predictions = []
        if isinstance(text, str):
            text = [text]
        for idx in tqdm(range(len(text)), desc="Detecting TDT"):
            score = self.compute_crit(text[idx])
            predictions.append(score)

        return predictions[0] if len(predictions) == 1 else predictions

    def find_threshold(self, train_scores, train_labels):
        print("Finding best threshold for f1...")
        thresholds = np.sort(train_scores)
        best_threshold = None
        best_accuracy = 0
        for t in thresholds:
            # According to TDT text detects AI text is lower discrepancy? Let's check.
            # Usually we try both or assume < t. Let's do < t.
            predictions = (train_scores < t).astype(int)
            accuracy = f1_score(train_labels, predictions)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t
        self.threshold = best_threshold
        return best_threshold, best_accuracy
