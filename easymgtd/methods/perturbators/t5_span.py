# T5-based span mask-and-fill perturbation strategy.
#
# Migrated from easymgtd/methods/perturb.py top-level functions.
# Implements the T5-specific <extra_id_N> masking and filling protocol
# used by DetectGPT and NPR detectors.

import re
import time
import torch
import random
import numpy as np

from ._base import TextPerturbator

# Regex to match all <extra_id_*> tokens, where * is an integer.
# This is specific to the T5 tokenizer's span corruption format.
_EXTRA_ID_PATTERN = re.compile(r"<extra_id_\d+>")


def _load_mask_model_to_gpu(mask_model, device, random_fills):
    """Move the mask-filling model to the specified device.

    Args:
        mask_model: The mask-filling model (e.g. T5).
        device: Target device (e.g. 'cuda:0').
        random_fills: If True, skip moving the model (not needed).
    """
    print("MOVING MASK MODEL TO GPU...", end="", flush=True)
    start = time.time()
    if not random_fills:
        mask_model.to(device)
    print(f"DONE ({time.time() - start:.2f}s)")


def _tokenize_and_mask(text, span_length, buffer_size, pct, ceil_pct=False):
    """Replace random spans in text with T5-style <extra_id_N> tokens.

    Args:
        text: Input text string (space-separated).
        span_length: Number of consecutive words per masked span.
        buffer_size: Minimum gap between masked spans (words).
        pct: Fraction of words to mask.
        ceil_pct: If True, round up the number of masked spans.

    Returns:
        str: Text with masked spans replaced by <extra_id_N> tokens.
    """
    tokens = text.split(" ")

    # Truncate to avoid exceeding typical model length limits
    if len(tokens) > 1024:
        tokens = tokens[:1024]
    mask_string = "<<<mask>>>"

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # Replace each <<<mask>>> with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = " ".join(tokens)
    return text


def _count_masks(texts):
    """Count the number of <extra_id_*> tokens in each text.

    Args:
        texts: List of text strings.

    Returns:
        list[int]: Number of mask tokens per text.
    """
    return [
        len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts
    ]


def _replace_masks(texts, mask_model, mask_tokenizer, mask_top_p, device):
    """Replace each masked span with a sample from the T5 mask model.

    Args:
        texts: List of texts containing <extra_id_*> tokens.
        mask_model: T5-family model for mask filling.
        mask_tokenizer: Corresponding tokenizer.
        mask_top_p: Top-p sampling threshold for generation.
        device: Device to run generation on.

    Returns:
        list[str]: Raw model outputs containing fill tokens.
    """
    n_expected = _count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(device)
    outputs = mask_model.generate(
        **tokens,
        max_length=150,
        do_sample=True,
        top_p=mask_top_p,
        num_return_sequences=1,
        eos_token_id=stop_id,
    )
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def _extract_fills(texts):
    """Extract the fill text between <extra_id_*> tokens in model output.

    Args:
        texts: Raw model output strings containing <extra_id_*> delimiters.

    Returns:
        list[list[str]]: For each text, a list of fill strings.
    """
    # Remove <pad> and </s> tokens
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]
    # Split on <extra_id_*> tokens and take content between them
    extracted_fills = [_EXTRA_ID_PATTERN.split(x)[1:-1] for x in texts]
    # Strip whitespace from each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills


def _apply_extracted_fills(masked_texts, extracted_fills):
    """Apply extracted fills back into masked texts.

    Args:
        masked_texts: Texts with <extra_id_*> placeholders.
        extracted_fills: Fill strings for each text.

    Returns:
        list[str]: Texts with fills applied. Empty string if fill count
                   doesn't match mask count.
    """
    # Split on spaces (not newlines)
    tokens = [x.split(" ") for x in masked_texts]
    n_expected = _count_masks(masked_texts)

    # Replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # Join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


class T5SpanPerturbator(TextPerturbator):
    """Mask-and-fill perturbation using T5-family models.

    This perturbator randomly masks spans in the input text using T5-style
    <extra_id_N> tokens and uses a T5 model to generate fill text.

    Also supports two fallback modes (for ablation studies):
    - random_fills=True, random_fills_tokens=True: replace with random tokens.
    - random_fills=True, random_fills_tokens=False: replace with random words
      from a fill dictionary.

    Args:
        mask_model: T5-family model for span filling.
        mask_tokenizer: Corresponding tokenizer.
        tokenizer: Scoring model tokenizer (only needed for random_fills_tokens mode).
    """

    def __init__(self, mask_model, mask_tokenizer, tokenizer=None):
        self.mask_model = mask_model
        self.mask_tokenizer = mask_tokenizer
        self.tokenizer = tokenizer  # scoring model tokenizer, for random_fills_tokens

    def perturb(self, texts, config, ceil_pct=False, **kwargs):
        """Perturb texts by masking and filling spans using the T5 model.

        Args:
            texts: List of input text strings.
            config: Configuration object with attributes:
                - span_length (int): Length of masked spans.
                - buffer_size (int): Min gap between spans.
                - mask_top_p (float): Top-p for generation sampling.
                - pct_words_masked (float): Fraction of words to mask.
                - DEVICE: Device for model inference.
                - random_fills (bool): Use random fills instead of model.
                - random_fills_tokens (bool): Use random token-level fills.
            ceil_pct: If True, round up the number of spans.

        Returns:
            list[str]: Perturbed texts, same length as input.
        """
        span_length = config.span_length
        buffer_size = config.buffer_size
        mask_top_p = config.mask_top_p
        pct = config.pct_words_masked
        DEVICE = config.DEVICE

        if not config.random_fills:
            # Standard T5 mask-and-fill path
            masked_texts = [
                _tokenize_and_mask(x, span_length, buffer_size, pct, ceil_pct)
                for x in texts
            ]
            raw_fills = _replace_masks(
                masked_texts, self.mask_model, self.mask_tokenizer, mask_top_p, DEVICE
            )
            extracted_fills = _extract_fills(raw_fills)
            perturbed_texts = _apply_extracted_fills(masked_texts, extracted_fills)

            # Retry loop: sometimes the model doesn't generate the right
            # number of fills, so we need to try again for those texts
            attempts = 1
            while "" in perturbed_texts:
                idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]
                masked_texts = [
                    _tokenize_and_mask(x, span_length, pct, ceil_pct)
                    for idx, x in enumerate(texts)
                    if idx in idxs
                ]
                raw_fills = _replace_masks(
                    masked_texts,
                    self.mask_model,
                    self.mask_tokenizer,
                    mask_top_p,
                    DEVICE,
                )
                extracted_fills = _extract_fills(raw_fills)
                new_perturbed_texts = _apply_extracted_fills(
                    masked_texts, extracted_fills
                )
                for idx, x in zip(idxs, new_perturbed_texts):
                    perturbed_texts[idx] = x
                attempts += 1
        else:
            if config.random_fills_tokens:
                # Token-level random replacement (ablation)
                tokenizer = self.tokenizer
                tokens = tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
                valid_tokens = tokens.input_ids != tokenizer.pad_token_id
                replace_pct = config.pct_words_masked * (
                    config.span_length / (config.span_length + 2 * config.buffer_size)
                )

                random_mask = (
                    torch.rand(tokens.input_ids.shape, device=DEVICE) < replace_pct
                )
                random_mask &= valid_tokens
                random_tokens = torch.randint(
                    0, tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE
                )
                # Ensure no special tokens are used as replacements
                while any(
                    tokenizer.decode(x) in tokenizer.all_special_tokens
                    for x in random_tokens
                ):
                    random_tokens = torch.randint(
                        0, tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE
                    )
                tokens.input_ids[random_mask] = random_tokens
                perturbed_texts = tokenizer.batch_decode(
                    tokens.input_ids, skip_special_tokens=True
                )
            else:
                # Word-level random replacement from dictionary (ablation)
                masked_texts = [
                    _tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts
                ]
                perturbed_texts = masked_texts
                for idx, text in enumerate(perturbed_texts):
                    filled_text = text
                    for fill_idx in range(_count_masks([text])[0]):
                        fill = random.sample(FILL_DICTIONARY, span_length)
                        filled_text = filled_text.replace(
                            f"<extra_id_{fill_idx}>", " ".join(fill)
                        )
                    assert (
                        _count_masks([filled_text])[0] == 0
                    ), "Failed to replace all masks"
                    perturbed_texts[idx] = filled_text

        return perturbed_texts
