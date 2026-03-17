# Truncate-and-regenerate perturbation strategy.
#
# Migrated from easymgtd/methods/perturb.py DNAGPTDetector.PrefixSampler.
# Truncates text at a configurable ratio and uses a causal language model
# to regenerate the remainder, producing divergent samples for comparison.

from ._base import TextPerturbator


class TruncateRegenPerturbator(TextPerturbator):
    """Truncate-and-regenerate perturbation (DNA-GPT style).

    Truncates each input text at a given ratio and uses a causal LM
    to regenerate the truncated portion. The regenerated text is then
    trimmed to match the original length.

    Args:
        base_model: Causal language model for text regeneration.
        base_tokenizer: Corresponding tokenizer.
        batch_size: Batch size for generation. Default 5.
        regen_number: Number of regenerated samples per text. Default 5.
        temperature: Sampling temperature for generation. Default 1.0.
        truncate_ratio: Fraction of text to keep as prefix. Default 0.5.
    """

    def __init__(self, base_model, base_tokenizer, **kwargs):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.batch_size = kwargs.get("batch_size", 5)
        self.regen_number = kwargs.get("regen_number", 5)
        self.temperature = kwargs.get("temperature", 1.0)
        self.truncate_ratio = kwargs.get("truncate_ratio", 0.5)

    def perturb(self, texts, config=None, **kwargs):
        """Generate regenerated samples for input texts.

        For each input text, produces multiple regenerated variants by
        truncating and regenerating. Returns the regenerated samples
        (not the original texts).

        Args:
            texts: List of input text strings.
            config: Optional config object (unused, kept for interface compat).
            **kwargs: Additional parameters (unused).

        Returns:
            list[str]: Regenerated text samples.
        """
        data = self.generate_samples(texts, batch_size=self.batch_size)
        return data["sampled"]

    def _sample_from_model(self, texts, min_words=55, truncate_ratio=0.5):
        """Generate text continuations from truncated prefixes.

        Truncates each text at the given ratio and uses the causal LM
        to generate a continuation. Retries if the output is too short.

        Args:
            texts: List of input text strings.
            min_words: Minimum number of words in generated output.
            truncate_ratio: Fraction of text to keep as prefix input.

        Returns:
            list[str]: Generated text continuations.
        """
        # Split and truncate to prefix
        texts = [t.split(" ") for t in texts]
        texts = [t if len(t) <= 1024 else t[:1024] for t in texts]
        texts = [" ".join(t[: int(len(t) * truncate_ratio)]) for t in texts]

        all_encoded = self.base_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.base_model.device)

        decoded = ["" for _ in range(len(texts))]
        # Regenerate until all outputs have at least min_words words
        tries = 0
        m = 0
        while m < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {"temperature": self.temperature}
            min_length = 150
            outputs = self.base_model.generate(
                **all_encoded,
                min_length=min_length,
                max_length=1024,
                do_sample=True,
                **sampling_kwargs,
                pad_token_id=self.base_tokenizer.eos_token_id,
                eos_token_id=self.base_tokenizer.eos_token_id,
            )
            decoded = self.base_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            m = min(len(x.split()) for x in decoded)
            tries += 1

        return decoded

    def generate_samples(self, raw_data, batch_size):
        """Generate paired original/sampled data in batches.

        For each batch of inputs, generates regenerated text and trims
        both original and regenerated to matching lengths.

        Args:
            raw_data: List of input text strings.
            batch_size: Number of texts to process per batch.

        Returns:
            dict: {'original': list[str], 'sampled': list[str]}
        """

        def _trim_to_shorter_length(texta, textb):
            """Truncate both texts to the shorter one's length."""
            shorter_length = min(len(texta.split(" ")), len(textb.split(" ")))
            texta = " ".join(texta.split(" ")[:shorter_length])
            textb = " ".join(textb.split(" ")[:shorter_length])
            return texta, textb

        data = {
            "original": [],
            "sampled": [],
        }

        assert len(raw_data) % batch_size == 0

        for batch in range(len(raw_data) // batch_size):
            original_text = raw_data[batch * batch_size : (batch + 1) * batch_size]
            sampled_text = self._sample_from_model(
                original_text, min_words=55, truncate_ratio=self.truncate_ratio
            )

            for o, s in zip(original_text, sampled_text):
                o, s = _trim_to_shorter_length(o, s)
                data["original"].append(o)
                data["sampled"].append(s)

        return data
