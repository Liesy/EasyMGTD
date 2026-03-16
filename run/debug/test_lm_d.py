"""
Debug script for the LM-D (Language Model Detector) method.

Experiment type: supervised
LM-D fine-tunes a pre-trained language model (e.g. DistilBERT) as a binary
classifier for human vs. machine text. It supports checkpoint management:
- If no checkpoint exists, it trains from scratch.
- If a checkpoint is found, it evaluates directly.

Usage:
    cd <project_root>
    python run/debug/test_lm_d.py
"""

import os
import re
import torch
from _common import init_env, load_demo_data, print_results, resolve_model_path
from easymgtd import AutoDetector, AutoExperiment

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "LM-D"
GPU_ID = "0"
DEBUG_TOPIC = "STEM"
DEBUG_LLM = "Moonshot"

# ============================================================
# Checkpoint helper (same logic as benchmark.py)
# ============================================================
checkpoint_pattern = re.compile(r"checkpoint-(\d+)")


def get_path(save_dir, model, llm, topic, epoch, batch_size):
    """
    Find the latest checkpoint directory for a given LM-D configuration.

    Args:
        save_dir: Base directory where checkpoints are stored.
        model: Base model name, e.g. "distilbert".
        llm: LLM name being detected, e.g. "Moonshot".
        topic: Topic name, e.g. "STEM".
        epoch: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        Path to the latest checkpoint, or None if not found.
    """
    dir_path = os.path.join(save_dir, f"{model}_{llm}_{topic}_{epoch}_{batch_size}")
    if os.path.exists(dir_path):
        checkpoint_dirs = [
            d for d in os.listdir(dir_path) if checkpoint_pattern.match(d)
        ]
        if checkpoint_dirs:
            checkpoint_paths = [os.path.join(dir_path, d) for d in checkpoint_dirs]
            latest = max(checkpoint_paths, key=os.path.getmtime)
            print(f"Loading checkpoint: {latest}")
            return latest
        else:
            print(f"No checkpoints found in {dir_path}")
    else:
        print(f"Checkpoint directory does not exist: {dir_path}")
    return None


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    config = init_env(gpu_id=GPU_ID)
    global_cfg = config.get("global", {})
    method_cfg = config.get(METHOD_NAME, {})
    detector_args = method_cfg.get("detector_args", {})
    experiment_args = method_cfg.get("experiment_args", {})

    train_size = global_cfg.get("train_size", 1000)
    test_size = global_cfg.get("test_size", 2000)
    max_train = method_cfg.get("train_size_max", 10000)
    epoch = experiment_args.get("epochs", 3)
    batch_size = experiment_args.get("batch_size", 64)

    save_dir = os.environ.get("DATASET_DIR_SAVE", "./exp_data/topic_models")
    os.makedirs(save_dir, exist_ok=True)

    lmd_base_models = method_cfg.get("base_model_names", ["distilbert"])
    model_paths_map = method_cfg.get("model_paths", {})

    # Load full data for the debug topic
    from easymgtd.loading.dataloader import load as load_raw

    dataset_name = global_cfg.get("dataset", "AITextDetect")
    data = load_raw(dataset_name, detectLLM=DEBUG_LLM, category=DEBUG_TOPIC)

    for model_name in lmd_base_models:
        model_path = resolve_model_path(model_paths_map.get(model_name, model_name))
        print(f"\n>>> LM-D base model: {model_name} ({model_path})")

        # Check for existing checkpoint
        ckpt_path = get_path(
            save_dir, model_name, DEBUG_LLM, DEBUG_TOPIC, epoch, batch_size
        )

        if ckpt_path is None:
            # No checkpoint -> train from scratch
            demo = {
                "train": {
                    "text": data["train"]["text"][:max_train],
                    "label": data["train"]["label"][:max_train],
                },
                "test": {
                    "text": data["test"]["text"][:test_size],
                    "label": data["test"]["label"][:test_size],
                },
            }
            detector = AutoDetector.from_detector_name(
                METHOD_NAME,
                model_name_or_path=model_path,
                tokenizer_path=model_path,
                **detector_args,
            )
            exp = AutoExperiment.from_experiment_name("supervised", detector=[detector])
            exp.load_data(demo)

            model_save_dir = os.path.join(
                save_dir,
                f"{model_name}_{DEBUG_LLM}_{DEBUG_TOPIC}_{epoch}_{batch_size}",
            )
            launch_kwargs = {**experiment_args, "save_path": model_save_dir}
            results = exp.launch(**launch_kwargs)
        else:
            # Checkpoint found -> evaluate only
            demo = {
                "train": {
                    "text": data["train"]["text"][:train_size],
                    "label": data["train"]["label"][:train_size],
                },
                "test": {
                    "text": data["test"]["text"][:test_size],
                    "label": data["test"]["label"][:test_size],
                },
            }
            detector = AutoDetector.from_detector_name(
                METHOD_NAME,
                model_name_or_path=ckpt_path,
                tokenizer_path=ckpt_path,
                **detector_args,
            )
            exp = AutoExperiment.from_experiment_name("supervised", detector=[detector])
            exp.load_data(demo)
            results = exp.launch(need_finetune=False)

        print_results(results)
        torch.cuda.empty_cache()
