"""
Debug script for the ConDA detection method.

Experiment type: supervised
ConDA fine-tunes a pre-trained model with contrastive data augmentation.

Usage:
    cd <project_root>
    python run/debug/test_conda.py
"""

import torch
from _common import init_env, load_demo_data, print_results, resolve_model_path
from easymgtd import AutoDetector, AutoExperiment

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "ConDA"
GPU_ID = "0"
DEBUG_TOPIC = "STEM"
DEBUG_LLM = "Moonshot"

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    config = init_env(gpu_id=GPU_ID)
    method_cfg = config.get(METHOD_NAME, {})
    detector_args = method_cfg.get("detector_args", {})
    experiment_args = method_cfg.get("experiment_args", {})

    det_model_path = resolve_model_path(detector_args.get("model_name_or_path"))
    det_tokenizer_path = resolve_model_path(
        detector_args.get("tokenizer_path", det_model_path)
    )

    detector = AutoDetector.from_detector_name(
        METHOD_NAME,
        model_name_or_path=det_model_path,
        tokenizer_path=det_tokenizer_path,
    )
    exp = AutoExperiment.from_experiment_name("supervised", detector=[detector])

    data = load_demo_data(config, topic=DEBUG_TOPIC, detectLLM=DEBUG_LLM)
    exp.load_data(data)

    results = exp.launch(**experiment_args)
    print_results(results)

    torch.cuda.empty_cache()
