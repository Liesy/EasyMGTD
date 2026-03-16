"""
Debug script for the Rank-GLTR detection method.

Experiment type: threshold
This method uses GLTR-style rank buckets from a base LM for detection.

Usage:
    cd <project_root>
    python run/debug/test_rank_gltr.py
"""

import torch
from _common import init_env, load_demo_data, print_results, resolve_model_path
from easymgtd import AutoDetector, AutoExperiment

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "rank_GLTR"
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
    base_models = method_cfg.get("base_models", [])

    if not base_models:
        raise ValueError(f"Config for '{METHOD_NAME}' must specify 'base_models' list.")

    for base_model in base_models:
        base_model = resolve_model_path(base_model)
        print(f"\n>>> Base model: {base_model}")

        det_args = {**detector_args, "model_name_or_path": base_model}
        detector = AutoDetector.from_detector_name(METHOD_NAME, **det_args)
        exp = AutoExperiment.from_experiment_name("threshold", detector=[detector])

        data = load_demo_data(config, topic=DEBUG_TOPIC, detectLLM=DEBUG_LLM)
        exp.load_data(data)

        results = exp.launch(**experiment_args)
        print_results(results)

        torch.cuda.empty_cache()
