"""
Debug script for the Log-Likelihood (ll) detection method.

Experiment type: threshold
This method computes per-token log-likelihood under a base LM and uses
a threshold / logistic regression classifier to distinguish human vs. machine text.

Usage:
    cd <project_root>
    python run/debug/test_ll.py
"""

import torch
from _common import init_env, load_demo_data, print_results, resolve_model_path
from easymgtd import AutoDetector, AutoExperiment

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "ll"
GPU_ID = "0,1"
# Only test on the first topic for quick debugging; change as needed.
DEBUG_TOPIC = "STEM"
DEBUG_LLM = "Moonshot"

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # 1. Initialize environment and load config
    config = init_env(gpu_id=GPU_ID)
    method_cfg = config.get(METHOD_NAME, {})
    detector_args = method_cfg.get("detector_args", {})
    experiment_args = method_cfg.get("experiment_args", {})
    base_models = method_cfg.get("base_models", [])

    if not base_models:
        raise ValueError(f"Config for '{METHOD_NAME}' must specify 'base_models' list.")

    # 2. Iterate over base models
    for base_model in base_models:
        base_model = resolve_model_path(base_model)
        print(f"\n>>> Base model: {base_model}")

        # Merge base_model into detector_args
        det_args = {**detector_args, "model_name_or_path": base_model}

        # 3. Create detector and experiment
        detector = AutoDetector.from_detector_name(METHOD_NAME, **det_args)
        exp = AutoExperiment.from_experiment_name("threshold", detector=[detector])

        # 4. Load data (single topic for debugging)
        data = load_demo_data(config, topic=DEBUG_TOPIC, detectLLM=DEBUG_LLM)
        exp.load_data(data)

        # 5. Run and print results
        results = exp.launch(**experiment_args)
        print_results(results)

        torch.cuda.empty_cache()
