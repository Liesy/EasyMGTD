"""
Debug script for the DetectGPT detection method.

Experiment type: perturb
DetectGPT uses perturbation-based curvature estimation to detect machine text.

Usage:
    cd <project_root>
    python run/debug/test_detectgpt.py
"""

import torch
from _common import init_env, load_demo_data, print_results, resolve_model_path
from easymgtd import AutoDetector, AutoExperiment

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "detectGPT"
GPU_ID = "0,1"
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

    # Resolve any $ENV_VAR references in detector_args
    for key, val in detector_args.items():
        detector_args[key] = resolve_model_path(val)

    # Create detector and experiment
    detector = AutoDetector.from_detector_name(METHOD_NAME, **detector_args)
    exp = AutoExperiment.from_experiment_name("perturb", detector=[detector])

    # Load data for a single topic
    data = load_demo_data(config, topic=DEBUG_TOPIC, detectLLM=DEBUG_LLM)
    exp.load_data(data)

    # Run and print results
    results = exp.launch(**experiment_args)
    print_results(results)

    torch.cuda.empty_cache()
