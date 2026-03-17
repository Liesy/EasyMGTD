"""
Debug script for the Fast-DetectGPT detection method.

Experiment type: perturb
Fast-DetectGPT is an efficient variant of DetectGPT that uses a separate
scoring model and reference model for perturbation analysis.

Usage:
    cd <project_root>
    python run/debug/test_fast_detectgpt.py
"""

import torch
from _common import init_env, load_demo_data, print_results, resolve_model_path
from easymgtd import AutoDetector, AutoExperiment

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "fast-detectGPT"
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

    for key, val in detector_args.items():
        detector_args[key] = resolve_model_path(val)

    detector = AutoDetector.from_detector_name(METHOD_NAME, **detector_args)
    exp = AutoExperiment.from_experiment_name("perturb", detector=[detector])

    data = load_demo_data(config, topic=DEBUG_TOPIC, detectLLM=DEBUG_LLM)
    exp.load_data(data)

    results = exp.launch(**experiment_args)
    print_results(results)

    torch.cuda.empty_cache()
