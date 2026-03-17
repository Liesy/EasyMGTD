"""
Debug script for the Demasq detection method.

Experiment type: demasq
Demasq trains a specialized model for machine text detection.

Usage:
    cd <project_root>
    python run/debug/test_demasq.py
"""

import torch
from _common import init_env, load_demo_data, print_results, resolve_model_path
from easymgtd import AutoDetector, AutoExperiment

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "demasq"
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

    # Create detector and experiment
    detector = AutoDetector.from_detector_name(METHOD_NAME, **detector_args)
    exp = AutoExperiment.from_experiment_name("demasq", detector=[detector])

    # Load data
    data = load_demo_data(config, topic=DEBUG_TOPIC, detectLLM=DEBUG_LLM)
    exp.load_data(data)

    # Run and print results
    results = exp.launch(**experiment_args)
    print_results(results)

    torch.cuda.empty_cache()
