"""
Debug script for the GPTZero detection method.

Experiment type: gptzero (uses threshold experiment internally)
GPTZero calls the GPTZero API for detection. You MUST set a valid
api_key in config.json before running this script.

Usage:
    cd <project_root>
    python run/debug/test_gptzero.py
"""

import torch
from _common import init_env, load_demo_data, print_results
from easymgtd import AutoDetector, AutoExperiment

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "GPTZero"
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

    if not detector_args.get("api_key"):
        print("WARNING: GPTZero api_key is empty in config.json.")
        print("Please set a valid api_key before running this script.")

    # Create detector and experiment
    detector = AutoDetector.from_detector_name(METHOD_NAME, **detector_args)
    exp = AutoExperiment.from_experiment_name("threshold", detector=[detector])

    # Load data
    data = load_demo_data(config, topic=DEBUG_TOPIC, detectLLM=DEBUG_LLM)
    exp.load_data(data)

    # Run and print results
    results = exp.launch(**experiment_args)
    print_results(results)

    torch.cuda.empty_cache()
