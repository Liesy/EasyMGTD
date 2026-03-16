"""
Debug script for the T-Discrepancy (TDT) detection method.

Experiment type: threshold
This method computes discrepancy between reference and scoring models
and uses a threshold / logistic regression classifier.

Usage:
    cd <project_root>
    python run/debug/test_tdt.py
"""

import torch
from _common import init_env, load_demo_data, print_results
from easymgtd import AutoDetector, AutoExperiment

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "tdt"
# TDT might use 2 GPUs by default if available
GPU_ID = "0,1"
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

    print(f"\n>>> Running {METHOD_NAME} with config: {detector_args}")

    # 2. Create detector and experiment
    detector = AutoDetector.from_detector_name(METHOD_NAME, **detector_args)
    exp = AutoExperiment.from_experiment_name("threshold", detector=[detector])

    # 3. Load data (single topic for debugging)
    data = load_demo_data(config, topic=DEBUG_TOPIC, detectLLM=DEBUG_LLM)
    exp.load_data(data)

    # 4. Run and print results
    results = exp.launch(**experiment_args)
    print_results(results)

    torch.cuda.empty_cache()
