"""
Debug script for the Binoculars detection method.

Experiment type: threshold
Binoculars uses a pair of observer/performer LMs to detect machine-generated text.
Unlike other threshold methods, it does NOT iterate over base_models.

Usage:
    cd <project_root>
    python run/debug/test_binoculars.py
"""

import torch
from _common import init_env, load_demo_data, print_results, resolve_model_path
from easymgtd import AutoDetector, AutoExperiment

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "Binoculars"
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

    # Create detector and experiment (no base_models iteration)
    detector = AutoDetector.from_detector_name(METHOD_NAME, **detector_args)
    exp = AutoExperiment.from_experiment_name("threshold", detector=[detector])

    # Load data for a single topic
    data = load_demo_data(config, topic=DEBUG_TOPIC, detectLLM=DEBUG_LLM)
    exp.load_data(data)

    # Run and print results
    results = exp.launch(**experiment_args)
    print_results(results)

    torch.cuda.empty_cache()
