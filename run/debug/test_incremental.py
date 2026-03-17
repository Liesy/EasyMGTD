"""
Debug script for the Incremental detection method.

Experiment type: incremental
The incremental detector uses class-incremental learning to adapt to new LLMs.
Unlike other methods, it uses the full data (not truncated by train/test size)
and loads data via load_incremental or prepare_incremental from the dataloader.

Usage:
    cd <project_root>
    python run/debug/test_incremental.py
"""

import torch
from _common import init_env, print_results, resolve_model_path
from easymgtd import AutoDetector, AutoExperiment
from easymgtd.loading.dataloader import load

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "incremental"
GPU_ID = "0,1"
DEBUG_TOPIC = "STEM"
DEBUG_LLM = "Moonshot"

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    config = init_env(gpu_id=GPU_ID)
    global_cfg = config.get("global", {})
    method_cfg = config.get(METHOD_NAME, {})
    detector_args = method_cfg.get("detector_args", {})
    experiment_args = method_cfg.get("experiment_args", {})

    for key, val in detector_args.items():
        detector_args[key] = resolve_model_path(val)

    # Create detector and experiment
    detector = AutoDetector.from_detector_name(METHOD_NAME, **detector_args)
    exp = AutoExperiment.from_experiment_name("incremental", detector=[detector])

    # Load data (incremental uses the full data, no truncation)
    dataset_name = global_cfg.get("dataset", "AITextDetect")
    data = load(dataset_name, detectLLM=DEBUG_LLM, category=DEBUG_TOPIC)
    exp.load_data(data)

    # Run and print results
    results = exp.launch(**experiment_args)
    print_results(results)

    torch.cuda.empty_cache()
