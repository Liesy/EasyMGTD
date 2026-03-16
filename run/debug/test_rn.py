"""
Debug script for the RN (Relation Network) few-shot detection method.

Experiment type: fewshot
RN uses a relation network for few-shot classification.

Usage:
    cd <project_root>
    python run/debug/test_rn.py
"""

import torch
from _common import init_env, print_results, resolve_model_path
from easymgtd import AutoDetector, AutoExperiment
from easymgtd.loading.dataloader import load

# ============================================================
# Configuration
# ============================================================
METHOD_NAME = "rn"
GPU_ID = "0"
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

    detector = AutoDetector.from_detector_name(METHOD_NAME, **detector_args)
    exp = AutoExperiment.from_experiment_name("fewshot", detector=[detector])

    dataset_name = global_cfg.get("dataset", "AITextDetect")
    data = load(dataset_name, detectLLM=DEBUG_LLM, category=DEBUG_TOPIC)
    exp.load_data(data)

    results = exp.launch(**experiment_args)
    print_results(results)

    torch.cuda.empty_cache()
