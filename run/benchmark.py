import os
from dotenv import load_dotenv

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "3")
import torch
import argparse
import csv
import re
import json
from easymgtd import AutoDetector, AutoExperiment
from easymgtd.loading.dataloader import load
from easymgtd.utils import setup_seed
from easymgtd.auto import DetectOutput, DETECTOR_MAPPING

# ============================================================
# Constants: category / topic / LLM lists
# ============================================================
CATEGORIES = [
    "Physics",
    "Medicine",
    "Biology",
    "Electrical_engineering",
    "Computer_science",
    "Literature",
    "History",
    "Education",
    "Art",
    "Law",
    "Management",
    "Philosophy",
    "Economy",
    "Math",
    "Statistics",
    "Chemistry",
]

TOPICS = ["STEM", "Humanities", "Social_sciences"]

LLMS = ["Moonshot", "gpt35", "Mixtral", "Llama3", "gpt-4omini"]


# ============================================================
# Experiment type -> AutoExperiment name mapping
# Maps the user-facing experiment_type string in config.json
# to the internal experiment name used by AutoExperiment.
# ============================================================
EXPERIMENT_TYPE_MAP = {
    "threshold": "threshold",
    "perturb": "perturb",
    "supervised": "supervised",
    "demasq": "demasq",
    "incremental": "incremental",
    "incremental_threshold": "incremental_threshold",
    "fewshot": "fewshot",
    "gptzero": "threshold",  # GPTZero uses threshold experiment
}

# ============================================================
# Load directories
# ============================================================
save_dir = os.environ.get("DATASET_DIR_SAVE", "./exp_data/topic_models")
base_dir = save_dir
os.makedirs(base_dir, exist_ok=True)


# ============================================================
# CSV logging helpers
# ============================================================
def log_result_lmd(
    csv_file: str,
    result: DetectOutput,
    method: str,
    detectLLM: str,
    category: str,
    base_model,
    epoch: int,
    batch_size: int,
):
    """Log results for LM-D style experiments with epoch/batch_size columns."""
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "method",
                    "detectLLM",
                    "category",
                    "base_model",
                    "epoch",
                    "batch_size",
                    "criterion",
                    "train_acc",
                    "train_precision",
                    "train_recall",
                    "train_f1",
                    "train_auc",
                    "test_acc",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "test_auc",
                ]
            )

    with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        criterion = result.name if result.name else "None"
        csvwriter.writerow(
            [
                method,
                detectLLM,
                category,
                base_model,
                epoch,
                batch_size,
                criterion,
                round(result.train.acc, 4),
                round(result.train.precision, 4),
                round(result.train.recall, 4),
                round(result.train.f1, 4),
                round(result.train.auc, 4),
                round(result.test.acc, 4),
                round(result.test.precision, 4),
                round(result.test.recall, 4),
                round(result.test.f1, 4),
                round(result.test.auc, 4),
            ]
        )


def log_result(
    csv_file: str,
    result: DetectOutput,
    method: str,
    detectLLM: str,
    category: str,
    base_model="None",
):
    """Log results for standard experiments."""
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "method",
                    "detectLLM",
                    "category",
                    "base_model",
                    "criterion",
                    "train_acc",
                    "train_precision",
                    "train_recall",
                    "train_f1",
                    "train_auc",
                    "test_acc",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "test_auc",
                ]
            )

    with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        criterion = result.name if result.name else "None"
        csvwriter.writerow(
            [
                method,
                detectLLM,
                category,
                base_model,
                criterion,
                round(result.train.acc, 4),
                round(result.train.precision, 4),
                round(result.train.recall, 4),
                round(result.train.f1, 4),
                round(result.train.auc, 4),
                round(result.test.acc, 4),
                round(result.test.precision, 4),
                round(result.test.recall, 4),
                round(result.test.f1, 4),
                round(result.test.auc, 4),
            ]
        )


# ============================================================
# Checkpoint loading helper (for LM-D)
# ============================================================
checkpoint_pattern = re.compile(r"checkpoint-(\d+)")


def get_path(model, llm, topic, epoch, batch_size):
    """Find the latest checkpoint directory for a given LM-D configuration."""
    dir_path = os.path.join(base_dir, f"{model}_{llm}_{topic}_{epoch}_{batch_size}")

    if os.path.exists(dir_path):
        checkpoint_dirs = [
            d for d in os.listdir(dir_path) if checkpoint_pattern.match(d)
        ]
        if checkpoint_dirs:
            checkpoint_paths = [os.path.join(dir_path, d) for d in checkpoint_dirs]
            latest_checkpoint_path = max(checkpoint_paths, key=os.path.getmtime)
            print(f"Loading {latest_checkpoint_path}")
        else:
            print(f"No checkpoints found in {dir_path}")
            latest_checkpoint_path = None
    else:
        print(f"Directory does not exist: {dir_path}")
        latest_checkpoint_path = None

    return latest_checkpoint_path


def get_demo_data(data, train_size, test_size):
    """Truncate data to given train/test sizes for quick experimentation."""
    demo = {}
    demo["train"] = {
        "text": data["train"]["text"][:train_size],
        "label": data["train"]["label"][:train_size],
    }
    demo["test"] = {
        "text": data["test"]["text"][:test_size],
        "label": data["test"]["label"][:test_size],
    }
    return demo


# ============================================================
# Helper: resolve environment variable overrides for model paths
# If a model path starts with "$", treat it as an env var reference.
# Otherwise, return the path as-is.
# ============================================================
def resolve_model_path(path):
    """Resolve a model path, supporting $ENV_VAR syntax for environment variable overrides."""
    if path and isinstance(path, str) and path.startswith("$"):
        env_var = path[1:]
        return os.environ.get(env_var, path)
    return path


# ============================================================
# Main experiment dispatcher
# Reads method-specific config from the full config dict and
# dispatches to the appropriate experiment flow.
# ============================================================
def experiment(csv_file, method, detectLLM, config):
    """
    Run a detection experiment for the given method.

    All method-specific parameters are read from `config[method]`.
    The `config["global"]` section provides shared settings like
    seed, train_size, test_size, topics, and dataset name.

    Args:
        csv_file: Path to the output CSV file for logging results.
        method: Name of the detection method (must be a key in DETECTOR_MAPPING).
        detectLLM: Name of the LLM whose generated text is being detected.
        config: Full configuration dict loaded from config.json.
    """
    # --- Global settings ---
    global_cfg = config.get("global", {})
    seed = global_cfg.get("seed", 3407)
    train_size = global_cfg.get("train_size", 1000)
    test_size = global_cfg.get("test_size", 2000)
    topics = global_cfg.get("topics", TOPICS)
    dataset_name = global_cfg.get("dataset", "AITextDetect")
    setup_seed(seed)

    # --- Method-specific config ---
    method_cfg = config.get(method, {})
    if not method_cfg:
        print(
            f"Warning: No config found for method '{method}'. "
            f"Using defaults from the method/experiment classes."
        )

    experiment_type = method_cfg.get("experiment_type", None)
    detector_args = method_cfg.get("detector_args", {})
    experiment_args = method_cfg.get("experiment_args", {})

    # Resolve any $ENV_VAR references in detector_args
    for key, val in detector_args.items():
        detector_args[key] = resolve_model_path(val)

    # ---- Metric-based methods: threshold experiment ----
    # These methods require a base_model (scoring LM) and use threshold/logistic
    # regression for classification.
    if method in ["ll", "rank", "LRR", "rank_GLTR", "entropy"]:
        base_models = method_cfg.get("base_models", [])
        if not base_models:
            raise ValueError(f"Config for '{method}' must specify 'base_models' list.")
        for base_model in base_models:
            base_model = resolve_model_path(base_model)
            # Merge base_model into detector_args
            det_args = {**detector_args, "model_name_or_path": base_model}
            detector = AutoDetector.from_detector_name(method, **det_args)
            exp = AutoExperiment.from_experiment_name("threshold", detector=[detector])
            for topic in topics:
                data = load(dataset_name, detectLLM=detectLLM, category=topic)
                data = get_demo_data(data, train_size=train_size, test_size=test_size)
                exp.load_data(data)
                res = exp.launch(**experiment_args)
                print("==========")
                print("train:", res[0].train)
                print("test:", res[0].test)
                log_result(
                    csv_file, res[0], method, detectLLM, topic, base_model=base_model
                )
                if len(res) > 1:
                    print("==========")
                    print("train:", res[1].train)
                    print("test:", res[1].test)
                    log_result(
                        csv_file,
                        res[1],
                        method,
                        detectLLM,
                        topic,
                        base_model=base_model,
                    )
                torch.cuda.empty_cache()

    # ---- Binoculars: threshold experiment without base_models iteration ----
    elif method == "Binoculars":
        detector = AutoDetector.from_detector_name("Binoculars", **detector_args)
        exp = AutoExperiment.from_experiment_name("threshold", detector=[detector])
        for topic in topics:
            data = load(dataset_name, detectLLM=detectLLM, category=topic)
            data = get_demo_data(data, train_size=train_size, test_size=test_size)
            exp.load_data(data)
            res = exp.launch(**experiment_args)
            print("==========")
            print("train:", res[0].train)
            print("test:", res[0].test)
            log_result(csv_file, res[0], method, detectLLM, topic)
            if len(res) > 1:
                print("==========")
                print("train:", res[1].train)
                print("test:", res[1].test)
                log_result(csv_file, res[1], method, detectLLM, topic)
            torch.cuda.empty_cache()

    # ---- LM-D: supervised experiment with checkpoint management ----
    elif method == "LM-D":
        lmd_base_models = method_cfg.get("base_model_names", ["distilbert"])
        model_paths_map = method_cfg.get("model_paths", {})
        max_train = method_cfg.get("train_size_max", 10000)
        epoch = experiment_args.get("epochs", 3)
        batch_size = experiment_args.get("batch_size", 64)

        for topic in topics:
            data = load(dataset_name, detectLLM=detectLLM, category=topic)
            for model_name in lmd_base_models:
                model_path = resolve_model_path(
                    model_paths_map.get(model_name, model_name)
                )

                # Check if a checkpoint already exists
                path = get_path(model_name, detectLLM, topic, epoch, batch_size)
                if path is None:
                    # No checkpoint -> train from scratch
                    train_data = get_demo_data(
                        data, train_size=max_train, test_size=test_size
                    )
                    detector = AutoDetector.from_detector_name(
                        method,
                        model_name_or_path=model_path,
                        tokenizer_path=model_path,
                        **detector_args,
                    )
                    exp = AutoExperiment.from_experiment_name(
                        "supervised", detector=[detector]
                    )
                    exp.load_data(train_data)

                    # Build launch kwargs from experiment_args
                    model_save_dir = os.path.join(
                        save_dir,
                        f"{model_name}_{detectLLM}_{topic}_{epoch}_{batch_size}",
                    )
                    launch_kwargs = {**experiment_args, "save_path": model_save_dir}
                    res = exp.launch(**launch_kwargs)

                    print("==========")
                    print("train:", res[0].train)
                    print("test:", res[0].test)
                    log_result(
                        csv_file,
                        res[0],
                        method,
                        detectLLM,
                        topic,
                        base_model=model_name,
                    )
                    torch.cuda.empty_cache()
                else:
                    # Checkpoint found -> evaluate only
                    eval_data = get_demo_data(
                        data, train_size=train_size, test_size=test_size
                    )
                    detector = AutoDetector.from_detector_name(
                        method,
                        model_name_or_path=path,
                        tokenizer_path=path,
                        **detector_args,
                    )
                    exp = AutoExperiment.from_experiment_name(
                        "supervised", detector=[detector]
                    )
                    exp.load_data(eval_data)
                    res = exp.launch(need_finetune=False)
                    print("==========")
                    print("train:", res[0].train)
                    print("test:", res[0].test)
                    log_result(
                        csv_file,
                        res[0],
                        method,
                        detectLLM,
                        topic,
                        base_model=model_name,
                    )
                    torch.cuda.empty_cache()

    # ---- RADAR / ChatGPT-D / other pre-trained supervised detectors ----
    elif experiment_type == "supervised" and method != "LM-D":
        det_model_path = resolve_model_path(detector_args.get("model_name_or_path"))
        det_tokenizer_path = resolve_model_path(
            detector_args.get("tokenizer_path", det_model_path)
        )
        for topic in topics:
            data = load(dataset_name, detectLLM=detectLLM, category=topic)
            data = get_demo_data(data, train_size=train_size, test_size=test_size)
            detector = AutoDetector.from_detector_name(
                method,
                model_name_or_path=det_model_path,
                tokenizer_path=det_tokenizer_path,
            )
            exp = AutoExperiment.from_experiment_name("supervised", detector=[detector])
            exp.load_data(data)
            res = exp.launch(**experiment_args)
            print("==========")
            print("train:", res[0].train)
            print("test:", res[0].test)
            log_result(csv_file, res[0], method, detectLLM, topic)
            torch.cuda.empty_cache()

    # ---- Perturbation-based methods ----
    elif experiment_type == "perturb":
        detector = AutoDetector.from_detector_name(method, **detector_args)
        exp = AutoExperiment.from_experiment_name("perturb", detector=[detector])

        for topic in topics:
            data = load(dataset_name, detectLLM=detectLLM, category=topic)
            data = get_demo_data(data, train_size=train_size, test_size=test_size)
            exp.load_data(data)
            res = exp.launch(**experiment_args)
            print("==========")
            print(res[0])
            print("train:", res[0].train)
            print("test:", res[0].test)
            log_result(csv_file, res[0], method, detectLLM, topic)
            if len(res) > 1:
                print("==========")
                print(res[1])
                print("train:", res[1].train)
                print("test:", res[1].test)
                log_result(csv_file, res[1], method, detectLLM, topic)
            torch.cuda.empty_cache()

    # ---- Demasq ----
    elif experiment_type == "demasq":
        detector = AutoDetector.from_detector_name(method, **detector_args)
        exp = AutoExperiment.from_experiment_name("demasq", detector=[detector])
        for topic in topics:
            data = load(dataset_name, detectLLM=detectLLM, category=topic)
            data = get_demo_data(data, train_size=train_size, test_size=test_size)
            exp.load_data(data)
            res = exp.launch(**experiment_args)
            print("==========")
            for r in res:
                print("train:", r.train)
                print("test:", r.test)
                log_result(csv_file, r, method, detectLLM, topic)
            torch.cuda.empty_cache()

    # ---- Incremental ----
    elif experiment_type == "incremental":
        detector = AutoDetector.from_detector_name(method, **detector_args)
        exp = AutoExperiment.from_experiment_name("incremental", detector=[detector])
        for topic in topics:
            data = load(dataset_name, detectLLM=detectLLM, category=topic)
            exp.load_data(data)
            res = exp.launch(**experiment_args)
            print("==========")
            for r in res:
                print("test:", r.test)
                log_result(csv_file, r, method, detectLLM, topic)
            torch.cuda.empty_cache()

    # ---- Few-shot ----
    elif experiment_type == "fewshot":
        detector = AutoDetector.from_detector_name(method, **detector_args)
        exp = AutoExperiment.from_experiment_name("fewshot", detector=[detector])
        for topic in topics:
            data = load(dataset_name, detectLLM=detectLLM, category=topic)
            exp.load_data(data)
            res = exp.launch(**experiment_args)
            print("==========")
            for r in res:
                print("test:", r.test)
                log_result(csv_file, r, method, detectLLM, topic)
            torch.cuda.empty_cache()

    # ---- GPTZero ----
    elif experiment_type == "gptzero":
        detector = AutoDetector.from_detector_name(method, **detector_args)
        exp = AutoExperiment.from_experiment_name("threshold", detector=[detector])
        for topic in topics:
            data = load(dataset_name, detectLLM=detectLLM, category=topic)
            data = get_demo_data(data, train_size=train_size, test_size=test_size)
            exp.load_data(data)
            res = exp.launch(**experiment_args)
            print("==========")
            for r in res:
                print("test:", r.test)
                log_result(csv_file, r, method, detectLLM, topic)
            torch.cuda.empty_cache()

    else:
        raise ValueError(
            f"Unknown experiment_type '{experiment_type}' for method '{method}'. "
            f"Please check config.json."
        )


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MGT detection benchmarks with per-method configuration."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the output CSV file for logging results.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list(DETECTOR_MAPPING.keys()),
        help="Name of the detection method to run.",
    )
    parser.add_argument(
        "--detectLLM",
        type=str,
        required=True,
        help="Name of the LLM whose generated text is being detected.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="run/config.json",
        help="Path to the JSON configuration file.",
    )

    args = parser.parse_args()
    csv_file = args.csv_path
    method = args.method
    detectLLM = args.detectLLM
    config_path = args.config_path

    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        print(
            f"Warning: Config file {config_path} not found. Proceeding with defaults."
        )

    experiment(csv_file, method, detectLLM, config)
