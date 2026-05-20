"""
Shared constants for the EasyMGTD data loading system.

Extracted from the original dataloader.py to centralize dataset,
model, category, and topic definitions.
"""

import os

# ==============================================================================
# Environment Configuration
# ==============================================================================

DATASET_AITextDetect = os.getenv("DATASET_AITextDetect", "AITextDetect/AI_Polish_clean")
DATASET_DIR_OTHERS = os.getenv("DATASET_DIR_OTHERS", "datasets")
SAVED_DATA_DIR = os.getenv("DATASET_DIR_SAVE", "exp_data")

# ==============================================================================
# Dataset and model lists
# ==============================================================================

DATASETS = ["TruthfulQA", "SQuAD1", "NarrativeQA"]

MODELS = ["Moonshot", "gpt35", "Mixtral", "Llama3", "gpt-4omini"]

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

TOPIC_MAPPING = {
    "Physics": "STEM",
    "Math": "STEM",
    "Chemistry": "STEM",
    "Biology": "STEM",
    "Electrical_engineering": "STEM",
    "Computer_science": "STEM",
    "Statistics": "STEM",
    "Medicine": "STEM",
    "Literature": "Humanities",
    "History": "Humanities",
    "Law": "Humanities",
    "Art": "Humanities",
    "Philosophy": "Humanities",
    "Economy": "Social_sciences",
    "Management": "Social_sciences",
    "Education": "Social_sciences",
}

LABEL_MAPPING = {
    "Human": 0,
    "Moonshot": 1,
    "gpt35": 2,
    "Mixtral": 3,
    "Llama3": 4,
    "gpt-4omini": 5,
}

# Source directories for AITextDetect human data
AITEXTDETECT_SOURCE_DICT = {
    "Physics": ["wiki", "arxiv"],
    "Medicine": ["wiki"],
    "Biology": ["wiki", "arxiv"],
    "Electrical_engineering": ["wiki", "arxiv"],
    "Computer_science": ["wiki", "arxiv"],
    "Literature": ["wiki", "gutenberg"],
    "History": ["wiki", "gutenberg"],
    "Education": ["wiki", "gutenberg"],
    "Art": ["wiki", "gutenberg"],
    "Law": ["wiki", "gutenberg"],
    "Management": ["wiki"],
    "Philosophy": ["wiki", "gutenberg"],
    "Economy": ["wiki", "Finance_wiki", "arxiv"],
    "Math": ["wiki", "arxiv"],
    "Statistics": ["wiki", "arxiv"],
    "Chemistry": ["wiki"],
}
