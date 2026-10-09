"""Paths, labels and shared settings for the whole project.

Every notebook and script imports from here, so a setting only has to be
changed in one place.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- folders
RAW_DATA_DIRECTORY = ROOT / "data" / "raw"
PROCESSED_DATA_DIRECTORY = ROOT / "data" / "processed"
RESULTS_DIRECTORY = ROOT / "results"
FIGURES_DIRECTORY = RESULTS_DIRECTORY / "figures"
MODELS_DIRECTORY = ROOT / "models"

# ------------------------------------------------------------ data files
# MasakhaNEWS (Adelani et al., 2023), Lingala subset, official splits.
# Hugging Face: https://huggingface.co/datasets/masakhane/masakhanews (config "lin")
# GitHub mirror: https://github.com/masakhane-io/masakhane-news/tree/main/data/lin
HUGGING_FACE_DATASET = "masakhane/masakhanews"
HUGGING_FACE_CONFIG = "lin"
GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/masakhane-io/masakhane-news/main/data/lin/{split}.tsv"
)

RAW_SPLIT_FILES = {
    "train": RAW_DATA_DIRECTORY / "train.tsv",
    "validation": RAW_DATA_DIRECTORY / "dev.tsv",
    "test": RAW_DATA_DIRECTORY / "test.tsv",
}

TRAIN_FILE = PROCESSED_DATA_DIRECTORY / "train.csv"
VALIDATION_FILE = PROCESSED_DATA_DIRECTORY / "validation.csv"
TEST_FILE = PROCESSED_DATA_DIRECTORY / "test.csv"

# ---------------------------------------------------------------- labels
# Ordered from most to least frequent in the training data.
LABELS = ["politics", "health", "sports", "business"]

# Lingala words for the categories, shown in the web app next to the English label.
LABEL_LINGALA = {
    "politics": "politiki",
    "health": "bokolongono",
    "sports": "masano",
    "business": "mombongo",
}

# ------------------------------------------------------------- settings
RANDOM_SEED = 42

# Neural models are trained with several seeds and we report mean and std,
# because the validation set (87 articles) and test set (175) are small.
FINAL_RUN_SEEDS = [42, 7, 2026]

# Text field that every model reads. "content" = headline + text,
# which MasakhaNEWS reports as the stronger setting.
TEXT_COLUMN = "content"
