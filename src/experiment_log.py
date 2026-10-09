"""One CSV per model family, one row per experiment.

Every experiment is logged, including the ones that did badly: they are
evidence for what does and does not help on this dataset.
"""

import json
from datetime import datetime

import pandas as pd

from src import config

EXPERIMENT_LOGS_DIRECTORY = config.RESULTS_DIRECTORY / "experiment_logs"


def log_experiment(
    model_family: str,
    experiment_name: str,
    split_name: str,
    settings: dict,
    scores: dict[str, float],
    notes: str = "",
) -> None:
    row = {
        "model_family": model_family,
        "experiment": experiment_name,
        "split": split_name,
        **scores,
        "settings": json.dumps(settings, default=str),
        "notes": notes,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    log_file = EXPERIMENT_LOGS_DIRECTORY / f"{model_family}.csv"
    EXPERIMENT_LOGS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if log_file.exists():
        log = pd.read_csv(log_file)
        # Re-running a notebook replaces the old row for the same experiment
        same = (log["experiment"] == experiment_name) & (log["split"] == split_name)
        log = pd.concat([log[~same], pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])

    log.to_csv(log_file, index=False)
    print(f"Logged '{experiment_name}' ({split_name}) to {log_file.name}")


def load_all_experiments() -> pd.DataFrame:
    files = sorted(EXPERIMENT_LOGS_DIRECTORY.glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
