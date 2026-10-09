"""Metrics, confidence intervals and misclassified examples.

Main metric: macro-F1. With 4 unbalanced classes (politics is 58% of the
data, business 9%), accuracy and weighted-F1 are dominated by politics, while
macro-F1 gives each topic the same weight, so a model cannot score well by
ignoring business or sports.

We also report weighted-F1 because it is the metric used in the MasakhaNEWS
paper (Adelani et al., 2023), which lets us compare with published baselines.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from src import config

MISCLASSIFIED_DIRECTORY = config.RESULTS_DIRECTORY / "misclassified"
PREDICTIONS_DIRECTORY = config.RESULTS_DIRECTORY / "predictions"
ALL_LABEL_IDS = list(range(len(config.LABELS)))


def get_predicted_ids(probabilities: np.ndarray) -> np.ndarray:
    return np.argmax(probabilities, axis=1)


def compute_scores(true_ids: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted_ids = get_predicted_ids(probabilities)
    average_options = dict(labels=ALL_LABEL_IDS, zero_division=0)

    scores = {
        "accuracy": accuracy_score(true_ids, predicted_ids),
        "macro_precision": precision_score(true_ids, predicted_ids, average="macro", **average_options),
        "macro_recall": recall_score(true_ids, predicted_ids, average="macro", **average_options),
        "macro_f1": f1_score(true_ids, predicted_ids, average="macro", **average_options),
        "weighted_f1": f1_score(true_ids, predicted_ids, average="weighted", **average_options),
        # Log loss checks whether the probabilities are sensible, not just the top label
        "log_loss": log_loss(true_ids, np.clip(probabilities, 1e-7, 1), labels=ALL_LABEL_IDS),
    }

    per_class_f1 = f1_score(true_ids, predicted_ids, average=None, **average_options)
    for label, class_f1 in zip(config.LABELS, per_class_f1):
        scores[f"f1_{label}"] = class_f1

    return {name: round(float(value), 4) for name, value in scores.items()}


def print_classification_report(true_ids: np.ndarray, probabilities: np.ndarray) -> None:
    print(
        classification_report(
            true_ids,
            get_predicted_ids(probabilities),
            labels=ALL_LABEL_IDS,
            target_names=config.LABELS,
            digits=3,
            zero_division=0,
        )
    )


def summarise_runs(list_of_scores: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation over seeds or folds."""
    table = pd.DataFrame(list_of_scores)
    summary = {}
    for name in table.columns:
        summary[name] = round(float(table[name].mean()), 4)
        summary[f"{name}_std"] = round(float(table[name].std(ddof=1)), 4) if len(table) > 1 else 0.0
    return summary


# ------------------------------------------------------ uncertainty


def bootstrap_confidence_interval(
    true_ids: np.ndarray,
    predicted_ids: np.ndarray,
    metric: str = "macro",
    number_of_samples: int = 2000,
    seed: int = config.RANDOM_SEED,
) -> tuple[float, float]:
    """95% bootstrap interval for F1 on the test set.

    We resample the 175 test articles with replacement many times and see how
    much the score moves. A wide interval means differences between models
    smaller than that width may just be luck in which articles are in the test set.
    """
    generator = np.random.default_rng(seed)
    size = len(true_ids)
    samples = []
    for _ in range(number_of_samples):
        rows = generator.integers(0, size, size)
        samples.append(
            f1_score(true_ids[rows], predicted_ids[rows], average=metric,
                     labels=ALL_LABEL_IDS, zero_division=0)
        )
    low, high = np.percentile(samples, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def paired_bootstrap_test(
    true_ids: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    number_of_samples: int = 2000,
    seed: int = config.RANDOM_SEED,
) -> float:
    """Share of bootstrap samples where model B is NOT better than model A on macro-F1.

    This works like a one-sided p-value (Koehn, 2004; Dror et al., 2018):
    a small number (< 0.05) means B's advantage is unlikely to be luck.
    """
    generator = np.random.default_rng(seed)
    size = len(true_ids)
    not_better = 0
    for _ in range(number_of_samples):
        rows = generator.integers(0, size, size)
        score_a = f1_score(true_ids[rows], predictions_a[rows], average="macro", labels=ALL_LABEL_IDS, zero_division=0)
        score_b = f1_score(true_ids[rows], predictions_b[rows], average="macro", labels=ALL_LABEL_IDS, zero_division=0)
        if score_b <= score_a:
            not_better += 1
    return round(not_better / number_of_samples, 4)


# -------------------------------------------------- error analysis


def build_prediction_table(
    data: pd.DataFrame, true_ids: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    predicted_ids = get_predicted_ids(probabilities)
    table = data[["headline", "content"]].copy()
    table["true_label"] = [config.LABELS[i] for i in true_ids]
    table["predicted_label"] = [config.LABELS[i] for i in predicted_ids]
    table["confidence"] = probabilities.max(axis=1).round(4)
    for label_id, label in enumerate(config.LABELS):
        table[f"p_{label}"] = probabilities[:, label_id].round(4)
    table["correct"] = table["true_label"] == table["predicted_label"]
    table["words"] = table["content"].str.split().str.len()
    return table


def save_misclassified_examples(
    data: pd.DataFrame, true_ids: np.ndarray, probabilities: np.ndarray, model_name: str
) -> pd.DataFrame:
    table = build_prediction_table(data, true_ids, probabilities)
    wrong = table[~table["correct"]].sort_values("confidence", ascending=False)
    MISCLASSIFIED_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = MISCLASSIFIED_DIRECTORY / f"{model_name}.csv"
    wrong.to_csv(path, index=False)
    print(f"Saved {len(wrong)} misclassified test articles to {path}")
    return wrong


def save_test_predictions(probabilities: np.ndarray, model_name: str) -> None:
    """Store test-set probabilities so the comparison notebook can rebuild every score."""
    PREDICTIONS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(probabilities, columns=config.LABELS).round(6)
    table.to_csv(PREDICTIONS_DIRECTORY / f"{model_name}.csv", index=False)
    print(f"Saved test probabilities to {PREDICTIONS_DIRECTORY / (model_name + '.csv')}")


def load_test_predictions() -> dict[str, np.ndarray]:
    return {
        path.stem: pd.read_csv(path)[config.LABELS].to_numpy()
        for path in sorted(PREDICTIONS_DIRECTORY.glob("*.csv"))
    }
