from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src import config
from src.evaluation import get_predicted_ids

sns.set_theme(style="whitegrid", context="notebook")


def figures_directory(name: str) -> Path:
    directory = config.FIGURES_DIRECTORY / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save(figure, folder: str, file_name: str) -> Path:
    path = figures_directory(folder) / file_name
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_confusion_matrix(true_ids, probabilities, model_name: str, split_name: str, normalise: bool = False):
    predicted_ids = get_predicted_ids(probabilities)
    matrix = confusion_matrix(true_ids, predicted_ids, labels=range(len(config.LABELS)))
    figure, axis = plt.subplots(figsize=(5.2, 4.4))
    if normalise:
        shown = matrix / matrix.sum(axis=1, keepdims=True)
        sns.heatmap(shown, annot=matrix, fmt="d", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=config.LABELS, yticklabels=config.LABELS, ax=axis, cbar=False)
        axis.set_title(f"{model_name} ({split_name})\ncounts, shaded by row share")
    else:
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=config.LABELS, yticklabels=config.LABELS, ax=axis, cbar=False)
        axis.set_title(f"{model_name} ({split_name})")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    save(figure, model_name, f"confusion_matrix_{split_name}.png")
    return figure


def plot_learning_curves(history: dict[str, list[float]], model_name: str, run_name: str):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, (loss_axis, f1_axis) = plt.subplots(1, 2, figsize=(10, 3.6))
    loss_axis.plot(epochs, history["train_loss"], label="train")
    loss_axis.plot(epochs, history["validation_loss"], label="validation")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.legend()
    f1_axis.plot(epochs, history["validation_macro_f1"], color="tab:green")
    f1_axis.set_xlabel("Epoch")
    f1_axis.set_ylabel("Validation macro-F1")
    f1_axis.set_ylim(0, 1)
    figure.suptitle(f"{model_name}: {run_name}")
    save(figure, model_name, f"learning_curves_{run_name}.png")
    return figure
