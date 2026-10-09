import numpy as np
import pandas as pd

from src import config


def load_train() -> pd.DataFrame:
    return pd.read_csv(config.TRAIN_FILE)


def load_validation() -> pd.DataFrame:
    return pd.read_csv(config.VALIDATION_FILE)


def load_test() -> pd.DataFrame:
    return pd.read_csv(config.TEST_FILE)


def load_all_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_train(), load_validation(), load_test()


def labels_to_ids(labels: pd.Series) -> np.ndarray:
    label_to_id = {label: index for index, label in enumerate(config.LABELS)}
    return labels.map(label_to_id).to_numpy()


def ids_to_labels(ids) -> list[str]:
    return [config.LABELS[int(index)] for index in ids]
