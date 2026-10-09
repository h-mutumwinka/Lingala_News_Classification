"""Repeated stratified k-fold cross-validation for the fast (scikit-learn) models.

The validation set has only 87 articles (8 business, 10 sports), so one
article changes macro-F1 by several points. To choose settings for the
bag-of-words models we therefore also use 5-fold cross-validation on the 608
training articles, repeated 3 times with different shuffles, and compare the
mean and standard deviation over the 15 folds.
"""

from typing import Callable

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src import config, data_loading
from src.evaluation import compute_scores, summarise_runs

NUMBER_OF_FOLDS = 5
NUMBER_OF_REPEATS = 3


def cross_validate(build_model: Callable, data: pd.DataFrame, text_column: str = config.TEXT_COLUMN) -> dict:
    texts = data[text_column]
    label_ids = data_loading.labels_to_ids(data["category"])
    folds = RepeatedStratifiedKFold(
        n_splits=NUMBER_OF_FOLDS, n_repeats=NUMBER_OF_REPEATS, random_state=config.RANDOM_SEED
    )
    fold_scores = []
    for train_rows, holdout_rows in folds.split(texts, label_ids):
        model = build_model()  # a fresh, untrained model for every fold
        model.fit(texts.iloc[train_rows], label_ids[train_rows])
        fold_scores.append(compute_scores(label_ids[holdout_rows], model.predict_proba(texts.iloc[holdout_rows])))
    return summarise_runs(fold_scores)
