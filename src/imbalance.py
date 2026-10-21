"""Strategies for the class imbalance of Challenge 2 (scc is only ~7% of the images).

Every strategy is applied only to the training part of each CV fold, so the held-out fold
(and later the validation set) keeps its real class proportions.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from src.config import RANDOM_STATE
from src.experiments import CV, multiclass_metrics, balanced_fit_params

METHODS = ["none", "weights", "undersample", "oversample"]


def resample(X, y, method, seed=RANDOM_STATE):
    """Random undersampling (every class down to the smallest one) or
    oversampling (every class up to the largest one, duplicating images)."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y)
    target = counts.min() if method == "undersample" else counts.max()
    idx = np.concatenate([
        rng.choice(np.flatnonzero(y == c), target, replace=(method == "oversample"))
        for c in range(len(counts))
    ])
    return X[idx], y[idx]


def fit_with_method(model, X, y, method):
    """Train a copy of model on (X, y) with one strategy."""
    model = clone(model)
    if method == "weights":
        return model.fit(X, y, **balanced_fit_params(model, y))
    if method in ("undersample", "oversample"):
        return model.fit(*resample(X, y, method))
    return model.fit(X, y)                                       # "none"


def cv_method(model, X, y, method, n_jobs=-2):
    """5-fold CV of one strategy. Returns mean and std of the multiclass metrics."""
    def one_fold(train_idx, test_idx):
        fitted = fit_with_method(model, X[train_idx], y[train_idx], method)
        return multiclass_metrics(y[test_idx], fitted.predict(X[test_idx]))

    folds = pd.DataFrame(Parallel(n_jobs=n_jobs)(
        delayed(one_fold)(tr, te) for tr, te in CV.split(X, y)))
    return {**folds.mean().to_dict(), **{f"{m}_std": s for m, s in folds.std(ddof=0).items()}}
