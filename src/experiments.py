"""
Standard model comparison for one experiment, for both challenges:
  - Challenge 1 (binary, nevus vs others): models chosen by balanced accuracy.
  - Challenge 2 (multiclass, mel / bcc / scc): models chosen by Cohen's kappa.

Every model is a pipeline:  scaler -> (optional) feature selector -> classifier.
With weighted=True the classifier is trained with class weights whose strength (alpha)
is chosen by cross-validation like any other hyperparameter.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import make_scorer, recall_score, cohen_kappa_score, get_scorer
from sklearn.utils.metaestimators import available_if
from scipy.stats import loguniform, randint, uniform
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.config import RANDOM_STATE, MULTI_CLASS_NAMES

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------------------------------------------------------
# Metrics per challenge: (scorers, metric used to choose the best model)
# ---------------------------------------------------------------------------
BINARY_SCORING = {
    "accuracy":          "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "sensitivity":       make_scorer(recall_score, pos_label=1),   # others detected
    "specificity":       make_scorer(recall_score, pos_label=0),   # nevus detected
    "auc":               "roc_auc",
}

MULTI_SCORING = {
    "kappa":    make_scorer(cohen_kappa_score),
    **{f"recall_{name}": make_scorer(recall_score, labels=[label], average="macro")
       for label, name in MULTI_CLASS_NAMES.items()},
    "f1_macro": "f1_macro",
}

TASKS = {
    "binary":     (BINARY_SCORING, "balanced_accuracy"),
    "multiclass": (MULTI_SCORING, "kappa"),
}

# ---------------------------------------------------------------------------
# Class weights with a tunable strength
# ---------------------------------------------------------------------------
class WeightedClassifier(ClassifierMixin, BaseEstimator):
    """
    Trains `estimator` with class weights  (n / (n_classes * n_class)) ** alpha.
      alpha = 0 -> every image weighs 1 (no weighting)
      alpha = 1 -> every class has the same total weight (fully balanced)
    Weights are rescaled to mean 1, so the meaning of C (SVM) or the learning rate does not change.
    """
    def __init__(self, estimator, alpha=1.0):
        self.estimator = estimator
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_, counts = np.unique(y, return_counts=True)
        class_weight = (len(y) / (len(self.classes_) * counts)) ** self.alpha
        sample_weight = class_weight[np.searchsorted(self.classes_, y)]
        sample_weight = sample_weight / sample_weight.mean()
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return np.asarray(self.estimator_.predict(X)).ravel()   # CatBoost returns a column

    @available_if(lambda self: hasattr(self.estimator, "predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(X)


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ---------------------------------------------------------------------------
# Models and search spaces
# ---------------------------------------------------------------------------
# Hyperparameters of the classifier itself; the pipeline prefix is added in search_space()
PARAM_DISTS = {
    "SVM (RBF)": {
        "C":     loguniform(0.1, 100),
        "gamma": loguniform(1e-4, 1e-1),
    },
    "XGBoost": {
        "n_estimators":     randint(200, 600),
        "learning_rate":    loguniform(0.03, 0.2),
        "max_depth":        randint(3, 8),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.4, 0.6),
        "min_child_weight": randint(1, 10),
    },
    "CatBoost": {
        "iterations":    randint(200, 600),
        "learning_rate": loguniform(0.03, 0.2),
        "depth":         randint(4, 8),
        "l2_leaf_reg":   loguniform(1, 10),
    },
}
N_ITER = {"SVM (RBF)": 40, "XGBoost": 20, "CatBoost": 15}


def build_models(selector="passthrough", weighted=False):
    """The 6 standard models, each as a pipeline: scaler -> selector -> classifier."""
    def pipe(clf):
        select = selector if selector == "passthrough" else clone(selector)
        if weighted:
            clf = WeightedClassifier(clf)
        return Pipeline([("scaler", StandardScaler()), ("select", select), ("clf", clf)])

    return {
        "LogReg":        pipe(LogisticRegression(max_iter=5000)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=RANDOM_STATE)),
        "HistGradBoost": pipe(HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        "SVM (RBF)":     pipe(SVC(kernel="rbf")),
        "XGBoost":       pipe(XGBClassifier(n_jobs=1, random_state=RANDOM_STATE)),
        "CatBoost":      pipe(CatBoostClassifier(verbose=0, thread_count=1, allow_writing_files=False,
                                                 random_state=RANDOM_STATE)),
    }


def search_space(model_name, weighted=False):
    """Hyperparameters to search for one model (empty dict = nothing to search)."""
    prefix = "clf__estimator__" if weighted else "clf__"
    space = {prefix + k: v for k, v in PARAM_DISTS.get(model_name, {}).items()}
    if weighted:
        space["clf__alpha"] = ALPHAS
    return space

# ---------------------------------------------------------------------------
# One experiment
# ---------------------------------------------------------------------------
def run_experiment(name, X, y, task="binary", selector="passthrough", weighted=False, only=None, cv=CV):
    """
    Evaluate the standard models with cross-validation on (X, y).
      task:     "binary" or "multiclass" (decides the metrics and how the best configuration is chosen)
      selector: feature selection / reduction step inside the pipeline (e.g. SelectKBest, PCA)
      weighted: train with class weights; their strength alpha is searched in ALPHAS
      only:     list of model names to run (default: all 6)
      cv:       a CV splitter, or a list of (train_idx, test_idx) pairs (used for augmentation later)
    SVM, XGBoost and CatBoost are tuned with RandomizedSearchCV. The other 3 keep their default
    hyperparameters (only alpha is searched when weighted=True).
    Returns: results table (one row per model) and the fitted models (best configuration, trained on all of X).
    """
    scoring, refit = TASKS[task]
    models = build_models(selector, weighted)
    if only is not None:
        unknown = set(only) - set(models)
        assert not unknown, f"Unknown model names: {unknown}"

    rows, fitted = [], {}
    for model_name, model in models.items():
        if only is not None and model_name not in only:
            continue
        space = search_space(model_name, weighted)

        if model_name in PARAM_DISTS:          # tuned models
            search = RandomizedSearchCV(model, space, n_iter=N_ITER[model_name], scoring=scoring,
                                        refit=refit, cv=cv, n_jobs=-2, random_state=RANDOM_STATE)
        elif space:                            # default models, only alpha to choose
            search = GridSearchCV(model, space, scoring=scoring, refit=refit, cv=cv, n_jobs=-2)
        else:                                  # default models, nothing to choose
            search = None

        if search is not None:
            search.fit(X, y)
            r, i = search.cv_results_, search.best_index_
            means = {m: r[f"mean_test_{m}"][i] for m in scoring}
            stds = {m: r[f"std_test_{m}"][i] for m in scoring}
            fitted[model_name] = search.best_estimator_
            params = search.best_params_
        else:
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-2)
            means = {m: scores[f"test_{m}"].mean() for m in scoring}
            stds = {m: scores[f"test_{m}"].std() for m in scoring}
            fitted[model_name] = clone(model).fit(X, y)
            params = {}

        rows.append({"experiment": name, "model": model_name, **means,
                     **{f"{m}_std": s for m, s in stds.items()}, "params": params})
        print(f"[{name}] {model_name}: {refit} = {means[refit]:.3f}")

    return pd.DataFrame(rows), fitted


def show(table, task="binary"):
    """Readable 'mean ± std' view of a results table."""
    scoring, _ = TASKS[task]
    out = table[["experiment", "model"]].copy()
    for m in scoring:
        out[m] = table[m].map("{:.3f}".format) + " ± " + table[f"{m}_std"].map("{:.3f}".format)
    return out


def evaluate(model, X, y, task="binary"):
    """
    Metrics of an already fitted model on (X, y), with the same scorers as in CV.
    Only for the final evaluation: the validation set stays closed until the final model is chosen.
    """
    scoring, _ = TASKS[task]
    return {m: get_scorer(s)(model, X, y) for m, s in scoring.items()}

def oof_predictions(model, X, y, cv=CV):
    """
    Out-of-fold predictions and scores: every image is predicted by a copy of `model`
    trained on the other folds. Scores = decision_function if available, else predict_proba.
    """
    n_classes = len(np.unique(y))
    pred = np.empty_like(y)
    scores = np.zeros((len(y), n_classes)) if n_classes > 2 else np.zeros(len(y))
    for train_idx, test_idx in (cv.split(X, y) if hasattr(cv, "split") else cv):
        m = clone(model).fit(X[train_idx], y[train_idx])
        pred[test_idx] = np.asarray(m.predict(X[test_idx])).ravel()
        if hasattr(m, "decision_function"):
            scores[test_idx] = m.decision_function(X[test_idx])
        else:
            proba = m.predict_proba(X[test_idx])
            scores[test_idx] = proba if n_classes > 2 else proba[:, 1]
    return pred, scores

def augmented_cv(filenames, aug_sources, y, cv=CV):
    """
    CV folds for a dataset made of original rows followed by augmented rows.
      filenames:   filenames of the original images (rows 0 .. n-1)
      aug_sources: for every augmented row (rows n ..), the filename of the original it was made from
    Folds are made on the originals only (same folds as without augmentation). In each fold,
    the augmented copies of the TRAINING images are added to the training part; the test part
    only contains original images. Returns a list of (train_idx, test_idx) for run_experiment(cv=...).
    """
    filenames = np.asarray(filenames)
    n = len(filenames)
    position = {f: i for i, f in enumerate(filenames)}
    source_idx = np.array([position[s] for s in aug_sources])     # original row of each copy
    y_orig = np.asarray(y)[:n]

    splits = []
    for train_idx, test_idx in cv.split(np.zeros(n), y_orig):
        copies = n + np.flatnonzero(np.isin(source_idx, train_idx))
        splits.append((np.concatenate([train_idx, copies]), test_idx))
    return splits

# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------
from sklearn.ensemble import VotingClassifier, StackingClassifier

ENSEMBLE_KEYS = {"SVM (RBF)": "svm", "XGBoost": "xgb", "CatBoost": "cat", "HistGradBoost": "hgb",
                 "Random Forest": "rf", "LogReg": "logreg"}


def with_probabilities(model):
    """Copy of a pipeline whose SVM (if any) also outputs probabilities (needed for soft voting)."""
    model = clone(model)
    for key in ("clf__probability", "clf__estimator__probability"):
        if key in model.get_params():
            model.set_params(**{key: True})
    return model


def build_ensembles(base_models):
    """
    Soft voting and stacking of already configured models (their hyperparameters stay fixed).
      base_models: dict model name -> pipeline (e.g. the fitted models returned by run_experiment)
    """
    estimators = [(ENSEMBLE_KEYS.get(name, name), clone(m)) for name, m in base_models.items()]
    return {
        "Soft voting": VotingClassifier(
            [(key, with_probabilities(m)) for key, m in estimators], voting="soft"),
        "Stacking (LogReg)": StackingClassifier(
            estimators, final_estimator=LogisticRegression(max_iter=5000),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1), n_jobs=1),
    }


def run_ensembles(name, base_models, X, y, task="binary", cv=CV):
    """
    Cross-validate soft voting and stacking of base_models with the standard folds and metrics.
    Returns a results table (same format as run_experiment) and the ensembles trained on all of X.
    """
    scoring, refit = TASKS[task]
    members = " + ".join(base_models)
    rows, fitted = [], {}
    for ens_name, ens in build_ensembles(base_models).items():
        scores = cross_validate(ens, X, y, cv=cv, scoring=scoring, n_jobs=-2)
        means = {m: scores[f"test_{m}"].mean() for m in scoring}
        stds = {m: scores[f"test_{m}"].std() for m in scoring}
        fitted[ens_name] = clone(ens).fit(X, y)
        rows.append({"experiment": name, "model": ens_name, **means,
                     **{f"{m}_std": s for m, s in stds.items()}, "params": {"members": members}})
        print(f"[{name}] {ens_name}: {refit} = {means[refit]:.3f}")
    return pd.DataFrame(rows), fitted