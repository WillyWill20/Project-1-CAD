"""Run the standard 6-model comparison for one experiment.
   This comes from the exploratory notebook baseline-classification.ipynb."""
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import (make_scorer, recall_score, accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score)
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import loguniform, randint, uniform
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.config import RANDOM_STATE, MULTI_CLASS_NAMES

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

SCORING = {
    "accuracy":          "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "sensitivity":       make_scorer(recall_score, pos_label=1),
    "specificity":       make_scorer(recall_score, pos_label=0),
    "auc":               "roc_auc",
}

# Challenge 2 (mel / bcc / scc). No AUC: the SVM has no predict_proba.
# Per-class recall matters because scc is only ~7% of the images.
MULTI_SCORING = {
    "accuracy":          "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "kappa":             make_scorer(cohen_kappa_score),
    "f1_macro":          "f1_macro",
    **{f"recall_{name}": make_scorer(recall_score, labels=[label], average="macro")
       for label, name in MULTI_CLASS_NAMES.items()},
}


def multiclass_metrics(y_true, y_pred):
    """Same metrics as MULTI_SCORING, computed from predictions (e.g. on the validation set)."""
    recalls = recall_score(y_true, y_pred, labels=list(MULTI_CLASS_NAMES), average=None)
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "kappa":             cohen_kappa_score(y_true, y_pred),
        "f1_macro":          f1_score(y_true, y_pred, average="macro"),
        **{f"recall_{name}": r for name, r in zip(MULTI_CLASS_NAMES.values(), recalls)},
    }


def balanced_fit_params(model, y):
    """Fit params that weight each sample by 1 / frequency of its class (same as class_weight='balanced').
    Passed as sample_weight to the last pipeline step, so it works for all 6 models."""
    return {f"{model.steps[-1][0]}__sample_weight": compute_sample_weight("balanced", y)}

# Search spaces (step names come from make_pipeline: lower-case class names)
PARAM_DISTS = {
    "SVM (RBF)": {
        "svc__C":     loguniform(0.1, 100),
        "svc__gamma": loguniform(1e-4, 1e-1),
    },
    "XGBoost": {
        "xgbclassifier__n_estimators":     randint(200, 600),
        "xgbclassifier__learning_rate":    loguniform(0.03, 0.2),
        "xgbclassifier__max_depth":        randint(3, 8),
        "xgbclassifier__subsample":        uniform(0.6, 0.4),
        "xgbclassifier__colsample_bytree": uniform(0.4, 0.6),
        "xgbclassifier__min_child_weight": randint(1, 10),
    },
    "CatBoost": {
        "catboostclassifier__iterations":    randint(200, 600),
        "catboostclassifier__learning_rate": loguniform(0.03, 0.2),
        "catboostclassifier__depth":         randint(4, 8),
        "catboostclassifier__l2_leaf_reg":   loguniform(1, 10),
    },
}
N_ITER = {"SVM (RBF)": 40, "XGBoost": 20, "CatBoost": 15}


def build_models(selector="passthrough"):
    """The 6 standard models, each as scaler -> selector -> classifier."""
    def pipe(clf):
        step = selector if selector == "passthrough" else clone(selector)
        return make_pipeline(StandardScaler(), step, clf)

    return {
        "LogReg":        pipe(LogisticRegression(max_iter=5000)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=RANDOM_STATE)),
        "HistGradBoost": pipe(HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        "SVM (RBF)":     pipe(SVC(kernel="rbf")),
        "XGBoost":       pipe(XGBClassifier(n_jobs=1, random_state=RANDOM_STATE)),
        "CatBoost":      pipe(CatBoostClassifier(verbose=0, thread_count=1, allow_writing_files=False, random_state=RANDOM_STATE)),
    }


def run_experiment(name, X, y, selector="passthrough", scoring=SCORING, balanced=False):
    """
    Evaluate the 6 models with 5-fold CV (3 of them tuned).
    balanced=True weights the classes by their inverse frequency (see balanced_fit_params).
    Returns: results table (one row per model) and the fitted models (trained on all of X).
    """
    rows, fitted = [], {}
    for model_name, model in build_models(selector).items():
        fit_params = balanced_fit_params(model, y) if balanced else {}
        if model_name in PARAM_DISTS:
            search = RandomizedSearchCV(model, PARAM_DISTS[model_name], n_iter=N_ITER[model_name],
                                        scoring=scoring, refit="balanced_accuracy", cv=CV,
                                        n_jobs=-2, random_state=RANDOM_STATE)
            search.fit(X, y, **fit_params)
            r, i = search.cv_results_, search.best_index_
            means = {m: r[f"mean_test_{m}"][i] for m in scoring}
            stds = {m: r[f"std_test_{m}"][i] for m in scoring}
            fitted[model_name] = search.best_estimator_
        else:
            scores = cross_validate(model, X, y, cv=CV, scoring=scoring, n_jobs=-2, params=fit_params)
            means = {m: scores[f"test_{m}"].mean() for m in scoring}
            stds = {m: scores[f"test_{m}"].std() for m in scoring}
            fitted[model_name] = model.fit(X, y, **fit_params)

        rows.append({"experiment": name, "model": model_name,
                     **means, **{f"{m}_std": s for m, s in stds.items()}})
        print(f"[{name}] {model_name}: balanced acc = {means['balanced_accuracy']:.3f}")

    return pd.DataFrame(rows), fitted


def show(table, scoring=SCORING):
    """Readable 'mean ± std' view of a results table."""
    out = table[["experiment", "model"]].copy()
    for m in scoring:
        out[m] = table[m].map("{:.3f}".format) + " ± " + table[f"{m}_std"].map("{:.3f}".format)
    return out