"""Run the standard 6-model comparison for one experiment.
   This comes from the exploratory notebook baseline_classification.ipynb."""
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score
from scipy.stats import loguniform, randint, uniform
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.config import RANDOM_STATE

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

SCORING = {
    "accuracy":          "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "sensitivity":       make_scorer(recall_score, pos_label=1),
    "specificity":       make_scorer(recall_score, pos_label=0),
    "auc":               "roc_auc",
}

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


def run_experiment(name, X, y, selector="passthrough"):
    """
    Evaluate the 6 models with 5-fold CV (3 of them tuned).
    Returns: results table (one row per model) and the fitted models (trained on all of X).
    """
    rows, fitted = [], {}
    for model_name, model in build_models(selector).items():
        if model_name in PARAM_DISTS:
            search = RandomizedSearchCV(model, PARAM_DISTS[model_name], n_iter=N_ITER[model_name],
                                        scoring=SCORING, refit="balanced_accuracy", cv=CV,
                                        n_jobs=-2, random_state=RANDOM_STATE)
            search.fit(X, y)
            r, i = search.cv_results_, search.best_index_
            means = {m: r[f"mean_test_{m}"][i] for m in SCORING}
            stds = {m: r[f"std_test_{m}"][i] for m in SCORING}
            fitted[model_name] = search.best_estimator_
        else:
            scores = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=-2)
            means = {m: scores[f"test_{m}"].mean() for m in SCORING}
            stds = {m: scores[f"test_{m}"].std() for m in SCORING}
            fitted[model_name] = model.fit(X, y)

        rows.append({"experiment": name, "model": model_name,
                     **means, **{f"{m}_std": s for m, s in stds.items()}})
        print(f"[{name}] {model_name}: balanced acc = {means['balanced_accuracy']:.3f}")

    return pd.DataFrame(rows), fitted


def show(table):
    """Readable 'mean ± std' view of a results table."""
    out = table[["experiment", "model"]].copy()
    for m in SCORING:
        out[m] = table[m].map("{:.3f}".format) + " ± " + table[f"{m}_std"].map("{:.3f}".format)
    return out