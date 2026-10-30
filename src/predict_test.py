"""
Predictions on the test sets for the submission.
  1. Lists the test images and sorts them by the number in their filename (order required by the submission).
  2. Preprocesses them exactly as train/val (saved as PNG; existing ones are reused).
  3. Extracts base + regional features (saved as CSV; not recomputed unless --force).
  4. Predicts with the saved final model and writes a 1-column CSV without header.

Run from the project root:
    uv run python -m src.predict_test binary
    uv run python -m src.predict_test multiclass
"""
import argparse
import re
import cv2
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src.config import *
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_features, regional_features

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

TASKS = {
    "binary": dict(test_dir=BINARY_TEST_DIR, preproc_dir=BINARY_PREPROC_TEST_DIR,
                   features_csv=BINARY_FEATURES_TEST_CSV, model=MODELS_DIR / "binary_final_stacking_f1.joblib",
                   class_names=CLASS_NAMES),
    "multiclass": dict(test_dir=MULTI_TEST_DIR, preproc_dir=MULTI_PREPROC_TEST_DIR,
                       features_csv=MULTI_FEATURES_TEST_CSV, model=MODELS_DIR / "multi_final.joblib",
                       class_names=MULTI_CLASS_NAMES),
}


def image_number(path):
    """Number in the filename (the last group of digits), used to order the submission."""
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"No number in the filename: {path.name}")
    return int(numbers[-1])


def list_test_images(test_dir):
    """All images in test_dir (also inside subfolders), sorted by the number in their filename."""
    paths = [p for p in test_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise FileNotFoundError(f"No images found in {test_dir}")
    paths.sort(key=lambda p: (image_number(p), p.name))
    numbers = [image_number(p) for p in paths]
    if len(set(numbers)) != len(numbers):
        raise ValueError("Two images have the same number: the submission order would be ambiguous")
    return paths


def features_of_test_image(path, preproc_dir):
    """Preprocess one test image (or reuse its saved PNG) and extract base + regional features."""
    out_path = preproc_dir / (path.stem + ".png")
    if out_path.exists():
        img = cv2.cvtColor(cv2.imread(str(out_path)), cv2.COLOR_BGR2RGB)
    else:
        raw = cv2.imread(str(path))
        if raw is None:
            raise FileNotFoundError(f"Could not read {path}")
        img = preprocess_image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB), **PREPROCESSING_PARAMS)
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))   # PNG: lossless, same as train/val
    return {"filename": path.name, **extract_features(img), **regional_features(img)}


def predict(task, force):
    cfg = TASKS[task]

    # 1. Images in submission order
    paths = list_test_images(cfg["test_dir"])
    numbers = [image_number(p) for p in paths]
    print(f"[{task}] {len(paths)} test images, numbers {numbers[0]} .. {numbers[-1]}")
    print("  first:", [p.name for p in paths[:3]], "| last:", [p.name for p in paths[-3:]])
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        print("  WARNING: the numbers are not consecutive - check that this is the expected order")

    # 2-3. Preprocessing + features (cached)
    if cfg["features_csv"].exists() and not force:
        features = pd.read_csv(cfg["features_csv"])
        assert features["filename"].tolist() == [p.name for p in paths], \
            "Cached features do not match the test images (run again with --force)"
        print(f"  features read from {cfg['features_csv'].name}")
    else:
        cfg["preproc_dir"].mkdir(parents=True, exist_ok=True)
        rows = Parallel(n_jobs=-2, verbose=5)(delayed(features_of_test_image)(p, cfg["preproc_dir"]) for p in paths)
        features = pd.DataFrame(rows)
        cfg["features_csv"].parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(cfg["features_csv"], index=False)

    # 4. Predictions with the final model (same 442 columns, same order as in training)
    saved = joblib.load(cfg["model"])
    X = features[saved["features"]].values
    assert np.isfinite(X).all(), "NaN or inf in the test features"
    pred = saved["model"].predict(X).astype(int)

    SUBMISSION_DIR.mkdir(exist_ok=True)
    out_csv = SUBMISSION_DIR / f"{task}_predictions.csv"
    pd.Series(pred).to_csv(out_csv, index=False, header=False)                     # file to submit
    pd.DataFrame({"filename": features["filename"], "prediction": pred,             # for our own checks
                  "class": [cfg["class_names"][k] for k in pred]}).to_csv(
        SUBMISSION_DIR / f"{task}_predictions_with_filenames.csv", index=False)

    counts = pd.Series(pred).map(cfg["class_names"]).value_counts()
    print(f"\n  {len(pred)} predictions -> {out_csv}")
    print("  predicted classes (%):", (100 * counts / len(pred)).round(1).to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the test set of one challenge for the submission.")
    parser.add_argument("task", choices=["binary", "multiclass"])
    parser.add_argument("--force", action="store_true", help="recompute the test features")
    args = parser.parse_args()
    predict(args.task, args.force)