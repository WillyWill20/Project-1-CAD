"""
Extract features_1 from all preprocessed images and save them as CSV.

Run from the project root:
    uv run python -m src.generate_features
"""
import pandas as pd
from src.config import *
from src.utils import load_preprocessed
from src.feature_extraction import extract_features
from joblib import Parallel, delayed


def process_one(filename, label, preproc_dir):
    img = load_preprocessed(filename, label, preproc_dir)
    return {"filename": filename, "label": label, **extract_features(img)}


def extract_dataset_features(gt, preproc_dir, n_jobs=-2):
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_one)(r.filename, r.label, preproc_dir) for r in gt.itertuples()
    )
    return pd.DataFrame(rows)


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    gt_train = pd.read_csv(BINARY_GT_TRAIN_SUBSET_CSV)
    print(f"Train: {len(gt_train)} images")
    extract_dataset_features(gt_train, BINARY_PREPROC_TRAIN_DIR).to_csv(BINARY_FEATURES_TRAIN_CSV, index=False)

    gt_val = pd.read_csv(BINARY_GT_VAL_CSV)
    print(f"Val: {len(gt_val)} images")
    extract_dataset_features(gt_val, BINARY_PREPROC_VAL_DIR).to_csv(BINARY_FEATURES_VAL_CSV, index=False)

    print("Done.")


if __name__ == "__main__":
    main()

"""
Command to run the file:
        uv run python -m src.generate_features
"""