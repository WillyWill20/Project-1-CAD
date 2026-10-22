"""
Variant without hair removal (experiment 2): preprocess the same train subset + val,
then extract the Priority 1 features.

Run from the project root:
    uv run python -m src.generate_nohair_variant
"""
import pandas as pd
import cv2
from joblib import Parallel, delayed
from src.config import *
from src.utils import load_image
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_dataset_features


def preprocess_one(filename, label, data_dir, out_dir, params):
    out_path = out_dir / BINARY_CLASS_FOLDERS[label] / filename.replace(".jpg", ".png")
    if out_path.exists():                      # already done -> skip (lets you resume)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = preprocess_image(load_image(filename, label, data_dir), **params)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def preprocess_parallel(gt, data_dir, out_dir, params, n_jobs=-2):
    Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(preprocess_one)(r.filename, r.label, data_dir, out_dir, params) for r in gt.itertuples()
    )


def main():
    gt_train = pd.read_csv(BINARY_GT_TRAIN_SUBSET_CSV)   # same 5170 images as the baseline
    gt_val = pd.read_csv(BINARY_GT_VAL_CSV)

    print("Preprocessing without hair removal...")
    preprocess_parallel(gt_train, BINARY_TRAIN_DIR, BINARY_PREPROC_NOHAIR_TRAIN_DIR, PREPROCESSING_PARAMS_NOHAIR)
    preprocess_parallel(gt_val, BINARY_VAL_DIR, BINARY_PREPROC_NOHAIR_VAL_DIR, PREPROCESSING_PARAMS_NOHAIR)

    print("Extracting features...")
    FEATURES_2_DIR.mkdir(parents=True, exist_ok=True)
    extract_dataset_features(gt_train, BINARY_PREPROC_NOHAIR_TRAIN_DIR).to_csv(BINARY_FEATURES_NOHAIR_TRAIN_CSV, index=False)
    extract_dataset_features(gt_val, BINARY_PREPROC_NOHAIR_VAL_DIR).to_csv(BINARY_FEATURES_NOHAIR_VAL_CSV, index=False)
    print("Done.")


if __name__ == "__main__":
    main()