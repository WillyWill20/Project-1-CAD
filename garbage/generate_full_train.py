"""
Preprocess the FULL binary training set (15195 images, not only the subset) and
extract its features into a separate CSV.

Images already preprocessed (e.g. the subset) are skipped, so only the missing ones are processed.

Run from the project root:
    uv run python -m src.generate_full_train
"""
import cv2
import pandas as pd
from joblib import Parallel, delayed
from src.config import *
from src.utils import load_image
from src.preprocessing import preprocess_image
from garbage.generate_features import extract_dataset_features


def preprocess_one(filename, label, data_dir, out_dir, params):
    """Preprocess one image and save it as PNG (same layout as preprocess_dataset)."""
    out_path = out_dir / BINARY_CLASS_FOLDERS[label] / filename.replace(".jpg", ".png")
    if out_path.exists():                # already done -> skip
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = preprocess_image(load_image(filename, label, data_dir), **params)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))   # OpenCV saves BGR


def main():
    gt_train = pd.read_csv(BINARY_GT_TRAIN_CSV)

    print(f"Preprocessing full train: {len(gt_train)} images (existing ones are skipped)")
    Parallel(n_jobs=-2, verbose=5)(
        delayed(preprocess_one)(r.filename, r.label, BINARY_TRAIN_DIR, BINARY_PREPROC_TRAIN_DIR, PREPROCESSING_PARAMS)
        for r in gt_train.itertuples()
    )

    print(f"Extracting features: {len(gt_train)} images")
    FEATURES_1_DIR.mkdir(parents=True, exist_ok=True)
    extract_dataset_features(gt_train, BINARY_PREPROC_TRAIN_DIR).to_csv(BINARY_FEATURES_TRAIN_FULL_CSV, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
