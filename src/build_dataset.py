"""
Build the preprocessed images and the Priority 1 feature CSVs of one challenge.
  binary:     full train (15195 images) + val. The train subset (5170) is a row filter of the full train features.
  multiclass: train + val, preprocessed as an independent dataset (same preprocessing parameters as binary).

Preprocessed images that already exist are skipped, and feature CSVs that already exist are not recomputed
(use --force to recompute them), so the script can be stopped and run again.

Run from the project root:
    uv run python -m src.build_dataset binary
    uv run python -m src.build_dataset multiclass
"""
import argparse
import pandas as pd
from src.config import *
from src.utils import build_ground_truth
from src.preprocessing import preprocess_dataset
from src.feature_extraction import extract_dataset_features
import cv2
import numpy as np
from joblib import Parallel, delayed
from src.utils import load_preprocessed
from src.feature_extraction import extract_features
from src.augmentation import augment_image, COPY_TRANSFORMS


def sample_subset(gt, n_nevus, others_per_subtype, seed):
    """n_nevus random nevus images + a fixed number of random images per 'others' subtype (None = all)."""
    gt = gt.copy()
    gt["subtype"] = gt["filename"].str.replace(r"\d+\.jpg$", "", regex=True)
    parts = [gt[gt["label"] == 0].sample(n_nevus, random_state=seed)]
    for subtype, n in others_per_subtype.items():
        available = gt[gt["subtype"] == subtype]
        n = len(available) if n is None or n > len(available) else n
        parts.append(available.sample(n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def ground_truth(csv_path, data_dir, class_folders):
    """Read the ground-truth CSV if it exists, otherwise build it from the class folders and save it."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    gt = build_ground_truth(data_dir, class_folders)
    gt.to_csv(csv_path, index=False)
    return gt


def build_split(name, gt, data_dir, preproc_dir, features_csv, class_folders, force):
    """Preprocess one split and extract its features."""
    print(f"\n[{name}] Preprocessing {len(gt)} images (existing ones are skipped)")
    preprocess_dataset(gt, data_dir, preproc_dir, PREPROCESSING_PARAMS, class_folders)

    if features_csv.exists() and not force:
        print(f"[{name}] {features_csv.name} already exists, not recomputed (use --force)")
        return
    print(f"[{name}] Extracting features")
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    extract_dataset_features(gt, preproc_dir, class_folders).to_csv(features_csv, index=False)


def build_binary(force):
    gt_train = ground_truth(BINARY_GT_TRAIN_CSV, BINARY_TRAIN_DIR, BINARY_CLASS_FOLDERS)
    gt_val = ground_truth(BINARY_GT_VAL_CSV, BINARY_VAL_DIR, BINARY_CLASS_FOLDERS)

    build_split("binary train", gt_train, BINARY_TRAIN_DIR, BINARY_PREPROC_TRAIN_DIR,
                BINARY_FEATURES_TRAIN_FULL_CSV, BINARY_CLASS_FOLDERS, force)
    build_split("binary val", gt_val, BINARY_VAL_DIR, BINARY_PREPROC_VAL_DIR,
                BINARY_FEATURES_VAL_CSV, BINARY_CLASS_FOLDERS, force)

    # Train subset: fixed by RANDOM_STATE, created only once (all subset experiments depend on it)
    if BINARY_GT_TRAIN_SUBSET_CSV.exists():
        gt_subset = pd.read_csv(BINARY_GT_TRAIN_SUBSET_CSV)
    else:
        gt_subset = sample_subset(gt_train, SUBSET_N_NEVUS, SUBSET_OTHERS_PER_SUBTYPE, RANDOM_STATE)
        gt_subset.to_csv(BINARY_GT_TRAIN_SUBSET_CSV, index=False)

    if not BINARY_FEATURES_TRAIN_CSV.exists() or force:
        full = pd.read_csv(BINARY_FEATURES_TRAIN_FULL_CSV).set_index("filename")
        subset = full.loc[gt_subset["filename"]].reset_index()   # same row order as the subset CSV
        subset.to_csv(BINARY_FEATURES_TRAIN_CSV, index=False)
        print(f"[binary subset] {len(subset)} rows written to {BINARY_FEATURES_TRAIN_CSV.name}")


def build_multiclass(force):
    gt_train = ground_truth(MULTI_GT_TRAIN_CSV, MULTI_TRAIN_DIR, MULTI_CLASS_FOLDERS)
    gt_val = ground_truth(MULTI_GT_VAL_CSV, MULTI_VAL_DIR, MULTI_CLASS_FOLDERS)

    build_split("multiclass train", gt_train, MULTI_TRAIN_DIR, MULTI_PREPROC_TRAIN_DIR,
                MULTI_FEATURES_TRAIN_CSV, MULTI_CLASS_FOLDERS, force)
    build_split("multiclass val", gt_val, MULTI_VAL_DIR, MULTI_PREPROC_VAL_DIR,
                MULTI_FEATURES_VAL_CSV, MULTI_CLASS_FOLDERS, force)


def augment_one(index, filename, label, n_copies):
    """n_copies augmented versions of one preprocessed training image, one transform each (COPY_TRANSFORMS)."""
    assert n_copies <= len(COPY_TRANSFORMS), "more copies than transforms in COPY_TRANSFORMS"
    rng = np.random.default_rng([RANDOM_STATE, index])          # reproducible, independent per image
    img = load_preprocessed(filename, label, MULTI_PREPROC_TRAIN_DIR, MULTI_CLASS_FOLDERS)
    rows = []
    for k, (transform, params) in enumerate(COPY_TRANSFORMS[:n_copies]):
        aug = augment_image(img, rng, params)
        name = filename.replace(".jpg", f"_aug{k}_{transform}.png")
        out_path = MULTI_AUG_DIR / MULTI_CLASS_FOLDERS[label] / name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
        rows.append({"filename": name, "source_filename": filename, "transform": transform,
                     "label": label, **extract_features(aug)})
    return rows


def build_augmented(force):
    """Augmented copies of the multiclass TRAINING images of the classes in AUG_COPIES (never val)."""
    if MULTI_FEATURES_AUG_CSV.exists() and not force:
        print(f"{MULTI_FEATURES_AUG_CSV.name} already exists, not recomputed (use --force)")
        return
    gt = pd.read_csv(MULTI_GT_TRAIN_CSV)
    todo = [(i, r.filename, r.label, AUG_COPIES[r.label])
            for i, r in enumerate(gt.itertuples()) if r.label in AUG_COPIES]
    print(f"Augmenting {len(todo)} training images")
    rows = Parallel(n_jobs=-2, verbose=5)(delayed(augment_one)(*t) for t in todo)
    features = pd.DataFrame([row for image_rows in rows for row in image_rows])
    features.to_csv(MULTI_FEATURES_AUG_CSV, index=False)
    print(f"{len(features)} augmented images -> {MULTI_FEATURES_AUG_CSV.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images and extract features for one challenge.")
    parser.add_argument("task", choices=["binary", "multiclass", "augment"])
    parser.add_argument("--force", action="store_true", help="recompute feature CSVs that already exist")
    args = parser.parse_args()

    {"binary": build_binary, "multiclass": build_multiclass, "augment": build_augmented}[args.task](args.force)
    print("\nDone.")