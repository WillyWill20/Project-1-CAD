"""
Build the multiclass (mel / bcc / scc) ground truth and feature CSVs.

The multiclass images are the same files as the mel/bcc/scc images inside binary/{train,val}/others
(checked byte by byte), and those are already preprocessed and their features extracted.
So nothing is preprocessed here: the rows of the binary feature CSVs are filtered by filename
and relabelled with the multiclass labels (MULTI_CLASS_FOLDERS in config.py).

Run from the project root:
    uv run python -m src.generate_multiclass_features
"""
import pandas as pd
from src.config import *
from src.utils import build_ground_truth


def multiclass_features(gt, binary_features_csv):
    """Features of the images in gt, taken from a binary feature CSV, with the multiclass labels."""
    binary = pd.read_csv(binary_features_csv).rename(columns={"label": "binary_label"})
    missing = gt.loc[~gt["filename"].isin(binary["filename"]), "filename"]
    if len(missing):
        raise ValueError(f"{len(missing)} images have no features in {binary_features_csv.name}, e.g. {missing.iloc[0]}")

    out = gt.merge(binary, on="filename", validate="one_to_one")
    if (out["binary_label"] != 1).any():                     # they must all be 'others' in the binary problem
        raise ValueError("Some multiclass images are not labelled as 'others' in the binary CSV")

    return out.drop(columns="binary_label")


def check_labels(gt):
    """The filename prefix (mel/bcc/scc) must match the folder the image comes from."""
    prefix = gt["filename"].str.replace(r"\d+\.jpg$", "", regex=True)
    wrong = gt[prefix != gt["label"].map(MULTI_CLASS_FOLDERS)]
    if len(wrong):
        raise ValueError(f"{len(wrong)} filenames do not match their class folder, e.g. {wrong['filename'].iloc[0]}")


def main():
    FEATURES_1_DIR.mkdir(parents=True, exist_ok=True)
    splits = [
        ("Train", MULTI_TRAIN_DIR, MULTI_GT_TRAIN_CSV, BINARY_FEATURES_TRAIN_FULL_CSV, MULTI_FEATURES_TRAIN_CSV),
        ("Val",   MULTI_VAL_DIR,   MULTI_GT_VAL_CSV,   BINARY_FEATURES_VAL_CSV,        MULTI_FEATURES_VAL_CSV),
    ]
    for name, data_dir, gt_csv, binary_csv, out_csv in splits:
        gt = build_ground_truth(data_dir, MULTI_CLASS_FOLDERS)
        check_labels(gt)
        gt.to_csv(gt_csv, index=False)

        features = multiclass_features(gt, binary_csv)
        features.to_csv(out_csv, index=False)

        counts = gt["label"].map(MULTI_CLASS_NAMES).value_counts().to_dict()
        print(f"{name}: {len(features)} images {counts} -> {out_csv.name} ({features.shape[1] - 2} features)")

    print("Done.")


if __name__ == "__main__":
    main()
