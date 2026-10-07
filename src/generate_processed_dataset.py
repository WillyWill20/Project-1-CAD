"""
Preprocess a balanced subset of the binary training set + the full validation set,
and save the results as PNG.

Run from the project root:
    uv run python -m src.generate_processed_dataset
"""
import pandas as pd
from src.config import *
from src.utils import *
from src.preprocessing import *


def sample_subset(gt, n_nevus, others_per_subtype, seed):
    """
    Take n_nevus random nevus images, and a fixed number of random images
    per 'others' subtype (subtype = filename prefix, e.g. 'bcc' in 'bcc00012.jpg').
    """
    gt = gt.copy()
    gt["subtype"] = gt["filename"].str.replace(r"\d+\.jpg$", "", regex=True)

    # Nevus
    parts = [gt[gt["label"] == 0].sample(n_nevus, random_state=seed)]

    # Others: n images per subtype (None = all of them)
    for subtype, n in others_per_subtype.items():
        available = gt[gt["subtype"] == subtype]
        if n is None or n > len(available):
            n = len(available)
        parts.append(available.sample(n, random_state=seed))

    return pd.concat(parts, ignore_index=True)


def main():
    # Ground truth
    gt_train = pd.read_csv(BINARY_GT_TRAIN_CSV)
    gt_val = build_ground_truth(BINARY_VAL_DIR)
    gt_val.to_csv(BINARY_GT_VAL_CSV, index=False)

    # Train subset
    gt_subset = sample_subset(gt_train, SUBSET_N_NEVUS, SUBSET_OTHERS_PER_SUBTYPE, RANDOM_STATE)
    gt_subset.to_csv(BINARY_GT_TRAIN_SUBSET_CSV, index=False)
    print(gt_subset["subtype"].value_counts(), "\n")

    print(f"Preprocessing train subset: {len(gt_subset)} images")
    preprocess_dataset(gt_subset, BINARY_TRAIN_DIR, BINARY_PREPROC_TRAIN_DIR, **PREPROCESSING_PARAMS)

    # Validation (full)
    print(f"Preprocessing val: {len(gt_val)} images")
    preprocess_dataset(gt_val, BINARY_VAL_DIR, BINARY_PREPROC_VAL_DIR, **PREPROCESSING_PARAMS)

    print("Done.")


if __name__ == "__main__":
    main()
    
"""
Command to run the file:
        uv run python -m src.generate_processed_dataset
"""