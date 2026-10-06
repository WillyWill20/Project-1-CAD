"""
Preprocess the full binary dataset (train + val) and save the results as PNG.

Run from the project root:
    uv run python -m src.generate_processed_dataset
"""
import pandas as pd
from src.config import *
from src.utils import build_ground_truth, preprocess_dataset


def main():
    # Ground truth
    gt_train = pd.read_csv(BINARY_GT_TRAIN_CSV)
    gt_val = build_ground_truth(BINARY_VAL_DIR)
    gt_val.to_csv(BINARY_GT_VAL_CSV, index=False)

    # Train
    print(f"Preprocessing train: {len(gt_train)} images")
    preprocess_dataset(gt_train, BINARY_TRAIN_DIR, BINARY_PREPROC_TRAIN_DIR, **PREPROCESSING_PARAMS)

    # Validation
    print(f"Preprocessing val: {len(gt_val)} images")
    preprocess_dataset(gt_val, BINARY_VAL_DIR, BINARY_PREPROC_VAL_DIR, **PREPROCESSING_PARAMS)

    print("Done.")


if __name__ == "__main__":
    main()
    
"""
Command to run the file:
        uv run python -m src.generate_processed_dataset
"""