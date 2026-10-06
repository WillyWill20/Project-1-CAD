"""
Project-wide variables and paths. Import from anywhere with:  from src.config import *
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Top-level folders
# ---------------------------------------------------------------------------
DATASET_DIR   = ROOT_DIR / "dataset"
SRC_DIR       = ROOT_DIR / "src"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
RESULTS_DIR   = ROOT_DIR / "image-results"

# ---------------------------------------------------------------------------
# Challenge 1 - Binary (nevus vs others)
# ---------------------------------------------------------------------------
BINARY_DATASET_DIR = DATASET_DIR / "binary"
BINARY_TRAIN_DIR   = BINARY_DATASET_DIR / "train"
BINARY_VAL_DIR     = BINARY_DATASET_DIR / "val"
BINARY_GT_TRAIN_CSV = BINARY_DATASET_DIR / "train_ground_truth.csv"

# ---------------------------------------------------------------------------
# Challenge 2 - Multiclass (bcc / mel / scc)
# ---------------------------------------------------------------------------
MULTI_DATASET_DIR = DATASET_DIR / "multiclass"
MULTI_TRAIN_DIR   = MULTI_DATASET_DIR / "train"
MULTI_VAL_DIR     = MULTI_DATASET_DIR / "val"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
