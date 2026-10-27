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
RESULTS_DIR   = ROOT_DIR / "ml-results"

# ---------------------------------------------------------------------------
# Challenge 1 - Binary (nevus vs others)
# ---------------------------------------------------------------------------
BINARY_DATASET_DIR = DATASET_DIR / "binary"
BINARY_TRAIN_DIR   = BINARY_DATASET_DIR / "train"
BINARY_VAL_DIR     = BINARY_DATASET_DIR / "val"
BINARY_GT_TRAIN_CSV = BINARY_DATASET_DIR / "train_ground_truth.csv"

BINARY_CLASS_FOLDERS = {0: "nevus", 1: "others"}

BINARY_GT_VAL_CSV = BINARY_DATASET_DIR / "val_ground_truth.csv"
BINARY_GT_TRAIN_SUBSET_CSV = BINARY_DATASET_DIR / "train_subset_ground_truth.csv"

BINARY_PREPROC_DIR       = BINARY_DATASET_DIR / "preprocessed"
BINARY_PREPROC_TRAIN_DIR = BINARY_PREPROC_DIR / "train"
BINARY_PREPROC_VAL_DIR   = BINARY_PREPROC_DIR / "val"

PREPROCESSING_PARAMS = dict(
    resizing_dim=450,   # shorter side after vignette crop: 2.34% of train images < 450
    kernel_size=17,     # hair removal: black-hat kernel
    threshold=10,       # hair removal: black-hat threshold
    min_length=40,      # hair removal: min skeleton length of a hair (px)
    max_width=10,       # hair removal: max average width of a hair (px)
)

SUBSET_N_NEVUS = 2500
SUBSET_OTHERS_PER_SUBTYPE = {
    "ack": 500,
    "bcc": 500,
    "bkl": 500,
    "mel": 500,
    "scc": None,   # None = take all (376)
    "def": None,   # take all (143)
    "vac": None,   # take all (151)
}

FEATURES_1_DIR = ROOT_DIR / "features_1"
BINARY_FEATURES_TRAIN_CSV = FEATURES_1_DIR / "binary_train_features.csv"
BINARY_FEATURES_VAL_CSV   = FEATURES_1_DIR / "binary_val_features.csv"
BINARY_FEATURES_TRAIN_FULL_CSV = FEATURES_1_DIR / "binary_train_full_features.csv"   # all 15195 train images

CLASS_NAMES  = {0: "nevus", 1: "others"}
CLASS_COLORS = {0: "#3987e5", 1: "#d95926"}   # blue / orange

HIST_RANGES = {
    "hsv_h": (0, 180),
    "hsv_s": (0, 159),  # Values obtained from looking at 400 images
    "hsv_v": (0, 256),
    "lab_l": (0, 256),
    "lab_a": (120, 158), # Values obtained from looking at 400 images
    "lab_b": (117, 156), # VValues obtained from looking at 400 images
}

MODELS_DIR = ROOT_DIR / "models"
MODEL_COLORS = {
    "LogReg":        "#3987e5",   # blue
    "SVM (RBF)":     "#d95926",   # orange
    "Random Forest": "#199e70",   # aqua
    "HistGradBoost": "#c98500",   # yellow
    "XGBoost":       "#d55181",   # magenta
    "CatBoost":      "#008300",   # green
}

# Variant: same pipeline without hair removal (experiment 2)
PREPROCESSING_PARAMS_NOHAIR = {**PREPROCESSING_PARAMS, "remove_hairs": False}

BINARY_PREPROC_NOHAIR_DIR       = BINARY_DATASET_DIR / "preprocessed_nohair"
BINARY_PREPROC_NOHAIR_TRAIN_DIR = BINARY_PREPROC_NOHAIR_DIR / "train"
BINARY_PREPROC_NOHAIR_VAL_DIR   = BINARY_PREPROC_NOHAIR_DIR / "val"

FEATURES_2_DIR = ROOT_DIR / "features_2_nohair"
BINARY_FEATURES_NOHAIR_TRAIN_CSV = FEATURES_2_DIR / "binary_train_features.csv"
BINARY_FEATURES_NOHAIR_VAL_CSV   = FEATURES_2_DIR / "binary_val_features.csv"

# ---------------------------------------------------------------------------
# Challenge 2 - Multiclass (bcc / mel / scc)
# ---------------------------------------------------------------------------
MULTI_DATASET_DIR = DATASET_DIR / "multiclass"
MULTI_TRAIN_DIR   = MULTI_DATASET_DIR / "train"
MULTI_VAL_DIR     = MULTI_DATASET_DIR / "val"
MULTI_GT_TRAIN_CSV = MULTI_DATASET_DIR / "train_ground_truth.csv"
MULTI_GT_VAL_CSV   = MULTI_DATASET_DIR / "val_ground_truth.csv"

# Label encoding (order of the assignment: melanoma vs basal cell vs squamous cell carcinoma).
# Change it only here if the test submission asks for another one.
MULTI_CLASS_FOLDERS = {0: "mel", 1: "bcc", 2: "scc"}
MULTI_CLASS_NAMES   = {0: "mel", 1: "bcc", 2: "scc"}
MULTI_CLASS_COLORS  = {0: "#3987e5", 1: "#d95926", 2: "#199e70"}   # blue / orange / aqua

# Multiclass: preprocessed and feature-extracted as an independent dataset,
# with the same preprocessing parameters as the binary problem (PREPROCESSING_PARAMS, HIST_RANGES)
MULTI_PREPROC_DIR       = MULTI_DATASET_DIR / "preprocessed"
MULTI_PREPROC_TRAIN_DIR = MULTI_PREPROC_DIR / "train"
MULTI_PREPROC_VAL_DIR   = MULTI_PREPROC_DIR / "val"

FEATURES_MULTI_DIR       = ROOT_DIR / "features_multiclass"
MULTI_FEATURES_TRAIN_CSV = FEATURES_MULTI_DIR / "multi_train_features.csv"
MULTI_FEATURES_VAL_CSV   = FEATURES_MULTI_DIR / "multi_val_features.csv"

# Data augmentation (Step 4): class label -> augmented copies per training image (scc only)
AUG_COPIES = {2: 4}
MULTI_AUG_DIR          = MULTI_DATASET_DIR / "augmented" / "train"
MULTI_FEATURES_AUG_CSV = FEATURES_MULTI_DIR / "multi_train_aug_features.csv"

# --- Extra features (G1 regional + HoG), binary full train + val ---
FEATURES_3_DIR = ROOT_DIR / "features_3_extra"
BINARY_EXTRA_TRAIN_FULL_CSV = FEATURES_3_DIR / "binary_train_full_extra.csv"
BINARY_EXTRA_VAL_CSV        = FEATURES_3_DIR / "binary_val_extra.csv"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
