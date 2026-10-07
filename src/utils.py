import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
from src.config import *
from skimage.segmentation import clear_border
from skimage.morphology import skeletonize

def random_sample(df, n=9, seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000)
    print("seed:", seed)
    return df.sample(n, random_state=seed)

def load_image(filename, label, data_dir=BINARY_TRAIN_DIR):
    path = data_dir / BINARY_CLASS_FOLDERS[label] / filename
    img = cv2.imread(str(path))                  # loads as BGR
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

def load_preprocessed(filename, label, preproc_dir):
    path = preproc_dir / BINARY_CLASS_FOLDERS[label] / filename.replace(".jpg", ".png")
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def build_ground_truth(data_dir):
    rows = []
    for label, folder in BINARY_CLASS_FOLDERS.items():
        for path in sorted((data_dir / folder).glob("*.jpg")):
            rows.append({"filename": path.name, "label": label})
    return pd.DataFrame(rows)

def show_grid(images, titles):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(df, cols, ncols=6, bins=30):
    """One small histogram per feature, nevus vs others overlaid."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))

    for ax, col in zip(axes.flat, cols):
        bin_edges = np.histogram_bin_edges(df[col].dropna(), bins=bins)   # same bins for both classes
        for label in [0, 1]:
            values = df.loc[df["label"] == label, col]
            ax.hist(values, bins=bin_edges, density=True, histtype="step", linewidth=2,
                    color=CLASS_COLORS[label], label=CLASS_NAMES[label])
        ax.set_title(col, fontsize=9)
        ax.set_yticks([])

    for ax in axes.flat[len(cols):]:          # hide empty panels
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

UNIT_CONVERSIONS = {
    "hsv_h": (lambda v: v * 2,          "Hue (°)"),
    "hsv_s": (lambda v: v / 255 * 100,  "Saturation (%)"),
    "hsv_v": (lambda v: v / 255 * 100,  "Value (%)"),
    "lab_l": (lambda v: v / 255 * 100,  "L* (0-100)"),
    "lab_a": (lambda v: v - 128,        "a*  (green -  /  red +)"),
    "lab_b": (lambda v: v - 128,        "b*  (blue -  /  yellow +)"),
}


def histogram_x_axis(prefix, n_bins):
    """x positions and axis label for the bins of one histogram feature group."""
    if prefix.startswith("hist_"):
        channel = prefix[len("hist_"):].rstrip("_")           # 'hist_lab_a_' -> 'lab_a'
        low, high = HIST_RANGES[channel]
        edges = np.linspace(low, high, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2                 # centre of each bin
        convert, label = UNIT_CONVERSIONS[channel]
        return convert(centers), label

    # LBP: bin i = number of neighbours >= centre pixel (uniform patterns), last bin = non-uniform
    return np.arange(n_bins), "LBP code (neighbours >= centre; last = non-uniform)"

def plot_mean_histograms(df, prefixes, ncols=3):
    """Average histogram per class (line) +- 1 std (band), one panel per prefix."""
    nrows = int(np.ceil(len(prefixes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)

    for ax, prefix in zip(axes.flat, prefixes):
        cols = [c for c in df.columns if c.startswith(prefix)]
        x, xlabel = histogram_x_axis(prefix, len(cols))
        for label in [0, 1]:
            values = df.loc[df["label"] == label, cols]
            mean, std = values.mean().values, values.std().values
            ax.plot(x, mean, linewidth=2, color=CLASS_COLORS[label], label=CLASS_NAMES[label])
            ax.fill_between(x, mean - std, mean + std, color=CLASS_COLORS[label], alpha=0.2)
        ax.set_title(prefix.rstrip("_"), fontsize=10)
        ax.set_xlabel(xlabel)

    for ax in axes.flat[len(prefixes):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()