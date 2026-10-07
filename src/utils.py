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

def plot_mean_histograms(df, prefixes, ncols=3):
    """Average histogram per class (line) +- 1 std (band), one panel per prefix."""
    nrows = int(np.ceil(len(prefixes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)

    for ax, prefix in zip(axes.flat, prefixes):
        cols = [c for c in df.columns if c.startswith(prefix)]
        x = np.arange(len(cols))
        for label in [0, 1]:
            values = df.loc[df["label"] == label, cols]
            mean, std = values.mean().values, values.std().values
            ax.plot(x, mean, linewidth=2, color=CLASS_COLORS[label], label=CLASS_NAMES[label])
            ax.fill_between(x, mean - std, mean + std, color=CLASS_COLORS[label], alpha=0.2)
        ax.set_title(prefix.rstrip("_"), fontsize=10)
        ax.set_xlabel("bin")

    for ax in axes.flat[len(prefixes):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()