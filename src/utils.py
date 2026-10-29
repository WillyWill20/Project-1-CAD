import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
from src.config import *
from skimage.segmentation import clear_border
from skimage.morphology import skeletonize
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def random_sample(df, n=9, seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000)
    print("seed:", seed)
    return df.sample(n, random_state=seed)

def load_image(filename, label, data_dir=BINARY_TRAIN_DIR, class_folders=BINARY_CLASS_FOLDERS):
    path = data_dir / class_folders[label] / filename
    img = cv2.imread(str(path))                  # loads as BGR
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

def load_preprocessed(filename, label, preproc_dir, class_folders=BINARY_CLASS_FOLDERS):
    path = preproc_dir / class_folders[label] / filename.replace(".jpg", ".png")
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Preprocessed image not found: {path}  (run: python -m src.build_dataset)")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def build_ground_truth(data_dir, class_folders=BINARY_CLASS_FOLDERS):
    rows = []
    for label, folder in class_folders.items():
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

def plot_feature_distributions(df, cols, ncols=6, bins=30, class_names=CLASS_NAMES, class_colors=CLASS_COLORS):
    """One small histogram per feature, classes overlaid."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), squeeze=False)

    for ax, col in zip(axes.flat, cols):
        bin_edges = np.histogram_bin_edges(df[col].dropna(), bins=bins)   # same bins for all classes
        for label, name in class_names.items():
            values = df.loc[df["label"] == label, col]
            ax.hist(values, bins=bin_edges, density=True, histtype="step", linewidth=2,
                    color=class_colors[label], label=name)
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

FEATURE_GROUP_PREFIXES = {
    "colour moments":          ("rgb_", "hsv_", "lab_"),
    "colour histograms":       "hist_",
    "LBP":                     "lbp_",
    "GLCM":                    "glcm_",
    "reg. centre":             "reg_centre_",
    "reg. centre - periphery": "reg_cmp_",
    "reg. quadrant std":       "reg_qstd_",
    "reg. asymmetry":          "reg_asym_",
    "HoG":                     "hog_",
}


def feature_groups(columns):
    """Feature columns grouped by family (base, regional, HoG) from their name prefixes."""
    groups = {name: [c for c in columns if c.startswith(prefix)] for name, prefix in FEATURE_GROUP_PREFIXES.items()}
    return {name: cols for name, cols in groups.items() if cols}


def auc_separation(df, cols, pos=1, neg=0):
    """|AUC - 0.5| of every feature for class pos vs class neg (0 = no separation, 0.5 = perfect)."""
    sub = df[df["label"].isin([pos, neg])]
    y = (sub["label"] == pos).astype(int)
    return pd.Series({c: abs(roc_auc_score(y, sub[c]) - 0.5) for c in cols})


def separation_by_group(separation, groups, top=50):
    """Best and mean separation per feature group, and how many of the overall top features it has."""
    top_features = set(separation.nlargest(top).index)
    return pd.DataFrame({name: {"features": len(cols),
                                "best": separation[cols].max(),
                                "mean": separation[cols].mean(),
                                f"in top-{top}": len(top_features & set(cols))}
                         for name, cols in groups.items()}).T


def max_abs_correlation(df, new_cols, ref_cols):
    """For every new feature, its highest |Pearson correlation| with any reference feature (1 = redundant)."""
    def zscore(M):
        M = M - M.mean(axis=0)
        std = M.std(axis=0)
        std[std == 0] = 1                       # constant columns -> correlation 0
        return M / std
    corr = zscore(df[new_cols].values.astype(float)).T @ zscore(df[ref_cols].values.astype(float)) / len(df)
    return pd.Series(np.abs(corr).max(axis=1), index=new_cols)


def pca_3d(df, cols, title, class_names=CLASS_NAMES, class_colors=CLASS_COLORS, n_points=3000, seed=RANDOM_STATE):
    """3-component PCA of the standardised features; interactive 3D scatter of a random sample of images."""
    import plotly.express as px
    pca = PCA(n_components=3, random_state=seed)
    P = pca.fit_transform(StandardScaler().fit_transform(df[cols]))
    points = pd.DataFrame(P, columns=["PC1", "PC2", "PC3"])
    points["class"] = df["label"].map(class_names).values
    points["filename"] = df["filename"].values
    points = points.sample(min(n_points, len(points)), random_state=seed)

    fig = px.scatter_3d(points, x="PC1", y="PC2", z="PC3", color="class", hover_data=["filename"], opacity=0.6,
                        color_discrete_map={class_names[k]: v for k, v in class_colors.items()},
                        title=f"{title} - {len(cols)} features, explained variance {pca.explained_variance_ratio_.sum():.1%}")
    fig.update_traces(marker=dict(size=3))
    fig.show()
    return pca

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

def plot_roc(ax, y_true, scores, label, color, linestyle="-"):
    """Add one ROC curve to ax, with its AUC in the legend."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=2, label=f"{label} (AUC {auc:.3f})")


def finish_roc(ax, title):
    """Chance line, axis labels, legend."""
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")


def plot_confusion(ax, y_true, y_pred, title, class_names=("nevus", "others")):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    ConfusionMatrixDisplay(cm, display_labels=list(class_names)).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title, fontsize=10)