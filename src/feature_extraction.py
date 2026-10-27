import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from src.config import *
from joblib import Parallel, delayed
from src.utils import load_preprocessed
import pandas as pd
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog


def to_color_spaces(img):
    """The same image in RGB, HSV and L*a*b* (OpenCV, uint8)."""
    return {
        "rgb": img,
        "hsv": cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
        "lab": cv2.cvtColor(img, cv2.COLOR_RGB2LAB),
    }


def color_moments(img):
    """Mean, std, skewness and kurtosis of each channel in RGB, HSV and Lab -> 36 features_1."""
    features = {}
    for space, converted in to_color_spaces(img).items():
        for c, channel in enumerate(space):            # 'rgb' -> 'r', 'g', 'b'
            values = converted[:, :, c].ravel().astype(np.float64)
            features[f"{space}_{channel}_mean"] = values.mean()
            features[f"{space}_{channel}_std"] = values.std()
            features[f"{space}_{channel}_skew"] = skew(values)
            features[f"{space}_{channel}_kurt"] = kurtosis(values)
    return features


def color_histograms(img, n_bins=16, ranges=HIST_RANGES):
    """Normalised 16-bin histogram of each channel in HSV and Lab -> 96 features_1."""
    features = {}
    spaces = to_color_spaces(img)
    for space in ["hsv", "lab"]:
        for c, channel in enumerate(space):
            low, high = ranges[f"{space}_{channel}"]
            values = np.clip(spaces[space][:, :, c].ravel(), low, high)   # outliers go into the end bins
            hist, _ = np.histogram(values, bins=n_bins, range=(low, high))
            hist = hist / hist.sum()
            for i, v in enumerate(hist):
                features[f"hist_{space}_{channel}_{i}"] = v
    return features


def lbp_features(gray, configs=((8, 1), (16, 2), (24, 3), (24, 5))):
    """Rotation-invariant uniform LBP histograms for several (P, R) -> 10 + 18 + 26 + 26 = 80 features_1."""
    features = {}
    for P, R in configs:
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        hist = hist / hist.sum()
        for i, v in enumerate(hist):
            features[f"lbp_P{P}_R{R}_{i}"] = v
    return features


def glcm_features(gray, distances=(1, 3, 5), levels=64):
    """Haralick features_1 for 3 distances, averaged over 4 angles -> 18 features_1."""
    gray_q = (gray // (256 // levels)).astype(np.uint8)    # 256 -> 64 grey levels
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_q, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)

    features = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        values = graycoprops(glcm, prop).mean(axis=1)      # shape (distances, angles) -> average angles
        for d, v in zip(distances, values):
            features[f"glcm_{prop}_d{d}"] = v

    # Entropy is computed by hand: -sum(p * log2(p))
    entropy = -np.sum(glcm * np.log2(glcm + 1e-12), axis=(0, 1)).mean(axis=1)
    for d, v in zip(distances, entropy):
        features[f"glcm_entropy_d{d}"] = v
    return features


def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = {}
    features.update(color_moments(img))
    features.update(color_histograms(img))
    features.update(lbp_features(gray))
    features.update(glcm_features(gray))
    return features

# ---------------------------------------------------------------------------
# G1 - Regional features
# ---------------------------------------------------------------------------
CENTRE_FRAC = 0.6                     # side of the centre square / image side (median lesion size ~0.64-0.76)
QUADRANTS = ["TL", "TR", "BL", "BR"]
REGIONAL_LBP = ((8, 1), (16, 2))
REGIONAL_GLCM_DISTANCES = (1, 3)
REGIONAL_GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation"]


def region_masks(shape, centre_frac=CENTRE_FRAC):
    """Centre square (side centre_frac * size) + the 4 quadrants with the centre cut out (no overlap)."""
    h, w = shape[:2]
    ch, cw = int(h * centre_frac), int(w * centre_frac)
    top, left = (h - ch) // 2, (w - cw) // 2
    centre = np.zeros((h, w), dtype=bool)
    centre[top:top + ch, left:left + cw] = True

    rows, cols = np.mgrid[:h, :w]
    upper, left_half = rows < h // 2, cols < w // 2
    return {
        "centre": centre,
        "TL": upper & left_half & ~centre,
        "TR": upper & ~left_half & ~centre,
        "BL": ~upper & left_half & ~centre,
        "BR": ~upper & ~left_half & ~centre,
    }


def half_masks(shape):
    """Top / bottom / left / right halves of the whole image (for asymmetry)."""
    h, w = shape[:2]
    rows, cols = np.mgrid[:h, :w]
    return {"top": rows < h // 2, "bottom": rows >= h // 2,
            "left": cols < w // 2, "right": cols >= w // 2}


def masked_glcm(gray_q, mask, distances=REGIONAL_GLCM_DISTANCES):
    """
    GLCM of the pixels inside mask (any shape). gray_q has grey levels 1..63;
    pixels outside the mask get level 0, and every pair involving level 0 is dropped.
    """
    rows, cols = np.where(mask)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1   # bounding box (speed)
    region = np.where(mask, gray_q, 0)[r0:r1, c0:c1].astype(np.uint8)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(region, distances=distances, angles=angles, levels=64, symmetric=True)
    glcm = glcm[1:, 1:].astype(np.float64)                       # drop pairs with the outside level
    return glcm / glcm.sum(axis=(0, 1), keepdims=True)


def region_descriptor(lab, lbp_maps, gray_q, mask, n_bins=8, ranges=HIST_RANGES):
    """68 values of one region: Lab mean/std (6), Lab 8-bin histograms (24), LBP (28), GLCM (10)."""
    d = {}
    for c, ch in enumerate("lab"):
        values = lab[:, :, c][mask].astype(np.float64)
        d[f"lab_{ch}_mean"] = values.mean()
        d[f"lab_{ch}_std"] = values.std()
    for c, ch in enumerate("lab"):
        low, high = ranges[f"lab_{ch}"]
        values = np.clip(lab[:, :, c][mask], low, high)
        hist, _ = np.histogram(values, bins=n_bins, range=(low, high))
        for i, v in enumerate(hist / hist.sum()):
            d[f"hist_lab_{ch}_{i}"] = v
    for (P, R), lbp in lbp_maps.items():
        hist = np.bincount(lbp[mask].astype(int), minlength=P + 2)
        for i, v in enumerate(hist / hist.sum()):
            d[f"lbp_P{P}_R{R}_{i}"] = v
    glcm = masked_glcm(gray_q, mask)
    for prop in REGIONAL_GLCM_PROPS:
        for dist, v in zip(REGIONAL_GLCM_DISTANCES, graycoprops(glcm, prop).mean(axis=1)):
            d[f"glcm_{prop}_d{dist}"] = v
    entropy = -np.sum(glcm * np.log2(glcm + 1e-12), axis=(0, 1)).mean(axis=1)
    for dist, v in zip(REGIONAL_GLCM_DISTANCES, entropy):
        d[f"glcm_entropy_d{dist}"] = v
    return d


def chi2_distance(p, q):
    return 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-12))


def relative_distance(p, q):
    return np.mean(np.abs(p - q) / (np.abs(p) + np.abs(q) + 1e-12))


# One distance per descriptor block, used for asymmetry between opposite halves
ASYMMETRY_BLOCKS = {
    "lab": lambda p, q: np.linalg.norm(p - q),   # Lab mean/std, Euclidean (colour difference)
    "hist": chi2_distance,                       # colour histograms
    "lbp": chi2_distance,                        # texture histograms
    "glcm": relative_distance,                   # GLCM properties (different scales)
}


def regional_features(img):
    """G1: 212 order-free regional features (centre, centre - periphery, quadrant std, asymmetry)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lbp_maps = {(P, R): local_binary_pattern(gray, P, R, method="uniform") for P, R in REGIONAL_LBP}
    gray_q = 1 + (gray.astype(np.int32) * 62) // 255                # grey levels 1..63 (0 = outside)

    def describe(mask):
        return region_descriptor(lab, lbp_maps, gray_q, mask)

    regions = {name: describe(mask) for name, mask in region_masks(gray.shape).items()}
    keys = list(regions["centre"])
    centre = np.array([regions["centre"][k] for k in keys])
    quads = np.array([[regions[q][k] for k in keys] for q in QUADRANTS])   # 4 x 68

    features = {}
    features.update({f"reg_centre_{k}": v for k, v in zip(keys, centre)})
    features.update({f"reg_cmp_{k}": v for k, v in zip(keys, centre - quads.mean(axis=0))})
    features.update({f"reg_qstd_{k}": v for k, v in zip(keys, quads.std(axis=0))})

    halves = {name: describe(mask) for name, mask in half_masks(gray.shape).items()}
    for block, distance in ASYMMETRY_BLOCKS.items():
        block_keys = [k for k in keys if k.startswith(block + "_")]
        vec = {h: np.array([halves[h][k] for k in block_keys]) for h in halves}
        dists = [distance(vec["top"], vec["bottom"]), distance(vec["left"], vec["right"])]
        features[f"reg_asym_{block}_max"] = max(dists)
        features[f"reg_asym_{block}_min"] = min(dists)
    return features


# ---------------------------------------------------------------------------
# HoG
# ---------------------------------------------------------------------------
def hog_features(gray, size=128, cell=32):
    """Standard HoG (9 orientations, 2x2 cells per block, L2-Hys) of the image resized to size x size -> 324."""
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    values = hog(small, orientations=9, pixels_per_cell=(cell, cell),
                 cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    return {f"hog_{i}": v for i, v in enumerate(values)}


def extract_extra_features(img):
    """G1 regional (212, prefix reg_) + HoG (324, prefix hog_) -> 536 features."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = {}
    features.update(regional_features(img))
    features.update(hog_features(gray))
    return features


def draw_regions(img, centre_frac=CENTRE_FRAC, color=(255, 255, 0), thickness=2):
    """Copy of img with the 5 regions drawn: centre square + the quadrant lines outside it."""
    out = img.copy()
    h, w = img.shape[:2]
    ch, cw = int(h * centre_frac), int(w * centre_frac)
    top, left = (h - ch) // 2, (w - cw) // 2
    cv2.rectangle(out, (left, top), (left + cw - 1, top + ch - 1), color, thickness)
    cv2.line(out, (w // 2, 0), (w // 2, top), color, thickness)
    cv2.line(out, (w // 2, top + ch), (w // 2, h - 1), color, thickness)
    cv2.line(out, (0, h // 2), (left, h // 2), color, thickness)
    cv2.line(out, (left + cw, h // 2), (w - 1, h // 2), color, thickness)
    return out

def features_of_one(filename, label, preproc_dir, class_folders, extractor=extract_features):
    img = load_preprocessed(filename, label, preproc_dir, class_folders)
    return {"filename": filename, "label": label, **extractor(img)}


def extract_dataset_features(gt, preproc_dir, class_folders=BINARY_CLASS_FOLDERS, n_jobs=-2,
                             extractor=extract_features):
    """Features of every image in gt (in the same order as gt), one row per image."""
    rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(features_of_one)(r.filename, r.label, preproc_dir, class_folders, extractor)
        for r in gt.itertuples()
    )
    return pd.DataFrame(rows)