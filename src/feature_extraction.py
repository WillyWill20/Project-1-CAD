import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def to_color_spaces(img):
    """The same image in RGB, HSV and L*a*b* (OpenCV, uint8)."""
    return {
        "rgb": img,
        "hsv": cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
        "lab": cv2.cvtColor(img, cv2.COLOR_RGB2LAB),
    }


def color_moments(img):
    """Mean, std, skewness and kurtosis of each channel in RGB, HSV and Lab -> 36 features."""
    features = {}
    for space, converted in to_color_spaces(img).items():
        for c, channel in enumerate(space):            # 'rgb' -> 'r', 'g', 'b'
            values = converted[:, :, c].ravel().astype(np.float64)
            features[f"{space}_{channel}_mean"] = values.mean()
            features[f"{space}_{channel}_std"] = values.std()
            features[f"{space}_{channel}_skew"] = skew(values)
            features[f"{space}_{channel}_kurt"] = kurtosis(values)
    return features


def color_histograms(img, n_bins=16):
    """Normalised 16-bin histogram of each channel in HSV and Lab -> 96 features."""
    features = {}
    spaces = to_color_spaces(img)
    for space in ["hsv", "lab"]:
        for c, channel in enumerate(space):
            values = spaces[space][:, :, c].ravel()
            max_value = 180 if (space == "hsv" and channel == "h") else 256   # OpenCV hue: 0-179
            hist, _ = np.histogram(values, bins=n_bins, range=(0, max_value))
            hist = hist / hist.sum()
            for i, v in enumerate(hist):
                features[f"hist_{space}_{channel}_{i}"] = v
    return features


def lbp_features(gray):
    """Rotation-invariant uniform LBP histograms for (P=8, R=1) and (P=16, R=2) -> 28 features."""
    features = {}
    for P, R in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        n_bins = P + 2                                 # P+1 uniform patterns + 1 "non-uniform" bin
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        hist = hist / hist.sum()
        for i, v in enumerate(hist):
            features[f"lbp_P{P}_R{R}_{i}"] = v
    return features


def glcm_features(gray, distances=(1, 3, 5), levels=64):
    """Haralick features for 3 distances, averaged over 4 angles -> 18 features."""
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
    """All Priority 1 features of one preprocessed RGB image -> 178 features."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    features = {}
    features.update(color_moments(img))
    features.update(color_histograms(img))
    features.update(lbp_features(gray))
    features.update(glcm_features(gray))
    return features