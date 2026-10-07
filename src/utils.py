import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
from src.config import BINARY_TRAIN_DIR, BINARY_CLASS_FOLDERS
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