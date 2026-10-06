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

def get_valid_mask(img, threshold=40):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark = gray < threshold
    dark_inside = clear_border(dark) # We use this function from SciKitImage to obtain the black borders
    frame = dark.copy()
    frame[dark_inside] = False
    valid = ~frame # Valid pixels = everything except the frame
    return valid

def has_vignette(valid): # Check whether the image has a round dermatoscope vignette.
    top_left     = valid[0, 0]
    top_right    = valid[0, -1]
    bottom_left  = valid[-1, 0]
    bottom_right = valid[-1, -1]

    corners_are_valid = [top_left, top_right, bottom_left, bottom_right]

    return not any(corners_are_valid)

def crop_inside_halo(img, valid):
    ys, xs = np.where(valid)
    cy, cx = int(ys.mean()), int(xs.mean())       # circle centre = centre of valid pixels
    radius = np.sqrt(valid.sum() / np.pi)         # from circle area = π·r²
    half = int(radius / np.sqrt(2))               # half side of the inner square

    top, bottom = max(cy - half, 0), min(cy + half, img.shape[0])
    left, right = max(cx - half, 0), min(cx + half, img.shape[1])
    return img[top:bottom, left:right]

def resize_and_center_crop(img, size): #Resize so the shorter side equals `size`
    height, width = img.shape[:2]
    scale = size / min(height, width)
    new_width = round(width * scale)
    new_height = round(height * scale)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Cut a SizexSize square from the centre
    top = (new_height - size) // 2
    left = (new_width - size) // 2
    return img[top:top + size, left:left + size]

def detect_hair_candidates(img, kernel_size, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, candidates = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    return candidates

def keep_hair_components(mask, min_length, max_width):
    n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[:, cv2.CC_STAT_AREA]

    # Skeleton length of each blob: count skeleton pixels per blob label
    skeleton = skeletonize(mask > 0)
    lengths = np.bincount(labels[skeleton], minlength=n_blobs)

    avg_widths = areas / np.maximum(lengths, 1)   # avoid division by 0

    keep = (lengths >= min_length) & (avg_widths <= max_width)
    keep[0] = False                               # never keep the background
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def remove_hair(img, kernel_size=17, threshold=15, min_length=40, max_width=10):
    candidates = detect_hair_candidates(img, kernel_size, threshold)
    hair_mask = keep_hair_components(candidates, min_length, max_width)
    clean = cv2.inpaint(img, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return clean, hair_mask

def shades_of_gray(img, p=6):
    img_float = img.astype(np.float64)

    # 1. Estimate the light colour: p-norm of each channel (R, G, B)
    illuminant = np.power(np.mean(np.power(img_float, p), axis=(0, 1)), 1 / p)

    # 2. Normalise the light colour to length 1
    illuminant = illuminant / np.linalg.norm(illuminant)

    # 3. Correct each channel (a neutral light, (1,1,1)/sqrt(3), leaves the image unchanged)
    corrected = img_float / (illuminant * np.sqrt(3))

    return np.clip(corrected, 0, 255).astype(np.uint8)

def preprocess_image(img, resizing_dim, kernel_size, threshold, min_length, max_width):
    # Step 1: remove vignette
    valid = get_valid_mask(img)
    if has_vignette(valid):
        img = crop_inside_halo(img, valid)

    # Step 2: common size
    img = resize_and_center_crop(img, resizing_dim)

    # Step 3: hair removal
    img, _ = remove_hair(img, kernel_size, threshold, min_length, max_width)

    # Step 4: colour constancy
    img = shades_of_gray(img)
    return img

def preprocess_dataset(gt, data_dir, out_dir, **params):
    """Preprocess every image in gt and save it to out_dir/<class>/<name>.png"""
    for i, r in enumerate(gt.itertuples()):
        out_path = out_dir / BINARY_CLASS_FOLDERS[r.label] / r.filename.replace(".jpg", ".png")
        if out_path.exists():            # already done -> skip (lets you resume)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = load_image(r.filename, r.label, data_dir)
        img = preprocess_image(img, **params)
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # OpenCV saves BGR

        if (i + 1) % 1000 == 0:
            print(f"{i + 1} / {len(gt)}")