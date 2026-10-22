"""
Data augmentation of preprocessed images (Challenge 2: scc only).

Each augmented copy combines, with random parameters:
  1. random crop (85-100% of the area), resized back to the original size
  2. rotation (any angle in +-90 deg) and scaling (0.9-1.1), empty borders filled by mirroring
  3. brightness, contrast and saturation changes (x 0.8-1.2)
No flips or exact 90 deg rotations (identical features with our invariant descriptors),
no hue shift (colour casts are removed on purpose), no blur and no noise (images are already soft).
"""
import numpy as np
import cv2

AUG_PARAMS = dict(
    crop_area=(0.85, 1.0),
    angle=(-90, 90),
    scale=(0.9, 1.1),
    brightness=(0.8, 1.2),
    contrast=(0.8, 1.2),
    saturation=(0.8, 1.2),
)


def random_crop(img, rng, area_range):
    """Square crop with a random fraction of the area at a random position, resized back."""
    h, w = img.shape[:2]
    side = int(round(np.sqrt(rng.uniform(*area_range)) * min(h, w)))
    top = rng.integers(0, h - side + 1)
    left = rng.integers(0, w - side + 1)
    crop = img[top:top + side, left:left + side]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def rotate_and_scale(img, angle, scale):
    """Rotation around the centre + scaling; borders filled by mirroring the image."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def change_colour(img, brightness, contrast, saturation):
    """Brightness (multiply), contrast (stretch around the mean), saturation (HSV S channel)."""
    out = img.astype(np.float32) * brightness
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * contrast + mean
    out = np.clip(out, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment_image(img, rng, params=AUG_PARAMS):
    """One augmented copy of a preprocessed RGB image (crop -> rotate/scale -> colour)."""
    out = random_crop(img, rng, params["crop_area"])
    out = rotate_and_scale(out, rng.uniform(*params["angle"]), rng.uniform(*params["scale"]))
    out = change_colour(out, rng.uniform(*params["brightness"]),
                        rng.uniform(*params["contrast"]), rng.uniform(*params["saturation"]))
    return out