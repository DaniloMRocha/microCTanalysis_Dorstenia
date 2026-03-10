"""
Side-by-side top-view comparison of two inflorescences, same scale and crop
"""

import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


# =========================
# USER INPUT
# =========================

TIFF_1 = r"INSERT PATH 1"
TIFF_2 = r"INSERT PATH 2"

BACKGROUND_LABEL = 0

# Optional physical pixel size (XY)
VOXEL_SIZE_XY_1 = None  # e.g. 5.0 (µm/px)
VOXEL_SIZE_XY_2 = None  # e.g. 5.0 (µm/px)

USE_COMMON_PHYSICAL_SCALE = True

DPI_OUT = 300


# =========================
# HELPERS
# =========================
def load_tiff(path):
    vol = tiff.imread(path)
    if vol.ndim != 3:
        raise ValueError("TIFF must be 3D (Z,Y,X)")
    return vol

def topview_mask(vol):
    return (vol != BACKGROUND_LABEL).any(axis=0)

def bbox_2d(mask):
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise ValueError("Empty mask")
    y0, x0 = idx.min(axis=0)
    y1, x1 = idx.max(axis=0) + 1
    return y0, y1, x0, x1

def crop(mask):
    y0, y1, x0, x1 = bbox_2d(mask)
    return mask[y0:y1, x0:x1]

def resample(mask, current_px, target_px):
    if current_px is None or target_px is None or np.isclose(current_px, target_px):
        return mask
    scale = current_px / target_px
    return zoom(mask.astype(np.uint8), (scale, scale), order=0).astype(bool)

def pad_to_same_size(a, b):
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])

    def pad(m):
        out = np.zeros((H, W), dtype=bool)
        y0 = (H - m.shape[0]) // 2
        x0 = (W - m.shape[1]) // 2
        out[y0:y0+m.shape[0], x0:x0+m.shape[1]] = m
        return out

    return pad(a), pad(b)


# =========================
# RUN
# =========================
vol1 = load_tiff(TIFF_1)
vol2 = load_tiff(TIFF_2)

m1 = crop(topview_mask(vol1))
m2 = crop(topview_mask(vol2))

# Normalize physical scale if provided
if USE_COMMON_PHYSICAL_SCALE and VOXEL_SIZE_XY_1 and VOXEL_SIZE_XY_2:
    target = max(VOXEL_SIZE_XY_1, VOXEL_SIZE_XY_2)
    m1 = resample(m1, VOXEL_SIZE_XY_1, target)
    m2 = resample(m2, VOXEL_SIZE_XY_2, target)

m1, m2 = pad_to_same_size(m1, m2)

# =========================
# PLOT
# =========================
fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")

titles = ["Inflorescence 1", "Inflorescence 2"]

for ax, mask, title in zip(axes, [m1, m2], titles):
    ax.imshow(
        mask.astype(np.uint8),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )
    ax.contour(mask.astype(np.uint8), levels=[0.5], colors="black", linewidths=1.5)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    ax.set_facecolor("white")

plt.tight_layout()

out_dir = os.path.dirname(TIFF_1)
out_path = os.path.join(out_dir, "INFLO_TOPVIEW_SIDE_BY_SIDE_GRAY_WHITE.png")
plt.savefig(out_path, dpi=DPI_OUT, bbox_inches="tight", facecolor="white")
plt.close()

print("Saved:", out_path)
