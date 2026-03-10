"""

QCs, HEATMAPS AND SPACIAL RELATIONS

Requires:
    pip install tifffile numpy scipy matplotlib pandas
"""

import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

from scipy.ndimage import rotate, label, generate_binary_structure, find_objects
from scipy.stats import chisquare
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as cc_label  

# =========================
# USER PARAMETERS
# =========================

TIFF_PATH = r"INSERT PATH HERE"
OUTPUT_DIR = os.path.dirname(TIFF_PATH)

# Rotation
ROT_ANGLE_X = 65.0

# Labels
LABEL_BACKGROUND = 0
LABEL_VASCULAR = 1
LABEL_PISTIL_OK = 2
LABEL_PISTIL_ABORT = 4  

# Filters
MIN_VOXELS_OK = 200
MIN_VOXELS_ABORT = 50

# 2D Distances
USE_3D_DISTANCES = False

# Biological center
VASC_CORE_TOP_PERCENT = 1.0  

# Smoothing vascular map for biological center
SMOOTH_VASC_MAP = True
GAUSS_SIGMA_VASC = 2.0 

# Heatmaps voxel-based
SAVE_VOXEL_HEATMAPS = True
SMOOTH_VOXEL_HEATMAPS = True
GAUSS_SIGMA_VOXEL = 2.0
DPI_OUT = 300

# QC/plots
DRAW_LINKS_ABORT_TO_OK = True
MAX_LINKS_TO_DRAW = 500

# =========================
# FUNCTIONS
# =========================

def rotate_volume_if_needed(img, angle_x):
    if angle_x is None or float(angle_x) == 0.0:
        return img
    return rotate(
        img,
        angle=float(angle_x),
        axes=(0, 1),   # (Z,Y)
        reshape=True,
        order=0,       # nearest-neighbor 
        mode="constant",
        cval=0
    )

def topview_density_from_label(img, label_value):
    """
    Heatmap voxel-based (top-view): soma em Z os voxels do label.
    Retorna (Y,X) float32.
    """
    return (img == label_value).sum(axis=0).astype(np.float32)

def normalize01(a):
    a = a.astype(np.float32)
    m = float(a.max())
    if m <= 0:
        return a
    return a / m

def biological_center_from_vascular_topview_component(dens2d, top_percent=1.0):
    """
    Biological Center
    """
    if dens2d.max() <= 0:
        raise ValueError("Check LABEL_VASCULAR.")

    vals = dens2d[dens2d > 0]
    if vals.size == 0:
        raise ValueError(" No vascular pixels.")

    thr = np.percentile(vals, 100.0 - float(top_percent))
    mask = dens2d >= thr

    lab, n = cc_label(mask.astype(np.uint8))
    if n == 0:
        y0, x0 = np.unravel_index(int(np.argmax(dens2d)), dens2d.shape)
        return float(x0), float(y0), mask

    best_id = None
    best_mass = -1.0
    for k in range(1, n + 1):
        m = (lab == k)
        mass = float(dens2d[m].sum())
        if mass > best_mass:
            best_mass = mass
            best_id = k

    core = (lab == best_id)
    ys, xs = np.nonzero(core)
    if ys.size == 0:
        y0, x0 = np.unravel_index(int(np.argmax(dens2d)), dens2d.shape)
        return float(x0), float(y0), core

    w = dens2d[ys, xs]
    cy = float(np.sum(ys * w) / np.sum(w))
    cx = float(np.sum(xs * w) / np.sum(w))
    return cx, cy, core

def equivalent_radius_from_mask(mask_core):
    area = float(mask_core.sum())
    if area <= 0:
        return np.nan
    return float(np.sqrt(area / np.pi))

def extract_objects(mask, min_voxels, label_name):
    """Centroids."""
    structure = generate_binary_structure(3, 1)  # 6-connected
    labeled, num = label(mask, structure=structure)
    objs = find_objects(labeled)

    rows = []
    for obj_id, bbox in enumerate(objs, start=1):
        if bbox is None:
            continue

        sub = labeled[bbox]
        local = (sub == obj_id)
        vox = int(local.sum())
        if vox < int(min_voxels):
            continue

        coords_local = np.argwhere(local)
        z0, y0, x0 = bbox[0].start, bbox[1].start, bbox[2].start
        z = coords_local[:, 0].astype(np.float32) + z0
        y = coords_local[:, 1].astype(np.float32) + y0
        x = coords_local[:, 2].astype(np.float32) + x0

        rows.append({
            "label_type": label_name,
            "object_id": int(obj_id),
            "voxel_count": vox,
            "centroid_z": float(z.mean()),
            "centroid_y": float(y.mean()),
            "centroid_x": float(x.mean()),
        })

    return pd.DataFrame(rows)

def pairwise_nearest(src_coords, dst_coords):
    diff = src_coords[:, None, :] - dst_coords[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    idx = np.argmin(dists, axis=1)
    mind = dists[np.arange(dists.shape[0]), idx]
    return idx, mind

def summarize_distances(name, d):
    d = np.asarray(d, dtype=np.float64)
    return {
        "pair": name,
        "n": int(d.size),
        "mean": float(np.mean(d)) if d.size else np.nan,
        "median": float(np.median(d)) if d.size else np.nan,
        "std": float(np.std(d, ddof=1)) if d.size > 1 else np.nan,
        "q05": float(np.quantile(d, 0.05)) if d.size else np.nan,
        "q25": float(np.quantile(d, 0.25)) if d.size else np.nan,
        "q75": float(np.quantile(d, 0.75)) if d.size else np.nan,
        "q95": float(np.quantile(d, 0.95)) if d.size else np.nan,
        "min": float(np.min(d)) if d.size else np.nan,
        "max": float(np.max(d)) if d.size else np.nan,
    }

def compute_quadrants(df, center_y, center_x):
    x = df["centroid_x"].values
    y = df["centroid_y"].values
    q = np.zeros_like(x, dtype=int)
    q[(x >= center_x) & (y <  center_y)] = 1
    q[(x <  center_x) & (y <  center_y)] = 2
    q[(x <  center_x) & (y >= center_y)] = 3
    q[(x >= center_x) & (y >= center_y)] = 4
    return q

def save_voxel_heatmap(hm2d, title, out_path, center_geom, center_bio, core_mask=None, core_radius=None):
    """
    Save voxel-based heatmaps with centers.
    """
    hm_show = normalize01(hm2d)

    plt.figure(figsize=(10, 7))
    plt.imshow(hm_show, cmap="hot")
    plt.colorbar(label="Normalized voxel density (0–1)")
    plt.title(title)
    plt.axis("off")

    # centros
    plt.scatter([center_geom[0]], [center_geom[1]], s=140, marker="x", linewidths=2, label="Geometric center (QC)")
    plt.scatter([center_bio[0]],  [center_bio[1]],  s=140, marker="x", linewidths=2, label="Biological center (USED)")

    # core mask contour (se fornecido)
    if core_mask is not None:
        plt.contour(core_mask.astype(np.uint8), levels=[0.5], linewidths=2)

    # círculo equivalente (opcional)
    if core_radius is not None and np.isfinite(core_radius) and core_radius > 0:
        circ = plt.Circle((center_bio[0], center_bio[1]), core_radius, fill=False,
                          linestyle="--", linewidth=2)
        plt.gca().add_patch(circ)

    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI_OUT)
    plt.close()
    print("Saved:", out_path)

# =========================
# PIPELINE
# =========================

print("Loading TIFF:", TIFF_PATH)
img_raw = tiff.imread(TIFF_PATH)
print("Original shape (Z,Y,X):", img_raw.shape)

print("Rotating volume (if needed)... angle =", ROT_ANGLE_X)
img = rotate_volume_if_needed(img_raw, ROT_ANGLE_X)
print("Working shape (Z,Y,X):", img.shape)

Zmax, Ymax, Xmax = img.shape

# -------------------------
# GEOMETRIC CENTER (QC)
# -------------------------
center_geom_x = Xmax / 2.0
center_geom_y = Ymax / 2.0
print(f"Geometric center (QC only): cx={center_geom_x:.2f}, cy={center_geom_y:.2f}")

# -------------------------
# VASCULAR MAP voxel-based (top-view)
# -------------------------
vasc2d = topview_density_from_label(img, LABEL_VASCULAR)
vasc2d_for_center = gaussian_filter(vasc2d, sigma=float(GAUSS_SIGMA_VASC)) if SMOOTH_VASC_MAP else vasc2d

# -------------------------
# BIOLOGICAL CENTER 1
# -------------------------
center_bio_x, center_bio_y, core_mask = biological_center_from_vascular_topview_component(
    vasc2d_for_center,
    top_percent=VASC_CORE_TOP_PERCENT
)
r_equiv = equivalent_radius_from_mask(core_mask)

print(f"Biological center (USED): cx={center_bio_x:.2f}, cy={center_bio_y:.2f}")
print(f"Core top_percent={VASC_CORE_TOP_PERCENT}%, equiv_radius~{r_equiv:.2f} px")

# -------------------------
# GEOMETRIC AND BIOLOGICAL CENTER
# -------------------------
qc_geom_path = os.path.join(OUTPUT_DIR, "QC_vascular_density_CENTER_GEOMETRIC.png")
save_voxel_heatmap(
    vasc2d_for_center,
    "QC: Vascular density (top-view) + GEOMETRIC center",
    qc_geom_path,
    center_geom=(center_geom_x, center_geom_y),
    center_bio=(center_bio_x, center_bio_y),  
    core_mask=None,
    core_radius=None
)

qc_bio_path = os.path.join(OUTPUT_DIR, "QC_vascular_density_CENTER_BIOLOGICAL_COMPONENT.png")
save_voxel_heatmap(
    vasc2d_for_center,
    "QC: Vascular density (top-view) + BIOLOGICAL center + core contour (component)",
    qc_bio_path,
    center_geom=(center_geom_x, center_geom_y),
    center_bio=(center_bio_x, center_bio_y),
    core_mask=core_mask,
    core_radius=None
)

# -------------------------
# ALVEOLI OK and ABORT (or empty)
# -------------------------
mask_ok = (img == LABEL_PISTIL_OK)
mask_ab = (img == LABEL_PISTIL_ABORT)

df_ok = extract_objects(mask_ok, MIN_VOXELS_OK, "Pistil_OK")
df_ab = extract_objects(mask_ab, MIN_VOXELS_ABORT, "Pistil_ABORT")

print("Objects kept OK:", df_ok.shape[0])
print("Objects kept ABORT:", df_ab.shape[0])

df_all = pd.concat([df_ok, df_ab], ignore_index=True)
centroids_csv = os.path.join(OUTPUT_DIR, "pistil_centroids_ok_abort.csv")
df_all.to_csv(centroids_csv, index=False)
print("Saved:", centroids_csv)

# -------------------------
# HEATMAPS voxel-based ALVEOLI OK and ABORT (or empty)
# -------------------------
if SAVE_VOXEL_HEATMAPS:
    ok2d = topview_density_from_label(img, LABEL_PISTIL_OK)
    ab2d = topview_density_from_label(img, LABEL_PISTIL_ABORT)

    if SMOOTH_VOXEL_HEATMAPS and float(GAUSS_SIGMA_VOXEL) > 0:
        ok2d_s = gaussian_filter(ok2d, sigma=float(GAUSS_SIGMA_VOXEL))
        ab2d_s = gaussian_filter(ab2d, sigma=float(GAUSS_SIGMA_VOXEL))
        suffix = f"_smoothed_sigma{float(GAUSS_SIGMA_VOXEL):.1f}".replace(".", "p")
    else:
        ok2d_s = ok2d
        ab2d_s = ab2d
        suffix = "_nosmooth"

    heat_ok_path = os.path.join(OUTPUT_DIR, f"HEATMAP_VOXEL_OK_topview{suffix}.png")
    heat_ab_path = os.path.join(OUTPUT_DIR, f"HEATMAP_VOXEL_ABORT_topview{suffix}.png")

    save_voxel_heatmap(
        ok2d_s,
        f"Heatmap (voxel-based, top-view): Pistils OK{'' if suffix=='_nosmooth' else ' (smoothed)'}",
        heat_ok_path,
        center_geom=(center_geom_x, center_geom_y),
        center_bio=(center_bio_x, center_bio_y),
        core_mask=None,
        core_radius=r_equiv
    )

    save_voxel_heatmap(
        ab2d_s,
        f"Heatmap (voxel-based, top-view): Pistils ABORT / empties{'' if suffix=='_nosmooth' else ' (smoothed)'}",
        heat_ab_path,
        center_geom=(center_geom_x, center_geom_y),
        center_bio=(center_bio_x, center_bio_y),
        core_mask=None,
        core_radius=r_equiv
    )

# -------------------------
# Nearest-neighbor distances
# -------------------------
summary_rows = []
if df_ok.shape[0] > 0 and df_ab.shape[0] > 0:
    if USE_3D_DISTANCES:
        ok_coords = df_ok[["centroid_z", "centroid_y", "centroid_x"]].values
        ab_coords = df_ab[["centroid_z", "centroid_y", "centroid_x"]].values
    else:
        ok_coords = df_ok[["centroid_y", "centroid_x"]].values
        ab_coords = df_ab[["centroid_y", "centroid_x"]].values

    idx_ab_to_ok, d_ab_to_ok = pairwise_nearest(ab_coords, ok_coords)
    idx_ok_to_ab, d_ok_to_ab = pairwise_nearest(ok_coords, ab_coords)

    # intra ABORT
    if df_ab.shape[0] > 1:
        diff = ab_coords[:, None, :] - ab_coords[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        idx_ab_to_ab = np.argmin(dist, axis=1)
        d_ab_to_ab = dist[np.arange(dist.shape[0]), idx_ab_to_ab]
    else:
        idx_ab_to_ab = np.array([0], dtype=int)
        d_ab_to_ab = np.array([np.nan], dtype=float)

    # intra OK
    if df_ok.shape[0] > 1:
        diff = ok_coords[:, None, :] - ok_coords[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        idx_ok_to_ok = np.argmin(dist, axis=1)
        d_ok_to_ok = dist[np.arange(dist.shape[0]), idx_ok_to_ok]
    else:
        idx_ok_to_ok = np.array([0], dtype=int)
        d_ok_to_ok = np.array([np.nan], dtype=float)

    # CSVs
    df_ab_out = df_ab.copy()
    df_ab_out["nearest_ok_object_id"] = df_ok.iloc[idx_ab_to_ok]["object_id"].values
    df_ab_out["dist_to_nearest_ok"] = d_ab_to_ok
    df_ab_out["nearest_abort_object_id"] = df_ab.iloc[idx_ab_to_ab]["object_id"].values if df_ab.shape[0] > 1 else np.nan
    df_ab_out["dist_to_nearest_abort"] = d_ab_to_ab

    df_ok_out = df_ok.copy()
    df_ok_out["nearest_abort_object_id"] = df_ab.iloc[idx_ok_to_ab]["object_id"].values
    df_ok_out["dist_to_nearest_abort"] = d_ok_to_ab
    df_ok_out["nearest_ok_object_id"] = df_ok.iloc[idx_ok_to_ok]["object_id"].values if df_ok.shape[0] > 1 else np.nan
    df_ok_out["dist_to_nearest_ok"] = d_ok_to_ok

    dist_csv_ab = os.path.join(OUTPUT_DIR, "nearest_distances_ABORT.csv")
    dist_csv_ok = os.path.join(OUTPUT_DIR, "nearest_distances_OK.csv")
    df_ab_out.to_csv(dist_csv_ab, index=False)
    df_ok_out.to_csv(dist_csv_ok, index=False)
    print("Saved:", dist_csv_ab)
    print("Saved:", dist_csv_ok)

    summary_rows.append(summarize_distances("ABORT -> nearest OK", d_ab_to_ok))
    summary_rows.append(summarize_distances("OK -> nearest ABORT", d_ok_to_ab))
    summary_rows.append(summarize_distances("ABORT -> nearest ABORT", d_ab_to_ab))
    summary_rows.append(summarize_distances("OK -> nearest OK", d_ok_to_ok))
else:
    print("Skipping distance analyses: need at least 1 OK and 1 ABORT object.")

summary_csv = os.path.join(OUTPUT_DIR, "distance_summary_metrics.csv")
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
print("Saved:", summary_csv)

# -------------------------
# QUADRANT TESTS
# -------------------------
quad_test_csv = os.path.join(OUTPUT_DIR, "quadrant_test_abort_chi_square.csv")
abort_quad_csv = os.path.join(OUTPUT_DIR, "abort_quadrants.csv")

if df_ab.shape[0] >= 5:
    q_ab = compute_quadrants(df_ab, center_y=center_bio_y, center_x=center_bio_x)

    df_ab_q = df_ab.copy()
    df_ab_q["quadrant"] = q_ab
    df_ab_q.to_csv(abort_quad_csv, index=False)
    print("Saved:", abort_quad_csv)

    counts = np.array([(q_ab == 1).sum(), (q_ab == 2).sum(), (q_ab == 3).sum(), (q_ab == 4).sum()], dtype=float)
    expected = np.ones(4) * (counts.sum() / 4.0)
    chi_stat, chi_p = chisquare(f_obs=counts, f_exp=expected)

    warn = ""
    if np.any(expected < 5):
        warn = "ATENCAO: valores esperados < 5; para n pequeno, prefira teste exato/permutacao."

    quad_df = pd.DataFrame([{
        "n_abort": int(counts.sum()),
        "q1": int(counts[0]),
        "q2": int(counts[1]),
        "q3": int(counts[2]),
        "q4": int(counts[3]),
        "chi2_stat": float(chi_stat),
        "chi2_pvalue": float(chi_p),
        "biological_center_x": float(center_bio_x),
        "biological_center_y": float(center_bio_y),
        "core_top_percent": float(VASC_CORE_TOP_PERCENT),
        "core_equiv_radius_px": float(r_equiv),
        "warning": warn
    }])
    quad_df.to_csv(quad_test_csv, index=False)
    print("Saved:", quad_test_csv)
else:
    print("Quadrant test skipped: recommended n_abort >= 5.")

# -------------------------
# CENTROID MAP - GEOMETRIC CENTER
# -------------------------
map_geom_path = os.path.join(OUTPUT_DIR, "map_centroids_CENTER_GEOMETRIC_QC.png")
plt.figure(figsize=(9, 9))
ax = plt.gca()

if df_ok.shape[0] > 0:
    ax.scatter(df_ok["centroid_x"], df_ok["centroid_y"], s=40, label="Pistil OK")
if df_ab.shape[0] > 0:
    ax.scatter(df_ab["centroid_x"], df_ab["centroid_y"], s=60, marker="D", label="Pistil ABORT")

ax.scatter([center_geom_x], [center_geom_y], s=160, marker="x", linewidths=2, label="Geometric center (QC)")
ax.axvline(center_geom_x, linewidth=1)
ax.axhline(center_geom_y, linewidth=1)

ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
ax.set_title("Spatial map (QC): GEOMETRIC center")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(map_geom_path, dpi=DPI_OUT)
plt.close()
print("Saved:", map_geom_path)

# -------------------------
# CENTROID MAP - BIOLOGICAL
# -------------------------
map_bio_path = os.path.join(OUTPUT_DIR, "map_centroids_CENTER_BIOLOGICAL_USED.png")
plt.figure(figsize=(9, 9))
ax = plt.gca()

if df_ok.shape[0] > 0:
    ax.scatter(df_ok["centroid_x"], df_ok["centroid_y"], s=40, label="Pistil OK")
if df_ab.shape[0] > 0:
    ax.scatter(df_ab["centroid_x"], df_ab["centroid_y"], s=60, marker="D", label="Pistil ABORT")

ax.scatter([center_bio_x], [center_bio_y], s=160, marker="x", linewidths=2, label="Biological center (USED)")
ax.axvline(center_bio_x, linewidth=1)
ax.axhline(center_bio_y, linewidth=1)

if np.isfinite(r_equiv) and r_equiv > 0:
    circ = plt.Circle((center_bio_x, center_bio_y), r_equiv, fill=False, linestyle="--", linewidth=2, edgecolor="black")
    ax.add_patch(circ)

# ABORT -> OK
if DRAW_LINKS_ABORT_TO_OK and df_ok.shape[0] > 0 and df_ab.shape[0] > 0:
    if USE_3D_DISTANCES:
        ok_coords_draw = df_ok[["centroid_z", "centroid_y", "centroid_x"]].values
        ab_coords_draw = df_ab[["centroid_z", "centroid_y", "centroid_x"]].values
    else:
        ok_coords_draw = df_ok[["centroid_y", "centroid_x"]].values
        ab_coords_draw = df_ab[["centroid_y", "centroid_x"]].values

    idx_ab_to_ok_draw, _ = pairwise_nearest(ab_coords_draw, ok_coords_draw)
    n_links = min(df_ab.shape[0], int(MAX_LINKS_TO_DRAW))
    for i in range(n_links):
        abx = df_ab.iloc[i]["centroid_x"]
        aby = df_ab.iloc[i]["centroid_y"]
        j = int(idx_ab_to_ok_draw[i])
        okx = df_ok.iloc[j]["centroid_x"]
        oky = df_ok.iloc[j]["centroid_y"]
        ax.plot([abx, okx], [aby, oky], linewidth=0.5, alpha=0.5)

ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
ax.set_title("Spatial map (USED): BIOLOGICAL center (vascular core component)")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(map_bio_path, dpi=DPI_OUT)
plt.close()
print("Saved:", map_bio_path)

print("Done.")
