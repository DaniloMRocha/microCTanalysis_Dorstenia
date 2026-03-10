"""
Compare two labeled 3D TIFF segmentations (Inflorescence 1 vs 2)
WITH DIFFERENT LABEL ID SCHEMES between inflorescences.

Requires:
    pip install tifffile numpy pandas matplotlib
"""

import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

# =========================
# USER INPUT
# =========================

TIFF_1 = r"PATH1"
TIFF_2 = r"PATH2"

BACKGROUND_LABEL = 0
DPI_OUT = 300

# Biological categories
CATEGORIES = ["Pistils", "Vascular bundles", "Parenchyma", "Empty alveoli"]

# Inflorescence 1:
# 0 Background
# 1 Pistils
# 2 Vascular bundles
# 3 Parenchyma
# 4 Empty alveoli
MAP_INFLO_1 = {
    0: "Background",
    1: "Pistils",
    2: "Vascular bundles",
    3: "Parenchyma",
    4: "Empty alveoli",
}

# Inflorescence 2:
# 0 Background
# 1 Vascular bundles
# 2 Pistils
# 3 Parenchyma
# 4 Empty alveoli
MAP_INFLO_2 = {
    0: "Background",
    1: "Vascular bundles",
    2: "Pistils",
    3: "Parenchyma",
    4: "Empty alveoli",
}

# =========================
# HELPERS
# =========================
def load_tiff(path: str) -> np.ndarray:
    vol = tiff.imread(path)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D TIFF (Z,Y,X). Got shape {vol.shape} for {path}")
    return vol

def count_voxels_by_label(vol: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(vol, return_counts=True)
    return {int(l): int(c) for l, c in zip(labels, counts)}

def compute_category_volumes(
    counts_by_label: dict[int, int],
    label_to_category: dict[int, str],
    background_label: int = 0
) -> dict:
    """
    Returns:
      - total_voxels
      - background_voxels
      - biological_voxels (total - background)
      - volumes_by_category (only for CATEGORIES)
      - fractions_by_category (relative to biological voxels, 0..1)
    """
    total_voxels = int(sum(counts_by_label.values()))
    bg_voxels = int(counts_by_label.get(background_label, 0))
    bio_voxels = total_voxels - bg_voxels
    if bio_voxels <= 0:
        raise ValueError("bio_voxels <= 0. Check BACKGROUND_LABEL or segmentation.")

    # initialize
    volumes = {cat: 0 for cat in CATEGORIES}

    # map each numeric label count into a category
    for lab, cnt in counts_by_label.items():
        cat = label_to_category.get(lab, None)
        if cat in volumes:
            volumes[cat] += int(cnt)

    fractions = {cat: (volumes[cat] / bio_voxels) for cat in CATEGORIES}

    return {
        "total_voxels": total_voxels,
        "background_voxels": bg_voxels,
        "biological_voxels": bio_voxels,
        "volumes_by_category": volumes,
        "fractions_by_category": fractions,
    }

def plot_grouped_bars(
    x_labels,
    series_a,
    series_b,
    label_a,
    label_b,
    title,
    ylabel,
    out_path,
    dpi=300
):
    x = np.arange(len(x_labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.bar(x - width/2, series_a, width, label=label_a)
    ax.bar(x + width/2, series_b, width, label=label_b)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_title(title, pad=12)
    ax.set_ylabel(ylabel)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)

    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print("Saved figure:", out_path)

# =========================
# RUN
# =========================
vol1 = load_tiff(TIFF_1)
vol2 = load_tiff(TIFF_2)

counts1 = count_voxels_by_label(vol1)
counts2 = count_voxels_by_label(vol2)

m1 = compute_category_volumes(counts1, MAP_INFLO_1, background_label=BACKGROUND_LABEL)
m2 = compute_category_volumes(counts2, MAP_INFLO_2, background_label=BACKGROUND_LABEL)

# -------------------------
# Build tables
# -------------------------
rows_abs = []
rows_frac = []

for cat in CATEGORIES:
    rows_abs.append({
        "category": cat,
        "inflorescence_1_voxels": m1["volumes_by_category"][cat],
        "inflorescence_2_voxels": m2["volumes_by_category"][cat],
        "ratio_inflo2_over_inflo1_voxels": (
            m2["volumes_by_category"][cat] / m1["volumes_by_category"][cat]
            if m1["volumes_by_category"][cat] > 0 else np.nan
        )
    })
    rows_frac.append({
        "category": cat,
        "inflorescence_1_fraction_of_biological": m1["fractions_by_category"][cat],
        "inflorescence_2_fraction_of_biological": m2["fractions_by_category"][cat],
        "inflorescence_1_percent_of_biological": 100.0 * m1["fractions_by_category"][cat],
        "inflorescence_2_percent_of_biological": 100.0 * m2["fractions_by_category"][cat],
        "delta_percent_inflo2_minus_inflo1": 100.0 * (m2["fractions_by_category"][cat] - m1["fractions_by_category"][cat]),
        "ratio_inflo2_over_inflo1_fraction": (
            m2["fractions_by_category"][cat] / m1["fractions_by_category"][cat]
            if m1["fractions_by_category"][cat] > 0 else np.nan
        )
    })

abs_df = pd.DataFrame(rows_abs)
frac_df = pd.DataFrame(rows_frac)

summary_df = pd.DataFrame([{
    "inflorescence_1_shape_ZYX": str(vol1.shape),
    "inflorescence_2_shape_ZYX": str(vol2.shape),
    "inflorescence_1_total_voxels": m1["total_voxels"],
    "inflorescence_2_total_voxels": m2["total_voxels"],
    "inflorescence_1_background_voxels": m1["background_voxels"],
    "inflorescence_2_background_voxels": m2["background_voxels"],
    "inflorescence_1_biological_voxels": m1["biological_voxels"],
    "inflorescence_2_biological_voxels": m2["biological_voxels"],
    "ratio_inflo2_over_inflo1_biological_voxels": m2["biological_voxels"] / m1["biological_voxels"],
}])

# -------------------------
# Save CSVs
# -------------------------
out_dir = os.path.dirname(TIFF_1)

out_summary = os.path.join(out_dir, "compare_inflo_summary_totals.csv")
out_abs = os.path.join(out_dir, "compare_inflo_category_volumes_voxels.csv")
out_frac = os.path.join(out_dir, "compare_inflo_category_fractions_biological.csv")

summary_df.to_csv(out_summary, index=False)
abs_df.to_csv(out_abs, index=False)
frac_df.to_csv(out_frac, index=False)

print("\n=== Saved CSVs ===")
print(out_summary)
print(out_abs)
print(out_frac)

print("\n=== Summary (totals) ===")
print(summary_df.to_string(index=False))

print("\n=== Absolute volumes by category (voxels) ===")
print(abs_df.to_string(index=False))

print("\n=== Fractions by category (relative to biological voxels) ===")
print(frac_df.to_string(index=False))

# -------------------------
# Figures
# -------------------------
cats = CATEGORIES

inf1_abs = [m1["volumes_by_category"][c] for c in cats]
inf2_abs = [m2["volumes_by_category"][c] for c in cats]

fig_abs = os.path.join(out_dir, "BAR_category_volumes_voxels_inflo1_vs_inflo2.png")
plot_grouped_bars(
    x_labels=cats,
    series_a=inf1_abs,
    series_b=inf2_abs,
    label_a="Inflorescence 1",
    label_b="Inflorescence 2",
    title="Absolute tissue volumes by category (voxels)",
    ylabel="Volume (voxels)",
    out_path=fig_abs,
    dpi=DPI_OUT
)

inf1_pct = [100.0 * m1["fractions_by_category"][c] for c in cats]
inf2_pct = [100.0 * m2["fractions_by_category"][c] for c in cats]

fig_frac = os.path.join(out_dir, "BAR_category_fractions_percent_of_biological_inflo1_vs_inflo2.png")
plot_grouped_bars(
    x_labels=cats,
    series_a=inf1_pct,
    series_b=inf2_pct,
    label_a="Inflorescence 1",
    label_b="Inflorescence 2",
    title="Tissue composition by category (percent of biological volume)",
    ylabel="Percent of biological volume (%)",
    out_path=fig_frac,
    dpi=DPI_OUT
)

# Extra: print “which is larger” per category (absolute)
print("\n=== Which inflorescence is larger (absolute voxels) per category ===")
for c in cats:
    v1 = m1["volumes_by_category"][c]
    v2 = m2["volumes_by_category"][c]
    if v1 > v2:
        who = "Inflorescence 1"
    elif v2 > v1:
        who = "Inflorescence 2"
    else:
        who = "Equal"
    print(f"{c}: {who}  (Inflo1={v1:,} | Inflo2={v2:,})")
