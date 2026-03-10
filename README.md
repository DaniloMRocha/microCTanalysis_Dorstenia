# MicroCT Inflorescence Spatial and Structural Analysis Toolkit

This repository provides three Python scripts developed to support the quantitative analysis and visualization of segmented 3D microCT datasets of plant inflorescences. The scripts were created as part of the analytical workflow used in the associated publication and are provided here as supplementary material to ensure full reproducibility of the spatial and volumetric analyses.

The tools operate on labeled 3D TIFF segmentation volumes (Z,Y,X) and enable extraction of biologically meaningful spatial metrics, volumetric comparisons, and standardized visualization of inflorescence structure.

Together, the scripts allow researchers to analyze spatial organization, tissue composition, and morphological differences between inflorescences derived from microCT segmentation datasets.

---

# Scripts

## 1. SpatialInflo.py

This script performs spatial analysis of segmented inflorescences, focusing on the spatial distribution of reproductive structures relative to the vascular system.

Main analyses include:

- Detection of individual structures (e.g., pistils) from labeled segmentation volumes  
- Extraction of object centroids and voxel counts  
- Identification of a biological center based on vascular bundle density  
- Generation of voxel-density heatmaps (top-view projections)  
- Spatial maps of reproductive structures  
- Nearest-neighbor spatial distance analyses between structures  
- Quadrant-based spatial distribution tests (Chi-square)  
- Visualization of spatial relationships between functional and abortive structures  

Outputs include:

- Spatial heatmaps  
- Centroid coordinate tables  
- Distance metrics between objects  
- Statistical summaries of spatial distributions  
- Spatial maps with biological center reference  

This script is intended to characterize developmental spatial organization and positional biases within inflorescences.

---

## 2. CompareVolume.py

This script performs quantitative volumetric comparison between two segmented inflorescences, even when their segmentation label IDs differ.

Key functions:

- Load two labeled 3D TIFF segmentation volumes  
- Map numeric segmentation labels to biological categories  
- Compute absolute voxel volumes for each tissue category  
- Compute relative tissue composition (fraction of biological volume)  
- Generate comparative tables and bar plots  

Biological categories can include for example:

- Pistils  
- Vascular bundles  
- Parenchyma  
- Empty alveoli  

Outputs:

- CSV tables with volumetric statistics  
- Tissue composition summaries  
- Bar plots comparing tissue volumes between inflorescences  

This script allows direct structural comparison of inflorescence architecture across specimens or experimental treatments.

---

## 3. CompareSize.py

This script generates standardized visual comparisons between two inflorescences, allowing size and shape differences to be assessed under identical scaling conditions.

Features include:

- Extraction of top-view projections from 3D segmentation volumes  
- Automatic cropping of biological structures  
- Optional rescaling to a common physical voxel size  
- Padding to ensure identical canvas dimensions  
- Side-by-side visualization of inflorescences  

Output:

- Standardized side-by-side comparison figure

This script is useful for morphological comparison and figure preparation, ensuring that structures are displayed at the same scale and aligned consistently.

---

# Dependencies

The scripts require Python and the following packages:
numpy
pandas
matplotlib
scipy
tifffile

Install them with: pip install numpy pandas matplotlib scipy tifffile


---

# Input Data

All scripts expect segmented microCT volumes stored as labeled 3D TIFF files, with integer labels representing biological structures.

A typical segmentation scheme may include labels such as:
0 Background
1 Vascular bundles
2 Pistils
3 Parenchyma
4 Empty alveoli


Label mappings can be adjusted inside the scripts if segmentation schemes differ between datasets.

---

# Purpose

These scripts were developed to support the quantitative spatial and structural analysis of plant reproductive organs using microCT segmentation data. By providing the full analysis code used in the study, this repository ensures that the spatial analyses, volumetric measurements, and visualizations described in the publication can be fully reproduced and independently verified.
