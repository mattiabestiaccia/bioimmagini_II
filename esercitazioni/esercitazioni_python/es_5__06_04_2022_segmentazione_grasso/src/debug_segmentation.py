"""Debug script per analizzare problema SAT=0"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_abdominal_volume,
    kmeans_fat_segmentation,
    remove_spurious_components
)

# Carica dati
dicom_dir = Path('../data/dicom')
volume, metadata = load_abdominal_volume(dicom_dir)

# K-means
labels_volume, centroids, tissue_map = kmeans_fat_segmentation(volume)
fat_label = tissue_map['fat']
fat_mask_kmeans = (labels_volume == fat_label).astype(np.uint8)
torso_mask_3d = remove_spurious_components(fat_mask_kmeans, keep_largest=True)

# Analizza slice centrale
mid_slice = volume.shape[0] // 2
print(f"Slice centrale: {mid_slice}")
print(f"Tissue map: {tissue_map}")
print(f"Centroids: {centroids}")
print(f"Torso mask sum: {np.sum(torso_mask_3d[mid_slice])}")

# Visualizza
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(volume[mid_slice], cmap='gray')
axes[0].set_title('Originale')

axes[1].imshow(labels_volume[mid_slice], cmap='tab10')
axes[1].set_title(f'K-means (K=3)')

axes[2].imshow(fat_mask_kmeans[mid_slice], cmap='Reds')
axes[2].set_title(f'Fat mask (label={fat_label})')

axes[3].imshow(torso_mask_3d[mid_slice], cmap='Reds')
axes[3].set_title('Torso mask')

plt.tight_layout()
plt.savefig('../results/debug_kmeans.png', dpi=150)
print("Salvato: ../results/debug_kmeans.png")

# Test active contour con parametri ridotti
from utils import segment_sat_with_active_contours

print("\nTest active contours...")
sat_mask, outer, inner = segment_sat_with_active_contours(
    volume[mid_slice],
    torso_mask_3d[mid_slice],
    outer_iterations=50,  # Ridotto per debug
    inner_iterations=50
)

print(f"Outer sum: {np.sum(outer)}")
print(f"Inner sum: {np.sum(inner)}")
print(f"SAT sum: {np.sum(sat_mask)}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(volume[mid_slice], cmap='gray')
axes[0].imshow(outer, cmap='Reds', alpha=0.5)
axes[0].set_title(f'Outer contour (sum={np.sum(outer)})')

axes[1].imshow(volume[mid_slice], cmap='gray')
axes[1].imshow(inner, cmap='Blues', alpha=0.5)
axes[1].set_title(f'Inner contour (sum={np.sum(inner)})')

axes[2].imshow(volume[mid_slice], cmap='gray')
axes[2].imshow(sat_mask, cmap='Greens', alpha=0.5)
axes[2].set_title(f'SAT mask (sum={np.sum(sat_mask)})')

plt.tight_layout()
plt.savefig('../results/debug_active_contours.png', dpi=150)
print("Salvato: ../results/debug_active_contours.png")
