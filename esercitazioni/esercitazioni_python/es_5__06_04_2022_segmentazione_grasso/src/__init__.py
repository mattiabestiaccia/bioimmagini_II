"""
Esercitazione 5: Segmentazione Grasso Addominale (SAT/VAT)

Pacchetto per segmentazione automatica del grasso addominale da MRI T1-pesate.

Moduli:
    - utils: Funzioni core per segmentazione (K-means, active contours, EM-GMM)
    - fat_segmentation: Script principale pipeline completa
    - visualize_results: Visualizzazione avanzata risultati

Pipeline:
    1. K-means clustering (K=3) per separazione aria/acqua/grasso
    2. Rimozione braccia tramite connected component labeling
    3. Active contours doppi per SAT (bordo esterno cute + interno fascia)
    4. EM-GMM su istogramma intra-addominale per VAT
    5. Calcolo volumi SAT, VAT, VAT/SAT ratio

Valori attesi:
    - SAT: ~2091 cm^3
    - VAT: ~970 cm^3
    - VAT/SAT: ~46%

Riferimenti:
    Positano et al., "Accurate segmentation of subcutaneous and
    intermuscular adipose tissue from MR images of the thigh",
    Journal of Magnetic Resonance Imaging, 2004.

Author: Generated with Claude Code
Date: 2025-11-20
"""

__version__ = '1.0.0'
__author__ = 'Claude Code'

# Import funzioni principali per accesso diretto
from .utils import (
    load_abdominal_volume,
    kmeans_fat_segmentation,
    remove_spurious_components,
    segment_sat_with_active_contours,
    fit_em_gmm,
    extract_vat_from_gmm,
    calculate_fat_volumes,
    process_fat_segmentation_pipeline
)

__all__ = [
    'load_abdominal_volume',
    'kmeans_fat_segmentation',
    'remove_spurious_components',
    'segment_sat_with_active_contours',
    'fit_em_gmm',
    'extract_vat_from_gmm',
    'calculate_fat_volumes',
    'process_fat_segmentation_pipeline'
]
