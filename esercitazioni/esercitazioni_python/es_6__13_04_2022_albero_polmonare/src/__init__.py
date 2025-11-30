"""
Esercitazione 6: Analisi Albero Bronchiale

Pacchetto per analisi automatica albero bronchiale da CT toracica ad alta risoluzione.

Moduli:
    - utils: Funzioni core per segmentazione e analisi
    - bronchial_tree_analysis: Script principale pipeline completa

Pipeline:
    1. Caricamento volume CT 3D con Hounsfield Units
    2. Interpolazione isotropa (voxel cubici)
    3. Region growing 3D da seed in trachea
    4. Filtraggio maschera (riempimento buchi)
    5. Skeletonization 3D per centerline extraction
    6. Identificazione endpoints
    7. Estrazione percorso da endpoint a trachea
    8. Sphere method per misurazione diametro
    9. Grafici diametro vs distanza

Valori attesi:
    - Diametro trachea: 15-18 mm
    - Diametro bronchi primari: 10-12 mm

Dataset:
    - 148 slice DICOM CT toracica
    - Hounsfield Units: aria ~-1000 HU
    - Cancer Imaging Archive (LIDC-IDRI)

Author: Generated with Claude Code
Date: 2025-11-20
"""

__version__ = '1.0.0'
__author__ = 'Claude Code'

from .utils import (
    load_ct_volume_hounsfield,
    check_isotropic,
    interpolate_to_isotropic,
    region_growing_3d,
    fill_holes_3d,
    skeletonize_3d_clean,
    find_skeleton_endpoints,
    extract_centerline_path,
    measure_diameter_along_path,
    smooth_diameters
)

__all__ = [
    'load_ct_volume_hounsfield',
    'check_isotropic',
    'interpolate_to_isotropic',
    'region_growing_3d',
    'fill_holes_3d',
    'skeletonize_3d_clean',
    'find_skeleton_endpoints',
    'extract_centerline_path',
    'measure_diameter_along_path',
    'smooth_diameters'
]
