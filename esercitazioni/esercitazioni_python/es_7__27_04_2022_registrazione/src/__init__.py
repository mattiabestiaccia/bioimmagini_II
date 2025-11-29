"""
Esercitazione 7: Registrazione Immagini con Algoritmi Genetici

Pacchetto per registrazione automatica immagini MRI synthetic (BrainWeb)
usando Differential Evolution (GA-like) + Mutual Information.

Author: Generated with Claude Code
Date: 2025-11-20
"""

__version__ = '1.0.0'
__author__ = 'Claude Code'

from .utils import (
    load_minc_slice,
    pad_to_square,
    random_rigid_transform_2d,
    apply_rigid_transform_2d,
    compute_mutual_information,
    register_with_differential_evolution,
    bland_altman_stats
)

__all__ = [
    'load_minc_slice',
    'pad_to_square',
    'random_rigid_transform_2d',
    'apply_rigid_transform_2d',
    'compute_mutual_information',
    'register_with_differential_evolution',
    'bland_altman_stats'
]
