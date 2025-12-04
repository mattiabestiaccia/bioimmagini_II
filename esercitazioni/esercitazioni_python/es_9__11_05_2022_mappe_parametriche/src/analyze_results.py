#!/usr/bin/env python3
"""
analyze_results.py - Analyze T2* Mapping Results

This script loads the generated T2* maps, defines ROIs,
computes statistics, and estimates iron concentration.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

# Add current directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent))

from utils import compute_roi_statistics, estimate_iron_concentration

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def main():
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / 'results'
    
    patients = ['paziente1', 'paziente2']
    
    # Define ROIs (approximate coordinates for demonstration)
    # In a real scenario, these would be drawn manually or segmented automatically
    # Coordinates: (x, y)
    rois_def = {
        'paziente1': {
            'septum': {'center': (128, 128), 'radius': 10},  # Septum
            'liver': {'center': (60, 180), 'radius': 15},    # Liver (approx)
            'muscle': {'center': (200, 200), 'radius': 10}   # Paraspinal muscle
        },
        'paziente2': {
            'septum': {'center': (128, 128), 'radius': 10},
            'liver': {'center': (60, 180), 'radius': 15},
            'muscle': {'center': (200, 200), 'radius': 10}
        }
    }

    summary_report = {}

    for patient in patients:
        print(f"\nAnalyzing {patient}...")
        patient_dir = results_dir / patient
        t2star_path = patient_dir / 't2star_map_c-exp.npy'
        
        if not t2star_path.exists():
            print(f"  Error: {t2star_path} not found. Run t2star_mapping.py first.")
            continue
            
        t2star_map = np.load(t2star_path)
        h, w = t2star_map.shape
        
        # Create ROI masks
        masks = {}
        patient_stats = {}
        
        # Plot ROIs
        plt.figure(figsize=(10, 10))
        plt.imshow(t2star_map, cmap='jet', vmin=0, vmax=50)
        plt.title(f"{patient} - T2* Map with ROIs")
        plt.colorbar(label='T2* (ms)')
        
        for roi_name, roi_params in rois_def[patient].items():
            mask = create_circular_mask(h, w, roi_params['center'], roi_params['radius'])
            masks[roi_name] = mask
            
            # Compute stats
            stats = compute_roi_statistics(t2star_map, mask)
            patient_stats[roi_name] = stats
            
            # Estimate Iron
            organ = 'heart' if roi_name == 'septum' else 'liver' if roi_name == 'liver' else 'heart' # Default to heart for others
            lic = estimate_iron_concentration(stats['mean'], organ=organ)
            patient_stats[roi_name]['iron_conc'] = lic
            
            print(f"  {roi_name.capitalize()}: T2* = {stats['mean']:.1f} ± {stats['std']:.1f} ms")
            if roi_name in ['septum', 'liver']:
                 print(f"    Iron Conc: {lic:.1f} mg/g")

            # Draw ROI on plot
            circle = plt.Circle(roi_params['center'], roi_params['radius'], color='white', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            plt.text(roi_params['center'][0], roi_params['center'][1]-roi_params['radius']-5, 
                     roi_name, color='white', ha='center', fontsize=10, weight='bold')

        plt.savefig(patient_dir / 'rois_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved ROI visualization to {patient_dir / 'rois_visualization.png'}")
        
        summary_report[patient] = patient_stats

    # Save summary to JSON
    with open(results_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=4)
    print(f"\nSaved analysis summary to {results_dir / 'analysis_summary.json'}")

if __name__ == "__main__":
    main()
