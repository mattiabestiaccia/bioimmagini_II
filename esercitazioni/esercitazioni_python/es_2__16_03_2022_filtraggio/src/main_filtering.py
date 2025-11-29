"""
ESERCITAZIONE 2 - Filtraggio 3D su Immagini CT

Script principale per valutare algoritmi di filtraggio 3D su fantoccio CT
in termini di SNR e conservazione delle transizioni.

Algoritmi testati:
1. Filtro a media mobile 7x7x7
2. Filtro Gaussiano 7x7x7 (con ottimizzazione sigma)
3. Filtro Wiener adattivo 7x7x7

Dataset: RIDER Phantom PET-CT
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Tuple

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent))

from dicom_utils import load_dicom_volume, make_isotropic, check_isotropy
from filters_3d import (
    moving_average_filter_3d,
    gaussian_filter_3d,
    wiener_filter_3d,
    estimate_noise_variance
)
from metrics import (
    calculate_snr,
    create_circular_roi,
    extract_profile,
    calculate_edge_sharpness,
    calculate_multiple_metrics
)


def optimize_gaussian_sigma(
    volume: np.ndarray,
    roi_mask_3d: np.ndarray,
    profile_start: Tuple[int, int],
    profile_end: Tuple[int, int],
    central_slice: int,
    sigma_range: np.ndarray = np.linspace(0.5, 3.0, 20),
    kernel_size: int = 7
) -> Tuple[float, dict]:
    """
    Ottimizza il valore di sigma per il filtro Gaussiano.

    Cerca il sigma che massimizza SNR mantenendo buona acutezza.

    Parameters
    ----------
    volume : np.ndarray
        Volume 3D isotropo
    roi_mask_3d : np.ndarray
        Maschera ROI 3D per calcolo SNR
    profile_start, profile_end : tuple
        Coordinate profilo per acutezza
    central_slice : int
        Indice slice centrale
    sigma_range : np.ndarray
        Range di sigma da testare
    kernel_size : int
        Dimensione kernel

    Returns
    -------
    best_sigma : float
        Valore ottimale di sigma
    results : dict
        Risultati per tutti i sigma testati
    """
    from typing import Tuple

    print("\n" + "="*60)
    print("OTTIMIZZAZIONE SIGMA FILTRO GAUSSIANO")
    print("="*60)

    snr_values = []
    sharpness_values = []

    for sigma in sigma_range:
        # Filtra con questo sigma
        filtered = gaussian_filter_3d(volume, kernel_size=kernel_size, sigma=sigma)

        # Estrai slice centrale
        slice_2d = filtered[:, :, central_slice]
        roi_2d = roi_mask_3d[:, :, central_slice]

        # Calcola metriche
        snr = calculate_snr(slice_2d, roi_2d)
        profile = extract_profile(slice_2d, profile_start, profile_end)
        sharpness = calculate_edge_sharpness(profile, method='gradient')

        snr_values.append(snr)
        sharpness_values.append(sharpness)

    snr_values = np.array(snr_values)
    sharpness_values = np.array(sharpness_values)

    # Normalizza per confronto
    snr_norm = (snr_values - snr_values.min()) / (snr_values.max() - snr_values.min())
    sharp_norm = (sharpness_values - sharpness_values.min()) / (sharpness_values.max() - sharpness_values.min())

    # Funzione obiettivo: massimizza SNR con peso alla conservazione acutezza
    # Peso maggiore all'SNR (70%) e minore all'acutezza (30%)
    objective = 0.7 * snr_norm + 0.3 * sharp_norm

    # Trova sigma ottimale
    best_idx = np.argmax(objective)
    best_sigma = sigma_range[best_idx]

    print(f"\nSigma ottimale: {best_sigma:.2f}")
    print(f"  SNR: {snr_values[best_idx]:.2f}")
    print(f"  Acutezza: {sharpness_values[best_idx]:.4f}")

    results = {
        'sigma_range': sigma_range,
        'snr_values': snr_values,
        'sharpness_values': sharpness_values,
        'best_sigma': best_sigma,
        'best_idx': best_idx
    }

    return best_sigma, results


def main():
    """Esegue l'esercitazione completa di filtraggio 3D."""

    # =========================================================================
    # CONFIGURAZIONE
    # =========================================================================

    # Path ai dati
    data_dir = Path(__file__).parent.parent / "data"
    ct_series_path = data_dir / "Phantom_CT_PET" / "2-CT 2.5mm-5.464"

    # Directory output
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Parametri filtri
    KERNEL_SIZE = 7

    # ROI e profilo (da definire dopo aver visualizzato l'immagine)
    # Questi sono valori di esempio, da adattare all'immagine specifica
    ROI_CENTER = (256, 256)  # Centro della ROI circolare (adattare)
    ROI_RADIUS = 80          # Raggio ROI in pixel (adattare)

    # Profilo per misurare acutezza (perpendicolare al fantoccio)
    PROFILE_START = (256, 150)  # Punto iniziale profilo (adattare)
    PROFILE_END = (256, 350)    # Punto finale profilo (adattare)

    print("="*60)
    print("ESERCITAZIONE 2 - FILTRAGGIO 3D SU IMMAGINI CT")
    print("="*60)
    print(f"Dataset: RIDER Phantom PET-CT")
    print(f"Serie CT: {ct_series_path.name}")
    print(f"Kernel size: {KERNEL_SIZE}x{KERNEL_SIZE}x{KERNEL_SIZE}")
    print("="*60)

    # =========================================================================
    # STEP 1: CARICAMENTO VOLUME DICOM
    # =========================================================================

    print("\n" + "="*60)
    print("STEP 1: CARICAMENTO VOLUME DICOM")
    print("="*60)

    volume, metadata = load_dicom_volume(str(ct_series_path))

    # Verifica range valori
    print(f"\nStatistiche volume:")
    print(f"  Min HU: {volume.min():.1f}")
    print(f"  Max HU: {volume.max():.1f}")
    print(f"  Mean HU: {volume.mean():.1f}")
    print(f"  Std HU: {volume.std():.1f}")

    # =========================================================================
    # STEP 2: INTERPOLAZIONE PER VOLUME ISOTROPO
    # =========================================================================

    print("\n" + "="*60)
    print("STEP 2: VERIFICA ISOTROPIA E INTERPOLAZIONE")
    print("="*60)

    if check_isotropy(metadata):
        print("Il volume è già isotropo. Nessuna interpolazione necessaria.")
        iso_volume = volume
        iso_spacing = metadata['PixelSpacing'][0]
    else:
        print("Il volume non è isotropo. Applico interpolazione trilineare...")
        iso_volume, iso_spacing = make_isotropic(volume, metadata)

    # =========================================================================
    # STEP 3: DEFINIZIONE ROI E PROFILO
    # =========================================================================

    print("\n" + "="*60)
    print("STEP 3: DEFINIZIONE ROI E PROFILO")
    print("="*60)

    # Seleziona slice centrale
    central_slice_idx = iso_volume.shape[2] // 2
    central_slice = iso_volume[:, :, central_slice_idx]

    print(f"\nSlice centrale: {central_slice_idx}/{iso_volume.shape[2]}")
    print(f"Shape slice: {central_slice.shape}")

    # Crea ROI per SNR (circolare nella slice centrale)
    roi_2d = create_circular_roi(central_slice.shape, ROI_CENTER, ROI_RADIUS)

    # Estendi ROI a 3D per alcune operazioni
    roi_3d = np.zeros_like(iso_volume, dtype=bool)
    roi_3d[:, :, central_slice_idx] = roi_2d

    print(f"\nROI definita:")
    print(f"  Centro: {ROI_CENTER}")
    print(f"  Raggio: {ROI_RADIUS} pixel")
    print(f"  Pixel nella ROI: {roi_2d.sum()}")

    # Stima varianza del rumore nella ROI
    noise_variance = estimate_noise_variance(iso_volume, roi_3d)
    noise_std = np.sqrt(noise_variance)

    print(f"\nRumore stimato nella ROI:")
    print(f"  Varianza (σ²): {noise_variance:.4f}")
    print(f"  Std (σ): {noise_std:.4f}")

    # =========================================================================
    # STEP 4: APPLICAZIONE FILTRI
    # =========================================================================

    print("\n" + "="*60)
    print("STEP 4: APPLICAZIONE FILTRI 3D")
    print("="*60)

    # 4.1 Filtro a media mobile
    print("\n[1/3] Filtro a media mobile...")
    filtered_mean = moving_average_filter_3d(iso_volume, kernel_size=KERNEL_SIZE)
    print(f"  Completato. Shape: {filtered_mean.shape}")

    # 4.2 Filtro Gaussiano con ottimizzazione sigma
    print("\n[2/3] Filtro Gaussiano (ottimizzazione sigma)...")

    # Ottimizza sigma
    best_sigma, opt_results = optimize_gaussian_sigma(
        iso_volume,
        roi_3d,
        PROFILE_START,
        PROFILE_END,
        central_slice_idx,
        sigma_range=np.linspace(0.5, 3.0, 20),
        kernel_size=KERNEL_SIZE
    )

    # Applica filtro con sigma ottimale
    filtered_gaussian = gaussian_filter_3d(
        iso_volume,
        kernel_size=KERNEL_SIZE,
        sigma=best_sigma
    )
    print(f"  Completato con sigma={best_sigma:.2f}. Shape: {filtered_gaussian.shape}")

    # 4.3 Filtro Wiener adattivo
    print("\n[3/3] Filtro Wiener adattivo...")
    filtered_wiener = wiener_filter_3d(
        iso_volume,
        kernel_size=KERNEL_SIZE,
        noise_variance=noise_variance
    )
    print(f"  Completato. Shape: {filtered_wiener.shape}")

    # =========================================================================
    # STEP 5: CALCOLO METRICHE
    # =========================================================================

    print("\n" + "="*60)
    print("STEP 5: CALCOLO METRICHE (SNR E ACUTEZZA)")
    print("="*60)

    # Estrai slice centrali da tutti i volumi
    slice_original = iso_volume[:, :, central_slice_idx]
    slice_mean = filtered_mean[:, :, central_slice_idx]
    slice_gaussian = filtered_gaussian[:, :, central_slice_idx]
    slice_wiener = filtered_wiener[:, :, central_slice_idx]

    # Calcola metriche per ogni immagine
    images = {
        'Originale': slice_original,
        'Media Mobile': slice_mean,
        'Gaussiano': slice_gaussian,
        'Wiener': slice_wiener
    }

    results_table = []

    for name, img in images.items():
        metrics = calculate_multiple_metrics(
            img,
            roi_2d,
            PROFILE_START,
            PROFILE_END,
            sharpness_method='gradient'
        )

        results_table.append({
            'Immagine': name,
            'SNR': metrics['snr'],
            'Acutezza': metrics['sharpness']
        })

    # =========================================================================
    # STEP 6: VISUALIZZAZIONE RISULTATI
    # =========================================================================

    print("\n" + "="*60)
    print("STEP 6: RISULTATI")
    print("="*60)

    # Stampa tabella risultati
    print("\nTABELLA RISULTATI:")
    print("-" * 60)
    print(f"{'Immagine':<20} {'SNR':>15} {'Acutezza':>20}")
    print("-" * 60)
    for row in results_table:
        print(f"{row['Immagine']:<20} {row['SNR']:>15.2f} {row['Acutezza']:>20.4f}")
    print("-" * 60)

    # =========================================================================
    # STEP 7: PLOT E SALVATAGGIO
    # =========================================================================

    print("\n" + "="*60)
    print("STEP 7: GENERAZIONE PLOT")
    print("="*60)

    # Plot 1: Confronto slice centrali
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    image_list = [
        (slice_original, 'Originale'),
        (slice_mean, 'Filtro Media Mobile'),
        (slice_gaussian, f'Filtro Gaussiano (σ={best_sigma:.2f})'),
        (slice_wiener, 'Filtro Wiener Adattivo')
    ]

    for ax, (img, title) in zip(axes, image_list):
        im = ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plot1_path = results_dir / "confronto_filtri.png"
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    print(f"Salvato: {plot1_path}")
    plt.close()

    # Plot 2: Confronto profili
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, img in images.items():
        profile = extract_profile(img, PROFILE_START, PROFILE_END)
        ax.plot(profile, label=name, linewidth=2)

    ax.set_xlabel('Posizione lungo il profilo (pixel)')
    ax.set_ylabel('Valore HU')
    ax.set_title('Confronto Profili - Conservazione Transizioni')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot2_path = results_dir / "confronto_profili.png"
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"Salvato: {plot2_path}")
    plt.close()

    # Plot 3: Differenze rispetto all'originale
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    differences = [
        (slice_mean - slice_original, 'Media Mobile - Originale'),
        (slice_gaussian - slice_original, 'Gaussiano - Originale'),
        (slice_wiener - slice_original, 'Wiener - Originale')
    ]

    for ax, (diff, title) in zip(axes, differences):
        # Usa scala centrata sullo zero
        vmax = np.abs(diff).max()
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plot3_path = results_dir / "differenze_filtri.png"
    plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
    print(f"Salvato: {plot3_path}")
    plt.close()

    # Plot 4: Ottimizzazione sigma
    if opt_results is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        sigma_range = opt_results['sigma_range']
        snr_vals = opt_results['snr_values']
        sharp_vals = opt_results['sharpness_values']
        best_idx = opt_results['best_idx']

        # SNR vs sigma
        ax1.plot(sigma_range, snr_vals, 'o-', linewidth=2)
        ax1.axvline(best_sigma, color='red', linestyle='--', label=f'Ottimale: σ={best_sigma:.2f}')
        ax1.set_xlabel('Sigma')
        ax1.set_ylabel('SNR')
        ax1.set_title('SNR vs Sigma')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Acutezza vs sigma
        ax2.plot(sigma_range, sharp_vals, 'o-', linewidth=2, color='orange')
        ax2.axvline(best_sigma, color='red', linestyle='--', label=f'Ottimale: σ={best_sigma:.2f}')
        ax2.set_xlabel('Sigma')
        ax2.set_ylabel('Acutezza')
        ax2.set_title('Acutezza vs Sigma')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot4_path = results_dir / "ottimizzazione_sigma.png"
        plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
        print(f"Salvato: {plot4_path}")
        plt.close()

    # Salva risultati in file di testo
    results_txt_path = results_dir / "risultati.txt"
    with open(results_txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ESERCITAZIONE 2 - FILTRAGGIO 3D SU IMMAGINI CT\n")
        f.write("="*60 + "\n\n")

        f.write("CONFIGURAZIONE:\n")
        f.write(f"  Kernel size: {KERNEL_SIZE}x{KERNEL_SIZE}x{KERNEL_SIZE}\n")
        f.write(f"  Sigma Gaussiano ottimale: {best_sigma:.2f}\n")
        f.write(f"  ROI centro: {ROI_CENTER}, raggio: {ROI_RADIUS}\n")
        f.write(f"  Rumore stimato (σ): {noise_std:.4f}\n\n")

        f.write("RISULTATI:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Immagine':<20} {'SNR':>15} {'Acutezza':>20}\n")
        f.write("-" * 60 + "\n")
        for row in results_table:
            f.write(f"{row['Immagine']:<20} {row['SNR']:>15.2f} {row['Acutezza']:>20.4f}\n")
        f.write("-" * 60 + "\n")

    print(f"Salvato: {results_txt_path}")

    print("\n" + "="*60)
    print("ESERCITAZIONE COMPLETATA!")
    print("="*60)
    print(f"\nRisultati salvati in: {results_dir}")


if __name__ == "__main__":
    main()
