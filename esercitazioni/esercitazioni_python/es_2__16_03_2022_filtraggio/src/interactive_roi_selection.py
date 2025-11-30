"""
Script interattivo per selezionare ROI e profilo sulla slice centrale.

Permette di visualizzare l'immagine e selezionare interattivamente:
1. Centro e raggio della ROI per il calcolo SNR
2. Punti iniziale e finale per il profilo di acutezza

I valori selezionati vengono stampati e possono essere copiati in main_filtering.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from dicom_utils import load_dicom_volume, make_isotropic, check_isotropy
from metrics import create_circular_roi, extract_profile


def interactive_roi_selection():
    """Selezione interattiva di ROI e profilo."""

    # Path ai dati
    data_dir = Path(__file__).parent.parent / "data"
    ct_series_path = data_dir / "Phantom_CT_PET" / "2-CT 2.5mm-5.464"

    print("="*60)
    print("SELEZIONE INTERATTIVA ROI E PROFILO")
    print("="*60)
    print("\nCaricamento volume DICOM...")

    # Carica volume
    volume, metadata = load_dicom_volume(str(ct_series_path))

    # Rendi isotropo se necessario
    if not check_isotropy(metadata):
        print("Interpolazione per volume isotropo...")
        iso_volume, iso_spacing = make_isotropic(volume, metadata)
    else:
        iso_volume = volume
        iso_spacing = metadata['PixelSpacing'][0]

    # Estrai slice centrale
    central_slice_idx = iso_volume.shape[2] // 2
    central_slice = iso_volume[:, :, central_slice_idx]

    print(f"Slice centrale: {central_slice_idx}")
    print(f"Shape: {central_slice.shape}")
    print(f"Range HU: [{central_slice.min():.1f}, {central_slice.max():.1f}]")

    # Applica windowing per visualizzazione migliore
    # Usa WindowCenter e WindowWidth dal DICOM se disponibili
    if metadata['WindowCenter'] is not None:
        # Gestisci caso in cui WindowCenter/Width sono liste
        wc = metadata['WindowCenter']
        ww = metadata['WindowWidth']
        if isinstance(wc, list):
            wc = wc[0]
        if isinstance(ww, list):
            ww = ww[0]
        vmin = wc - ww / 2
        vmax = wc + ww / 2
    else:
        # Usa percentili per auto-windowing
        vmin = np.percentile(central_slice, 1)
        vmax = np.percentile(central_slice, 99)

    print(f"\nWindowing: [{vmin:.1f}, {vmax:.1f}] HU")

    # =========================================================================
    # SELEZIONE ROI
    # =========================================================================

    print("\n" + "-"*60)
    print("SELEZIONE ROI PER SNR")
    print("-"*60)
    print("\nIstruzioni:")
    print("1. Clicca sul centro della ROI (area omogenea del fantoccio)")
    print("2. Chiudi la finestra quando hai finito")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(central_slice, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title('Clicca sul centro della ROI\n(area omogenea del fantoccio)')
    ax.axis('off')

    # Lista per memorizzare il click
    roi_center = []

    def onclick_roi(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            roi_center.clear()
            roi_center.append((y, x))  # (row, col)
            print(f"  Centro selezionato: row={y}, col={x}")

            # Disegna cerchio di esempio
            ax.clear()
            ax.imshow(central_slice, cmap='gray', vmin=vmin, vmax=vmax)
            circle = plt.Circle((x, y), 80, color='red', fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
            ax.set_title(f'Centro ROI: ({y}, {x})\nRaggio esempio: 80 pixel')
            ax.axis('off')
            fig.canvas.draw()

    cid_roi = fig.canvas.mpl_connect('button_press_event', onclick_roi)
    plt.show()

    if not roi_center:
        print("ERRORE: Nessun centro selezionato. Usando default.")
        roi_center = [(central_slice.shape[0]//2, central_slice.shape[1]//2)]

    roi_center = roi_center[0]

    # Chiedi raggio
    print(f"\nCentro ROI selezionato: {roi_center}")
    roi_radius = input("Inserisci raggio ROI in pixel [default=80]: ").strip()
    roi_radius = int(roi_radius) if roi_radius else 80

    print(f"ROI: centro={roi_center}, raggio={roi_radius}")

    # =========================================================================
    # SELEZIONE PROFILO
    # =========================================================================

    print("\n" + "-"*60)
    print("SELEZIONE PROFILO PER ACUTEZZA")
    print("-"*60)
    print("\nIstruzioni:")
    print("1. Clicca sul punto INIZIALE del profilo (dentro il fantoccio)")
    print("2. Clicca sul punto FINALE del profilo (fuori dal fantoccio)")
    print("3. Il profilo deve essere perpendicolare al bordo del fantoccio")
    print("4. Chiudi la finestra quando hai finito")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(central_slice, cmap='gray', vmin=vmin, vmax=vmax)

    # Disegna ROI selezionata
    roi_mask = create_circular_roi(central_slice.shape, roi_center, roi_radius)
    ax.contour(roi_mask, colors='red', linewidths=2, levels=[0.5])

    ax.set_title('Clicca punto INIZIALE e FINALE del profilo\n(perpendicolare al bordo)')
    ax.axis('off')

    profile_points = []

    def onclick_profile(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            profile_points.append((y, x))  # (row, col)
            print(f"  Punto {len(profile_points)}: row={y}, col={x}")

            ax.plot(x, y, 'go', markersize=10)

            if len(profile_points) == 2:
                # Disegna linea
                x_coords = [profile_points[0][1], profile_points[1][1]]
                y_coords = [profile_points[0][0], profile_points[1][0]]
                ax.plot(x_coords, y_coords, 'g-', linewidth=2)
                ax.set_title(f'Profilo: {profile_points[0]} → {profile_points[1]}')

            fig.canvas.draw()

    cid_profile = fig.canvas.mpl_connect('button_press_event', onclick_profile)
    plt.show()

    if len(profile_points) < 2:
        print("ERRORE: Profilo non completo. Usando default.")
        profile_points = [(roi_center[0], roi_center[1] - 100),
                         (roi_center[0], roi_center[1] + 100)]

    profile_start = profile_points[0]
    profile_end = profile_points[1]

    # =========================================================================
    # VISUALIZZAZIONE FINALE E OUTPUT
    # =========================================================================

    print("\n" + "="*60)
    print("PARAMETRI SELEZIONATI")
    print("="*60)

    print("\nCopia questi valori in main_filtering.py:\n")
    print("-"*60)
    print(f"ROI_CENTER = {roi_center}")
    print(f"ROI_RADIUS = {roi_radius}")
    print(f"PROFILE_START = {profile_start}")
    print(f"PROFILE_END = {profile_end}")
    print("-"*60)

    # Visualizzazione finale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Immagine con ROI e profilo
    ax1.imshow(central_slice, cmap='gray', vmin=vmin, vmax=vmax)
    roi_mask = create_circular_roi(central_slice.shape, roi_center, roi_radius)
    ax1.contour(roi_mask, colors='red', linewidths=2, levels=[0.5])
    ax1.plot([profile_start[1], profile_end[1]],
             [profile_start[0], profile_end[0]],
             'g-', linewidth=2, label='Profilo')
    ax1.plot(roi_center[1], roi_center[0], 'r+', markersize=15, markeredgewidth=2)
    ax1.set_title('ROI (rosso) e Profilo (verde)')
    ax1.legend()
    ax1.axis('off')

    # Profilo estratto
    profile = extract_profile(central_slice, profile_start, profile_end)
    ax2.plot(profile, linewidth=2)
    ax2.set_xlabel('Posizione lungo il profilo (pixel)')
    ax2.set_ylabel('Valore HU')
    ax2.set_title('Profilo Estratto')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nSelezione completata!")


if __name__ == "__main__":
    interactive_roi_selection()
