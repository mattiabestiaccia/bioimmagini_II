Esempi Pratici - Esercizio 1
=============================

Questa sezione presenta esempi completi e casi d'uso reali per l'analisi del rumore
in immagini MRI.

Esempio 1: Analisi Completa su Immagine Sintetica
--------------------------------------------------

Questo esempio replica completamente l'analisi di ``calcolo_sd.py``.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from src.utils import (
       create_synthetic_image,
       compute_sd_map,
       estimate_sigma_from_histogram
   )

   # Parametri
   dim = 512
   sigma_true = 5.0
   kernel_size = 5

   # 1. Crea immagine sintetica con rumore Gaussiano
   image_clean, image_noisy = create_synthetic_image(
       dim=dim,
       sigma_noise=sigma_true
   )

   print(f"Immagine creata: {dim}×{dim}, σ_true = {sigma_true}")
   print(f"Range intensità: [{image_noisy.min():.1f}, {image_noisy.max():.1f}]")

   # 2. Calcola SD map
   sd_map = compute_sd_map(image_noisy, kernel_size=kernel_size)

   # 3. Stima sigma con tre metodi
   sigma_mean = np.mean(sd_map)
   sigma_median = np.median(sd_map)
   sigma_max, hist, bins = estimate_sigma_from_histogram(sd_map.ravel(), bins=100)

   # 4. Confronto risultati
   print("\nStima σ con metodi diversi:")
   print(f"  Vero:        {sigma_true:.3f}")
   print(f"  Media:       {sigma_mean:.3f} (errore: {abs(sigma_mean-sigma_true)/sigma_true*100:.1f}%)")
   print(f"  Mediana:     {sigma_median:.3f} (errore: {abs(sigma_median-sigma_true)/sigma_true*100:.1f}%)")
   print(f"  Max hist:    {sigma_max:.3f} (errore: {abs(sigma_max-sigma_true)/sigma_true*100:.1f}%)")

   # 5. Visualizza
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   axes[0, 0].imshow(image_noisy, cmap='gray')
   axes[0, 0].set_title('Immagine Rumorosa')
   axes[0, 0].axis('off')

   axes[0, 1].imshow(sd_map, cmap='hot')
   axes[0, 1].set_title('SD Map')
   axes[0, 1].axis('off')

   axes[1, 0].hist(image_noisy.ravel(), bins=256, alpha=0.7, edgecolor='black')
   axes[1, 0].set_title('Istogramma Immagine')
   axes[1, 0].set_xlabel('Intensità')

   axes[1, 1].plot(bins, hist, linewidth=2)
   axes[1, 1].axvline(sigma_true, color='g', linestyle='--', label=f'Vero: {sigma_true:.2f}')
   axes[1, 1].axvline(sigma_max, color='r', linestyle='--', label=f'Stimato: {sigma_max:.2f}')
   axes[1, 1].set_title('Istogramma SD Map')
   axes[1, 1].set_xlabel('SD')
   axes[1, 1].legend()

   plt.tight_layout()
   plt.show()

**Output atteso**:

.. code-block:: text

   Immagine creata: 512×512, σ_true = 5.0
   Range intensità: [18.3, 267.8]

   Stima σ con metodi diversi:
     Vero:        5.000
     Media:       4.873 (errore: 2.5%)
     Mediana:     4.916 (errore: 1.7%)
     Max hist:    5.023 (errore: 0.5%)

Il metodo del **massimo dell'istogramma** risulta il più accurato.

Esempio 2: Analisi Fantoccio MRI con Correzione Rayleigh
---------------------------------------------------------

Analisi completa su immagine DICOM reale.

.. code-block:: python

   import pydicom
   import numpy as np
   import matplotlib.pyplot as plt
   from src.utils import (
       compute_sd_map,
       apply_rayleigh_correction,
       estimate_sigma_from_histogram
   )

   # 1. Carica DICOM
   dcm = pydicom.dcmread("data/phantom.dcm")
   image = dcm.pixel_array.astype(float)

   print(f"Immagine DICOM caricata: {image.shape}")
   print(f"Range: [{image.min():.0f}, {image.max():.0f}]")

   # 2. Calcola SD map
   sd_map = compute_sd_map(image, kernel_size=5)

   # 3. Identifica background automaticamente
   # Background = pixel con intensità < 20° percentile
   threshold = np.percentile(image, 20)
   background_mask = image < threshold

   print(f"\nThreshold background: {threshold:.1f}")
   print(f"Pixel background: {background_mask.sum()} ({background_mask.sum()/image.size*100:.1f}%)")

   # 4. Stima σ nel background (con e senza correzione)
   sd_bkg_values = sd_map[background_mask]
   sigma_bkg_mean = np.mean(sd_bkg_values)
   sigma_bkg_corrected = apply_rayleigh_correction(sigma_bkg_mean)

   print(f"\nRumore background:")
   print(f"  σ raw:       {sigma_bkg_mean:.2f}")
   print(f"  σ corrected: {sigma_bkg_corrected:.2f}")
   print(f"  Fattore:     {sigma_bkg_corrected/sigma_bkg_mean:.3f} (atteso: 1.526)")

   # 5. Calcola SNR per regione centrale (fantoccio)
   center_roi = image[
       image.shape[0]//2 - 50:image.shape[0]//2 + 50,
       image.shape[1]//2 - 50:image.shape[1]//2 + 50
   ]
   mean_signal = np.mean(center_roi)
   snr = mean_signal / sigma_bkg_corrected

   print(f"\nSNR fantoccio centrale:")
   print(f"  Segnale medio: {mean_signal:.1f}")
   print(f"  SNR:           {snr:.1f}")

   # 6. Visualizza risultati
   fig, axes = plt.subplots(2, 3, figsize=(15, 10))

   # Immagine originale
   axes[0, 0].imshow(image, cmap='gray')
   axes[0, 0].set_title('Fantoccio MRI')
   axes[0, 0].axis('off')

   # SD map
   im1 = axes[0, 1].imshow(sd_map, cmap='hot')
   axes[0, 1].set_title('SD Map')
   axes[0, 1].axis('off')
   plt.colorbar(im1, ax=axes[0, 1])

   # Maschera background
   axes[0, 2].imshow(background_mask, cmap='gray')
   axes[0, 2].set_title(f'Background Mask (threshold={threshold:.0f})')
   axes[0, 2].axis('off')

   # Istogramma immagine
   axes[1, 0].hist(image.ravel(), bins=256, alpha=0.7, edgecolor='black')
   axes[1, 0].axvline(threshold, color='r', linestyle='--', label='Threshold')
   axes[1, 0].set_title('Istogramma Intensità')
   axes[1, 0].legend()

   # Istogramma SD map background
   axes[1, 1].hist(sd_bkg_values, bins=100, alpha=0.7, edgecolor='black', color='orange')
   axes[1, 1].axvline(sigma_bkg_mean, color='b', linestyle='--', label=f'Mean: {sigma_bkg_mean:.2f}')
   axes[1, 1].axvline(sigma_bkg_corrected, color='r', linestyle='--', label=f'Corrected: {sigma_bkg_corrected:.2f}')
   axes[1, 1].set_title('SD Background')
   axes[1, 1].legend()

   # SNR map (per visualizzazione)
   snr_map = np.divide(image, sigma_bkg_corrected, where=image > threshold)
   snr_map[background_mask] = 0
   im2 = axes[1, 2].imshow(snr_map, cmap='viridis', vmin=0, vmax=30)
   axes[1, 2].set_title('SNR Map')
   axes[1, 2].axis('off')
   plt.colorbar(im2, ax=axes[1, 2])

   plt.tight_layout()
   plt.show()

Esempio 3: Test Monte Carlo - Convergenza ROI
----------------------------------------------

Verifica convergenza stima σ al variare della dimensione ROI.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from src.utils import create_synthetic_image

   # Parametri
   sigma_true = 5.0
   roi_sizes = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
   n_iterations = 1000

   # Crea immagine sintetica
   _, image = create_synthetic_image(dim=512, sigma_noise=sigma_true)

   # Risultati
   results = {size: [] for size in roi_sizes}

   print("Test Monte Carlo - Convergenza stima σ")
   print(f"σ vero = {sigma_true}, Iterazioni per ROI = {n_iterations}\n")

   for roi_size in roi_sizes:
       for _ in range(n_iterations):
           # Estrai ROI casuale
           max_y = image.shape[0] - roi_size
           max_x = image.shape[1] - roi_size

           y = np.random.randint(0, max_y)
           x = np.random.randint(0, max_x)

           roi = image[y:y+roi_size, x:x+roi_size]

           # Calcola SD
           sigma_est = np.std(roi, ddof=1)
           results[roi_size].append(sigma_est)

       # Statistiche
       values = np.array(results[roi_size])
       mean_sigma = np.mean(values)
       std_sigma = np.std(values)
       error_pct = abs(mean_sigma - sigma_true) / sigma_true * 100

       print(f"ROI {roi_size:3d}×{roi_size:<3d} → "
             f"μ={mean_sigma:.3f}, σ={std_sigma:.3f}, "
             f"errore={error_pct:.2f}%")

   # Visualizza convergenza
   fig, axes = plt.subplots(1, 2, figsize=(14, 5))

   # Plot 1: Media e SD dell'errore
   mean_errors = [abs(np.mean(results[s]) - sigma_true) / sigma_true * 100
                   for s in roi_sizes]
   std_errors = [np.std(results[s]) / sigma_true * 100
                  for s in roi_sizes]

   axes[0].plot(roi_sizes, mean_errors, 'o-', label='Errore medio (%)', linewidth=2)
   axes[0].plot(roi_sizes, std_errors, 's-', label='SD errore (%)', linewidth=2)
   axes[0].set_xlabel('Dimensione ROI (pixel)', fontsize=12)
   axes[0].set_ylabel('Errore relativo (%)', fontsize=12)
   axes[0].set_title('Convergenza stima σ vs dimensione ROI', fontweight='bold')
   axes[0].legend()
   axes[0].grid(True, alpha=0.3)
   axes[0].set_xscale('log')
   axes[0].set_yscale('log')

   # Plot 2: Distribuzione per ROI selezionate
   selected_sizes = [5, 20, 50, 100]
   for size in selected_sizes:
       axes[1].hist(results[size], bins=50, alpha=0.5,
                    label=f'ROI {size}×{size}', density=True)

   axes[1].axvline(sigma_true, color='r', linestyle='--', linewidth=2, label='σ vero')
   axes[1].set_xlabel('σ stimato', fontsize=12)
   axes[1].set_ylabel('Densità probabilità', fontsize=12)
   axes[1].set_title('Distribuzione stime per diverse ROI', fontweight='bold')
   axes[1].legend()
   axes[1].grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

**Output atteso**:

.. code-block:: text

   Test Monte Carlo - Convergenza stima σ
   σ vero = 5.0, Iterazioni per ROI = 1000

   ROI   5×  5 → μ=5.023, σ=1.598, errore=0.46%
   ROI  10× 10 → μ=5.006, σ=0.715, errore=0.12%
   ROI  15× 15 → μ=4.997, σ=0.463, errore=0.06%
   ROI  20× 20 → μ=5.001, σ=0.339, errore=0.02%
   ROI  30× 30 → μ=4.999, σ=0.226, errore=0.02%
   ROI  50× 50 → μ=5.000, σ=0.139, errore=0.00%
   ROI  75× 75 → μ=5.000, σ=0.092, errore=0.00%
   ROI 100×100 → μ=5.000, σ=0.069, errore=0.00%
   ROI 150×150 → μ=5.000, σ=0.046, errore=0.00%
   ROI 200×200 → μ=5.000, σ=0.035, errore=0.00%

**Conclusioni**:

- ROI 5×5: Alta variabilità (SD=1.6), inaffidabile
- ROI 20×20: Buon compromesso (SD=0.34)
- ROI ≥50×50: Eccellente accuratezza ma meno pratico

Esempio 4: Analisi Multi-Slice Serie Cardiaca
----------------------------------------------

Analisi rumore su serie DICOM multi-slice.

.. code-block:: python

   import pydicom
   import numpy as np
   import matplotlib.pyplot as plt
   from pathlib import Path
   from src.utils import compute_sd_map, apply_rayleigh_correction

   # Carica serie
   dcm_dir = Path("data/esempio_LGE")
   dcm_files = sorted(dcm_dir.glob("*.dcm"))

   print(f"Serie trovata: {len(dcm_files)} slices\n")

   # Array per risultati
   slice_numbers = []
   sigma_values = []
   snr_values = []

   for dcm_file in dcm_files:
       dcm = pydicom.dcmread(dcm_file)
       image = dcm.pixel_array.astype(float)
       slice_num = dcm.InstanceNumber

       # Calcola SD map
       sd_map = compute_sd_map(image, kernel_size=5)

       # Background: 10° percentile
       threshold = np.percentile(image, 10)
       background_mask = image < threshold

       # Stima rumore
       sigma_bkg = np.mean(sd_map[background_mask])
       sigma_corrected = apply_rayleigh_correction(sigma_bkg)

       # SNR medio (foreground)
       foreground = image[image > threshold]
       mean_signal = np.mean(foreground)
       snr = mean_signal / sigma_corrected

       slice_numbers.append(slice_num)
       sigma_values.append(sigma_corrected)
       snr_values.append(snr)

       print(f"Slice {slice_num:2d}: σ={sigma_corrected:5.2f}, SNR={snr:5.1f}")

   # Identifica slice ottimale
   best_idx = np.argmax(snr_values)
   print(f"\nSlice ottimale: {slice_numbers[best_idx]} (SNR={snr_values[best_idx]:.1f})")

   # Visualizza trend
   fig, axes = plt.subplots(1, 2, figsize=(14, 5))

   axes[0].plot(slice_numbers, sigma_values, 'o-', linewidth=2, markersize=8)
   axes[0].set_xlabel('Slice Number', fontsize=12)
   axes[0].set_ylabel('σ (rumore corrected)', fontsize=12)
   axes[0].set_title('Rumore attraverso le slices', fontweight='bold')
   axes[0].grid(True, alpha=0.3)

   axes[1].plot(slice_numbers, snr_values, 's-', color='green', linewidth=2, markersize=8)
   axes[1].axhline(snr_values[best_idx], color='r', linestyle='--',
                   label=f'Max SNR (slice {slice_numbers[best_idx]})')
   axes[1].set_xlabel('Slice Number', fontsize=12)
   axes[1].set_ylabel('SNR', fontsize=12)
   axes[1].set_title('SNR attraverso le slices', fontweight='bold')
   axes[1].legend()
   axes[1].grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Esempio 5: Confronto Kernel Size per SD Map
--------------------------------------------

Valuta l'effetto della dimensione del kernel sulla SD map.

.. code-block:: python

   from src.utils import create_synthetic_image, compute_sd_map
   import numpy as np
   import matplotlib.pyplot as plt

   # Crea immagine
   sigma_true = 5.0
   _, image = create_synthetic_image(dim=512, sigma_noise=sigma_true)

   # Testa kernel sizes
   kernel_sizes = [3, 5, 7, 9, 11]

   fig, axes = plt.subplots(2, len(kernel_sizes), figsize=(15, 6))

   for i, ksize in enumerate(kernel_sizes):
       sd_map = compute_sd_map(image, kernel_size=ksize)

       sigma_est = np.median(sd_map)
       error = abs(sigma_est - sigma_true) / sigma_true * 100

       # Immagine SD map
       im = axes[0, i].imshow(sd_map, cmap='hot', vmin=0, vmax=10)
       axes[0, i].set_title(f'Kernel {ksize}×{ksize}')
       axes[0, i].axis('off')
       plt.colorbar(im, ax=axes[0, i])

       # Istogramma
       axes[1, i].hist(sd_map.ravel(), bins=100, alpha=0.7, edgecolor='black')
       axes[1, i].axvline(sigma_true, color='g', linestyle='--', linewidth=2, label='True')
       axes[1, i].axvline(sigma_est, color='r', linestyle='--', linewidth=2, label='Est')
       axes[1, i].set_title(f'σ={sigma_est:.2f}, err={error:.1f}%')
       axes[1, i].set_xlabel('SD')
       if i == 0:
           axes[1, i].set_ylabel('Frequency')
       axes[1, i].legend(fontsize=8)

   plt.suptitle(f'Effetto Kernel Size (σ vero = {sigma_true})', fontweight='bold', fontsize=14)
   plt.tight_layout()
   plt.show()

**Conclusioni**: Kernel 5×5 è generalmente il miglior compromesso tra risoluzione spaziale
e accuratezza della stima.

Note Pratiche
-------------

Performance Tips
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Più veloce per immagini grandi
   from scipy.ndimage import uniform_filter

   def compute_sd_map_fast(image, kernel_size=5):
       # Media locale
       mean_local = uniform_filter(image, size=kernel_size)
       # Varianza locale
       mean_sq_local = uniform_filter(image**2, size=kernel_size)
       var_local = mean_sq_local - mean_local**2
       return np.sqrt(np.maximum(var_local, 0))  # evita negativi per errori numerici

Interpretazione Risultati
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def interpret_snr(snr):
       """Interpreta valore SNR."""
       if snr < 5:
           return "Basso - Immagine molto rumorosa, qualità diagnostica compromessa"
       elif snr < 10:
           return "Accettabile - Diagnostica possibile ma limitata"
       elif snr < 20:
           return "Buono - Qualità diagnostica adeguata"
       elif snr < 30:
           return "Molto buono - Eccellente qualità diagnostica"
       else:
           return "Eccellente - Qualità ottimale per analisi dettagliate"

   snr_measured = 18.5
   print(f"SNR={snr_measured:.1f}: {interpret_snr(snr_measured)}")

Riferimenti
-----------

Per teoria dettagliata: :doc:`theory`

Per documentazione API: :doc:`api`

Per guide d'uso: :doc:`usage`
