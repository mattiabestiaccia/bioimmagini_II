Usage - Esercizio 1
====================

Questa guida mostra come utilizzare i moduli dell'Esercitazione 1 per l'analisi
del rumore in immagini MRI.

Installazione e Setup
----------------------

Prerequisiti
~~~~~~~~~~~~

.. code-block:: bash

   cd esercitazioni/esercitazioni_python/es_1__09_03_2022_calcolo_sd
   source ../activate.sh  # Attiva virtual environment

Verifica installazione:

.. code-block:: python

   import numpy as np
   import scipy
   import matplotlib.pyplot as plt
   import pydicom  # Opzionale per DICOM
   print("Tutte le dipendenze installate!")

Script Disponibili
------------------

1. calcolo_sd.py
~~~~~~~~~~~~~~~~

**Scopo**: Analisi SD map su immagine sintetica con rumore Gaussiano noto.

**Esecuzione**:

.. code-block:: bash

   python src/calcolo_sd.py

**Output**:

- ``results/calcolo_sd/01_noisy_image.png`` - Immagine rumorosa e istogramma
- ``results/calcolo_sd/02_sd_map.png`` - SD map e istogramma
- ``results/calcolo_sd/03_sigma_comparison.png`` - Confronto metodi stima

**Console output**:

.. code-block:: text

   ======================================================================
   Standard Deviation Calculation on Synthetic Image
   ======================================================================

   Image dimensions: 512x512
   True sigma (Gaussian noise): 5.0
   SD map kernel size: 5x5

   Method               Value      Error (%)
   ----------------------------------------
   True Sigma           5.0000     -
   Mean of SD map       4.8732     2.54
   Median of SD map     4.9156     1.69
   Max Histogram        5.0234     0.47

2. esempio_calcolo_sd.py
~~~~~~~~~~~~~~~~~~~~~~~~~

**Scopo**: Analisi su fantoccio MRI reale (file DICOM).

**Esecuzione**:

.. code-block:: bash

   python src/esempio_calcolo_sd.py

**Parametri configurabili**:

.. code-block:: python

   KERNEL_SIZE = 5          # Dimensione kernel SD map
   HISTOGRAM_BINS = 256     # Numero bin istogramma
   PHANTOM_PATH = "data/phantom.dcm"

**Workflow**:

1. Carica DICOM fantoccio
2. Calcola SD map
3. Identifica background automaticamente (threshold)
4. Stima σ e applica correzione Rayleigh
5. Visualizza risultati

3. test_m_sd.py
~~~~~~~~~~~~~~~

**Scopo**: Test Monte Carlo per convergenza stime σ vs dimensione ROI.

**Esecuzione**:

.. code-block:: bash

   python src/test_m_sd.py

**Parametri**:

.. code-block:: python

   ROI_SIZES = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
   N_ITERATIONS = 1000  # Ripetizioni per ogni dimensione
   TRUE_SIGMA = 5.0

**Output**:

- Grafici convergenza media/SD dell'errore vs dimensione ROI
- Tabella statistica: mean error, SD error, min/max per ogni ROI size

Esempi Pratici
--------------

Esempio 1: Stima Rumore su Propria Immagine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils import compute_sd_map, estimate_sigma_from_histogram
   import numpy as np
   import pydicom

   # Carica tua immagine DICOM
   dcm = pydicom.dcmread("path/to/your/image.dcm")
   image = dcm.pixel_array.astype(float)

   # Calcola SD map
   sd_map = compute_sd_map(image, kernel_size=5)

   # Stima sigma con 3 metodi
   sigma_mean = np.mean(sd_map)
   sigma_median = np.median(sd_map)
   sigma_max, _, _ = estimate_sigma_from_histogram(sd_map.ravel(), bins=100)

   print(f"Sigma (media): {sigma_mean:.2f}")
   print(f"Sigma (mediana): {sigma_median:.2f}")
   print(f"Sigma (max hist): {sigma_max:.2f}")

Esempio 2: Correzione Rayleigh sul Background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils import apply_rayleigh_correction

   # Misura SD nel background (es. tramite ROI manuale)
   background_roi = image[10:50, 10:50]  # ROI 40x40
   sd_background = np.std(background_roi)

   print(f"SD background (raw): {sd_background:.2f}")

   # Applica correzione Rayleigh
   sd_corrected = apply_rayleigh_correction(sd_background)

   print(f"SD corretta: {sd_corrected:.2f}")
   print(f"Fattore moltiplicativo: {sd_corrected/sd_background:.3f}")  # ~1.526

Esempio 3: Calcolo SNR
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils import compute_sd_map, apply_rayleigh_correction
   import numpy as np

   # Supponi di avere immagine e maschera tessuto
   tissue_mask = (image > threshold)  # Maschera binaria
   background_mask = ~tissue_mask

   # Stima rumore dal background
   sd_bkg = np.mean(compute_sd_map(image, kernel_size=5)[background_mask])
   sigma_noise = apply_rayleigh_correction(sd_bkg)

   # Calcola intensità media tessuto
   mean_signal = np.mean(image[tissue_mask])

   # SNR
   snr = mean_signal / sigma_noise
   print(f"SNR = {snr:.1f}")

   # Interpretazione
   if snr < 5:
       quality = "Bassa - immagine molto rumorosa"
   elif snr < 20:
       quality = "Accettabile - diagnostica possibile"
   else:
       quality = "Alta - eccellente qualità"

   print(f"Qualità immagine: {quality}")

Esempio 4: Analisi Multi-Slice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   import pydicom
   import numpy as np
   from src.utils import compute_sd_map, apply_rayleigh_correction

   # Carica serie DICOM
   dcm_dir = Path("data/esempio_LGE")
   dcm_files = sorted(dcm_dir.glob("*.dcm"))

   snr_values = []

   for dcm_file in dcm_files:
       dcm = pydicom.dcmread(dcm_file)
       image = dcm.pixel_array.astype(float)

       # Stima rumore
       sd_map = compute_sd_map(image, kernel_size=5)
       sd_bkg = np.percentile(sd_map, 10)  # 10th percentile ~ background
       sigma = apply_rayleigh_correction(sd_bkg)

       # SNR medio (escludendo background)
       foreground = image[image > np.percentile(image, 20)]
       snr = np.mean(foreground) / sigma
       snr_values.append(snr)

       print(f"Slice {dcm.InstanceNumber:2d}: SNR = {snr:.1f}")

   # Identifica slice ottimale
   best_slice = np.argmax(snr_values)
   print(f"\nSlice con SNR massimo: {best_slice} (SNR = {snr_values[best_slice]:.1f})")

Workflow Tipico
---------------

Per Analisi Rumore Completa
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pathlib import Path
   from src.calcolo_sd import main as run_synthetic_analysis
   from src.esempio_calcolo_sd import main as run_phantom_analysis
   from src.test_m_sd import main as run_monte_carlo

   results_dir = Path("results")

   # 1. Validazione su sintetico
   print("=" * 60)
   print("Step 1: Validazione su immagine sintetica")
   print("=" * 60)
   synthetic_results = run_synthetic_analysis(
       output_dir=results_dir / "synthetic",
       show_plots=False
   )

   # 2. Analisi fantoccio
   print("\n" + "=" * 60)
   print("Step 2: Analisi fantoccio MRI")
   print("=" * 60)
   phantom_results = run_phantom_analysis(
       phantom_path="data/phantom.dcm",
       output_dir=results_dir / "phantom",
       show_plots=False
   )

   # 3. Test Monte Carlo
   print("\n" + "=" * 60)
   print("Step 3: Test Monte Carlo convergenza")
   print("=" * 60)
   mc_results = run_monte_carlo(
       roi_sizes=[5, 10, 20, 50, 100],
       n_iterations=500,
       output_dir=results_dir / "monte_carlo",
       show_plots=False
   )

   print("\n" + "=" * 60)
   print("Analisi completata! Risultati salvati in:", results_dir)
   print("=" * 60)

Troubleshooting
---------------

Problema: SD Map ha valori strani
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Sintomo**: SD map mostra valori molto alti o pattern inattesi.

**Soluzioni**:

1. Verifica tipo dati immagine:

   .. code-block:: python

      image = image.astype(float)  # Converte a float

2. Controlla range valori:

   .. code-block:: python

      print(f"Min: {image.min()}, Max: {image.max()}")

3. Normalizza se necessario:

   .. code-block:: python

      image = (image - image.min()) / (image.max() - image.min()) * 255

Problema: Correzione Rayleigh non migliora stima
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Causa**: Stai applicando la correzione a regioni con segnale (non background).

**Soluzione**: Applica correzione SOLO al background:

.. code-block:: python

   # SBAGLIATO - su tutta l'immagine
   sigma = apply_rayleigh_correction(np.std(image))

   # CORRETTO - solo sul background
   background_mask = (image < threshold)
   sigma_bkg = np.std(image[background_mask])
   sigma_corrected = apply_rayleigh_correction(sigma_bkg)

Problema: Import Error
~~~~~~~~~~~~~~~~~~~~~~

**Errore**: ``ModuleNotFoundError: No module named 'src'``

**Soluzione**:

.. code-block:: bash

   # Assicurati di essere nella directory corretta
   cd esercitazioni/esercitazioni_python/es_1__09_03_2022_calcolo_sd

   # Esegui script dalla directory principale
   python src/calcolo_sd.py

Oppure aggiungi al Python path:

.. code-block:: python

   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent / "src"))

Parametri Raccomandati
----------------------

+------------------------+------------------+--------------------------------+
| Parametro              | Valore Default   | Note                           |
+========================+==================+================================+
| ``kernel_size``        | 5                | 5×5 per immagini 512×512       |
+------------------------+------------------+--------------------------------+
| ``histogram_bins``     | 100-256          | 100 per SD map, 256 per image  |
+------------------------+------------------+--------------------------------+
| ``threshold_bkg``      | 10-20% percentile| Dipende da SNR immagine        |
+------------------------+------------------+--------------------------------+
| ``roi_size`` (Monte    | 20-50 pixel      | Compromesso accuratezza/pratica|
| Carlo)                 |                  |                                |
+------------------------+------------------+--------------------------------+

Riferimenti
-----------

Per approfondimenti teorici, vedi: :doc:`theory`

Per dettagli API complete, vedi: :doc:`api`

Per esempi avanzati, vedi: :doc:`examples`
