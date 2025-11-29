API Reference - Esercizio 1
============================

Questa sezione fornisce la documentazione completa auto-generata dai docstring
dei moduli Python dell'Esercitazione 1.

Package Overview
----------------

.. automodule:: src
   :members:
   :undoc-members:
   :show-inheritance:

Modulo: utils
-------------

Funzioni utility per l'analisi del rumore MRI.

.. automodule:: src.utils
   :members:
   :undoc-members:
   :show-inheritance:

Funzioni Principali
~~~~~~~~~~~~~~~~~~~

compute_sd_map
^^^^^^^^^^^^^^

.. autofunction:: src.utils.compute_sd_map

estimate_sigma_from_histogram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.utils.estimate_sigma_from_histogram

rayleigh_correction_factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.utils.rayleigh_correction_factor

apply_rayleigh_correction
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.utils.apply_rayleigh_correction

compute_statistics
^^^^^^^^^^^^^^^^^^

.. autofunction:: src.utils.compute_statistics

create_synthetic_image
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.utils.create_synthetic_image

exclude_zero_padding
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.utils.exclude_zero_padding

Modulo: calcolo_sd
------------------

Script per analisi SD map su immagine sintetica.

.. automodule:: src.calcolo_sd
   :members:
   :undoc-members:
   :show-inheritance:

Funzioni di Visualizzazione
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plot_noisy_image_with_histogram
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.calcolo_sd.plot_noisy_image_with_histogram

plot_sd_map_with_histogram
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.calcolo_sd.plot_sd_map_with_histogram

plot_sigma_comparison
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: src.calcolo_sd.plot_sigma_comparison

Main Function
~~~~~~~~~~~~~

.. autofunction:: src.calcolo_sd.main

Modulo: esempio_calcolo_sd
---------------------------

Analisi su fantoccio MRI reale.

.. automodule:: src.esempio_calcolo_sd
   :members:
   :undoc-members:
   :show-inheritance:

Modulo: test_m_sd
-----------------

Test Monte Carlo per convergenza stime.

.. automodule:: src.test_m_sd
   :members:
   :undoc-members:
   :show-inheritance:

Esempi di Utilizzo
------------------

Calcolo SD Map
~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils import compute_sd_map
   import numpy as np

   # Crea immagine test
   image = np.random.randn(256, 256) * 5 + 100

   # Calcola SD map con kernel 5x5
   sd_map = compute_sd_map(image, kernel_size=5)

   # Stima sigma
   sigma_mean = np.mean(sd_map)
   sigma_median = np.median(sd_map)

Correzione Rayleigh
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils import apply_rayleigh_correction

   # SD misurata nel background
   sd_background = 12.5

   # Applica correzione Rayleigh
   sd_corrected = apply_rayleigh_correction(sd_background)
   print(f"SD corretta: {sd_corrected:.2f}")  # ~19.08

Analisi Completa
~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.calcolo_sd import main
   from pathlib import Path

   # Esegui analisi completa
   results = main(
       output_dir=Path("results"),
       show_plots=True
   )

   # Accedi ai risultati
   print(f"Sigma vero: {results['sigma_true']:.2f}")
   print(f"Sigma stimato (media): {results['sigma_mean']:.2f}")
   print(f"Sigma stimato (mediana): {results['sigma_median']:.2f}")
   print(f"Sigma stimato (max hist): {results['sigma_max']:.2f}")

Note Implementative
-------------------

Equivalenze MATLAB-Python
~~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+----------------------------------------+
| MATLAB                    | Python (questo modulo)                 |
+===========================+========================================+
| ``stdfilt(I, ones(5,5))`` | ``compute_sd_map(I, kernel_size=5)``   |
+---------------------------+----------------------------------------+
| ``mean2(SD_map)``         | ``np.mean(sd_map)``                    |
+---------------------------+----------------------------------------+
| ``[counts, bins] =        | ``hist, bins =                         |
| hist(SD_map(:), 100)``    | np.histogram(sd_map.ravel(), 100)``    |
+---------------------------+----------------------------------------+

Performance
~~~~~~~~~~~

- **compute_sd_map**: O(N·M·k²) dove N×M è la dimensione immagine e k è kernel_size
- **estimate_sigma_from_histogram**: O(N) dove N è il numero di pixel
- Tipicamente ~0.5s per immagine 512×512 con kernel 5×5

Dipendenze
~~~~~~~~~~

- **NumPy** >= 1.20: Operazioni array
- **SciPy** >= 1.7: ``ndimage.generic_filter`` per SD map
- **Matplotlib** >= 3.3: Visualizzazioni
- **pydicom** >= 2.0: Lettura file DICOM (opzionale)
