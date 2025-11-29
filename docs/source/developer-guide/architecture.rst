Architettura del Sistema
========================

Questa pagina descrive i pattern architetturali e le scelte di design
del progetto Bioimmagini Positano.

Principi di Design
------------------

Il progetto segue questi principi fondamentali:

1. **Modularità**: Separazione chiara tra utility, business logic e scripts
2. **Riusabilità**: Funzioni comuni in moduli ``utils.py`` condivisi
3. **Type Safety**: Type hints per tutti i parametri e return values
4. **Documentazione**: Docstring NumPy completi per ogni funzione
5. **Testabilità**: Struttura che facilita unit testing
6. **Pedagogia**: Codice chiaro e didattico, commenti in italiano

Pattern Architetturali
-----------------------

Struttura a Tre Livelli
~~~~~~~~~~~~~~~~~~~~~~~~

Ogni esercitazione segue un'architettura a tre livelli:

.. code-block:: text

   ┌─────────────────────────────────┐
   │   Scripts (Entry Points)        │  ← calcolo_sd.py, main_filtering.py
   │   - Orchestrazione workflow     │
   │   - Parsing argomenti CLI       │
   │   - Visualizzazione risultati   │
   └─────────────┬───────────────────┘
                 │ importa
   ┌─────────────▼───────────────────┐
   │   Business Logic Modules        │  ← filters_3d.py, metrics.py
   │   - Algoritmi core              │
   │   - Processing pipelines        │
   │   - Domain logic                │
   └─────────────┬───────────────────┘
                 │ usa
   ┌─────────────▼───────────────────┐
   │   Utility Modules               │  ← utils.py, dicom_utils.py
   │   - Funzioni riutilizzabili    │
   │   - I/O operations              │
   │   - Helper functions            │
   └─────────────────────────────────┘

Esempio Esercizio 1
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   src/calcolo_sd.py (Script)
   ├─> usa src.utils.create_synthetic_image()
   ├─> usa src.utils.compute_sd_map()
   ├─> usa src.utils.estimate_sigma_from_histogram()
   └─> usa src.utils.compute_statistics()

Moduli Comuni
-------------

utils.py
~~~~~~~~

Ogni esercitazione ha un modulo ``utils.py`` che fornisce:

* **Data structures**: Classi per ROI, risultati, configurazioni
* **I/O functions**: Lettura/scrittura DICOM, salvataggio grafici
* **Statistical functions**: Calcoli statistici comuni
* **Preprocessing**: Normalizzazione, masking, resampling

**Esempio** (da Esercizio 1):

.. code-block:: python

   from src.utils import (
       compute_sd_map,
       apply_rayleigh_correction,
       estimate_sigma_from_histogram,
   )

Design Patterns Utilizzati
---------------------------

Factory Pattern
~~~~~~~~~~~~~~~

Per la creazione di filtri configurabili:

.. code-block:: python

   # Esercizio 2
   def create_filter(filter_type: str, **params):
       if filter_type == 'moving_average':
           return MovingAverageFilter(**params)
       elif filter_type == 'gaussian':
           return GaussianFilter(**params)
       # ...

Strategy Pattern
~~~~~~~~~~~~~~~~

Per intercambiare algoritmi (es. metodi di stima sigma):

.. code-block:: python

   # Diverse strategie per stimare sigma
   sigma_mean = np.mean(sd_map)
   sigma_median = np.median(sd_map)
   sigma_histogram, _, _ = estimate_sigma_from_histogram(sd_map)

Pipeline Pattern
~~~~~~~~~~~~~~~~

Per elaborazioni sequenziali:

.. code-block:: python

   # Esercizio 2
   volume = load_dicom_volume(path)
   volume = make_isotropic(volume, metadata)
   volume_filtered = apply_filter(volume, filter_params)
   snr = calculate_snr(volume_filtered, roi_mask)

Gestione Dati
--------------

DICOM Files
~~~~~~~~~~~

* Lettura con ``pydicom.dcmread()``
* Conversione HU corretta con ``RescaleIntercept`` e ``RescaleSlope``
* Metadata extraction per spacing e orientamento

NumPy Arrays
~~~~~~~~~~~~

* Tutte le immagini rappresentate come ``np.ndarray``
* Convenzione dimensioni: ``(height, width)`` per 2D, ``(depth, height, width)`` per 3D
* Type: ``float64`` per elaborazioni, ``uint8``/``uint16`` per salvataggio

Error Handling
--------------

Il progetto segue queste convenzioni per error handling:

.. code-block:: python

   def compute_sd_map(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
       """
       Compute standard deviation map.

       Parameters
       ----------
       image : np.ndarray
           Input image
       kernel_size : int
           Must be > 0

       Returns
       -------
       np.ndarray
           SD map

       Raises
       ------
       ValueError
           If kernel_size <= 0 or image is empty
       """
       if kernel_size <= 0:
           raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
       if image.size == 0:
           raise ValueError("image cannot be empty")

       # Implementation...

Testing Strategy
----------------

Unit Tests
~~~~~~~~~~

* Test per ogni funzione in ``utils.py``
* Verifica equivalenza MATLAB-Python
* Test casi limite e edge cases

**Directory**: ``tests/test_*.py``

Validazione Numerica
~~~~~~~~~~~~~~~~~~~~

* Confronto output Python vs MATLAB
* Tolleranza numerica: ``np.allclose(py_result, matlab_result, atol=1e-10)``
* Verifica proprietà matematiche (es. Rayleigh factor = 1.526)

Performance Considerations
--------------------------

Ottimizzazioni Implementate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Vectorizzazione NumPy**: Niente loop Python quando possibile
2. **Caching**: Risultati intermedi salvati in memoria
3. **Lazy loading**: Caricamento dati on-demand
4. **Efficient filtering**: ``scipy.ndimage`` invece di loop manuali

**Esempio stdfilt3** (Esercizio 2):

.. code-block:: python

   # ❌ Lento - loop Python
   for z in range(depth):
       for y in range(height):
           for x in range(width):
               std_map[z,y,x] = np.std(volume[z-k:z+k, y-k:y+k, x-k:x+k])

   # ✅ Veloce - vectorizzato
   mean_local = uniform_filter(volume, size=kernel_size)
   mean_of_squares = uniform_filter(volume**2, size=kernel_size)
   std_map = np.sqrt(mean_of_squares - mean_local**2)

Memory Management
~~~~~~~~~~~~~~~~~

* Volumi 3D possono essere grandi (>500MB)
* Liberare memoria con ``del volume`` quando non più necessario
* Uso di ``np.float32`` invece di ``float64`` quando la precisione lo permette

Estensibilità
-------------

Aggiungere Nuove Esercitazioni
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Crea directory con struttura standard
2. Implementa modulo ``utils.py`` con funzioni riutilizzabili
3. Crea script principali che usano ``utils.py``
4. Scrivi test in ``tests/``
5. Aggiungi documentazione in ``docs/source/exercises/es_N/``

Aggiungere Nuovi Filtri
~~~~~~~~~~~~~~~~~~~~~~~~

Esempio (Esercizio 2):

.. code-block:: python

   # In filters_3d.py
   def new_filter_3d(volume: np.ndarray, **params) -> np.ndarray:
       """
       Implement new filter.

       Parameters
       ----------
       volume : np.ndarray
           3D volume
       **params
           Filter-specific parameters

       Returns
       -------
       np.ndarray
           Filtered volume
       """
       # Implementation
       return filtered_volume

Prossimi Passi
--------------

* **Coding conventions**: :doc:`conventions`
* **Contribute**: :doc:`contributing`
* **Rebasing guide**: :doc:`rebasing-guide`
