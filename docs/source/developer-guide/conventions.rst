Convenzioni di Codifica
=======================

Questa pagina descrive le convenzioni di stile e best practices
seguite nel progetto.

Style Guide
-----------

Il progetto segue **PEP 8** con alcune estensioni specifiche per
codice scientifico.

Naming Conventions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Funzioni e variabili: snake_case
   def compute_sd_map(image, kernel_size):
       roi_mask = create_mask()

   # Classi: PascalCase
   class ROISelector:
       pass

   # Costanti: UPPER_CASE
   RAYLEIGH_FACTOR = 1.526
   DEFAULT_KERNEL_SIZE = 5

   # Variabili private: _leading_underscore
   def _internal_helper():
       pass

Type Hints
~~~~~~~~~~

Sempre presenti per parametri e return values:

.. code-block:: python

   from typing import Tuple, Optional
   import numpy as np

   def compute_sd_map(
       image: np.ndarray,
       kernel_size: int = 5
   ) -> np.ndarray:
       """Compute standard deviation map."""
       pass

   def load_dicom_volume(
       path: str
   ) -> Tuple[np.ndarray, dict]:
       """Load DICOM volume with metadata."""
       return volume, metadata

Docstrings
----------

Formato NumPy/SciPy
~~~~~~~~~~~~~~~~~~~

Tutti i docstring seguono il formato NumPy:

.. code-block:: python

   def wiener_filter_3d(
       volume: np.ndarray,
       kernel_size: int = 7,
       noise_variance: Optional[float] = None
   ) -> np.ndarray:
       """
       Applica un filtro Wiener adattivo 3D.

       Implementazione della formula:
       I_W = I_MM + α(I_OR - I_MM)

       Parameters
       ----------
       volume : np.ndarray
           Volume 3D originale (I_OR)
       kernel_size : int, default=7
           Dimensione del kernel cubico
       noise_variance : float, optional
           Varianza del rumore (σ²). Se None, viene stimata.

       Returns
       -------
       np.ndarray
           Volume filtrato con Wiener

       Notes
       -----
       Il filtro Wiener è adattivo:

       - Aree omogenee (I_VAR ≈ σ²): α ≈ 0 → media mobile
       - Contorni (I_VAR >> σ²): α ≈ 1 → preserva originale

       Examples
       --------
       >>> volume = load_dicom_volume("data/phantom.dcm")
       >>> filtered = wiener_filter_3d(volume, kernel_size=7)

       References
       ----------
       .. [1] Gonzalez & Woods, "Digital Image Processing", Ch. 5
       """
       pass

Sezioni Docstring
~~~~~~~~~~~~~~~~~

Ordine standard:

1. **Summary line** (una riga, imperativo)
2. **Extended description** (opzionale, dettagli algoritmo)
3. **Parameters**
4. **Returns**
5. **Raises** (se applicabile)
6. **Notes** (teoria, dettagli implementazione)
7. **Examples** (code examples runnable)
8. **References** (paper scientifici)

Code Style
----------

Imports
~~~~~~~

Ordine standard:

.. code-block:: python

   # 1. Standard library
   import os
   import sys
   from pathlib import Path
   from typing import Tuple, Optional

   # 2. Third-party libraries
   import numpy as np
   from scipy import ndimage
   import matplotlib.pyplot as plt
   import pydicom

   # 3. Local imports
   from src.utils import compute_sd_map
   from src.filters_3d import wiener_filter_3d

Function Length
~~~~~~~~~~~~~~~

* **Massimo 50 righe** per funzione
* Se supera, refactor in funzioni helper
* Una funzione = una responsabilità

.. code-block:: python

   # ❌ Troppo lungo
   def process_everything(data):
       # 200 righe di codice
       pass

   # ✅ Refactored
   def process_everything(data):
       preprocessed = preprocess_data(data)
       filtered = apply_filters(preprocessed)
       results = compute_metrics(filtered)
       return results

   def preprocess_data(data):
       # Implementazione
       pass

Line Length
~~~~~~~~~~~

* **Massimo 100 caratteri** per riga
* 88 caratteri raccomandato (black formatter)

Comments
--------

Inline Comments
~~~~~~~~~~~~~~~

* In **italiano** per chiarezza didattica
* Spiegano il "perché", non il "cosa"

.. code-block:: python

   # ✅ Buono - spiega il perché
   # Aggiungi epsilon per evitare divisione per zero in regioni flat
   alpha = (variance - noise_var) / (variance + epsilon)

   # ❌ Cattivo - ripete il codice
   # Sottrai noise_var da variance
   alpha = (variance - noise_var) / variance

Block Comments
~~~~~~~~~~~~~~

Per sezioni complesse:

.. code-block:: python

   # ============================================
   # Fase 1: Preprocessing del volume
   # ============================================
   # Convertiamo il volume DICOM in Hounsfield Units (HU)
   # applicando RescaleIntercept e RescaleSlope dal metadata.
   # Questo è necessario perché dicomread() di MATLAB ignora
   # questi campi fino alla versione 2021b.

   volume_hu = metadata['RescaleIntercept'] + \
               metadata['RescaleSlope'] * volume_raw

Error Handling
--------------

Validation
~~~~~~~~~~

Valida input all'inizio della funzione:

.. code-block:: python

   def compute_sd_map(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
       """Compute SD map."""
       # Validazione
       if not isinstance(image, np.ndarray):
           raise TypeError(f"image must be ndarray, got {type(image)}")

       if image.ndim != 2:
           raise ValueError(f"image must be 2D, got shape {image.shape}")

       if kernel_size <= 0:
           raise ValueError(f"kernel_size must be > 0, got {kernel_size}")

       if kernel_size % 2 == 0:
           raise ValueError(f"kernel_size must be odd, got {kernel_size}")

       # Implementazione
       ...

Exceptions
~~~~~~~~~~

* Usa eccezioni built-in quando possibile: ``ValueError``, ``TypeError``, ``FileNotFoundError``
* Crea custom exceptions solo se necessario
* Includi messaggi informativi

Testing
-------

Test File Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # tests/test_utils.py
   import pytest
   import numpy as np
   from src.utils import compute_sd_map, apply_rayleigh_correction

   class TestComputeSDMap:
       """Test suite for compute_sd_map function."""

       def test_output_shape(self):
           """SD map should have same shape as input."""
           image = np.random.randn(100, 100)
           sd_map = compute_sd_map(image, kernel_size=5)
           assert sd_map.shape == image.shape

       def test_uniform_image(self):
           """SD map of uniform image should be near zero."""
           image = np.ones((100, 100))
           sd_map = compute_sd_map(image)
           assert np.allclose(sd_map, 0, atol=1e-10)

       @pytest.mark.parametrize("kernel_size", [3, 5, 7, 9])
       def test_different_kernel_sizes(self, kernel_size):
           """Test with different kernel sizes."""
           image = np.random.randn(50, 50)
           sd_map = compute_sd_map(image, kernel_size=kernel_size)
           assert sd_map.shape == image.shape

Test Naming
~~~~~~~~~~~

* Test files: ``test_*.py``
* Test functions: ``test_*()``
* Test classes: ``Test*``

Descrittivo e specifico:

.. code-block:: python

   # ✅ Chiaro
   def test_rayleigh_correction_returns_expected_factor():
       pass

   # ❌ Vago
   def test_rayleigh():
       pass

Version Control
---------------

Commit Messages
~~~~~~~~~~~~~~~

Formato:

.. code-block:: text

   <type>: <subject>

   <body>

   <footer>

Types:

* ``feat``: Nuova feature
* ``fix``: Bug fix
* ``docs``: Documentazione
* ``refactor``: Refactoring
* ``test``: Aggiunta/modifica test
* ``chore``: Maintenance

Esempio:

.. code-block:: text

   feat: add 3D Wiener filter implementation

   Implementata versione 3D del filtro Wiener adattivo.
   MATLAB ha solo wiener2 (2D), questa è un'estensione completa.

   - Calcolo varianza locale 3D
   - Coefficiente α adattivo
   - Gestione regioni flat con epsilon

   Closes #42

Branching
~~~~~~~~~

* ``main``: Codice stabile
* ``develop``: Sviluppo attivo
* ``feature/nome-feature``: Nuove feature
* ``fix/nome-bug``: Bug fixes

Documentation
-------------

README Files
~~~~~~~~~~~~

Ogni esercitazione deve avere un ``README.md`` con:

1. Titolo e descrizione
2. Obiettivi didattici
3. Setup e installazione
4. Usage con esempi
5. Output attesi
6. Troubleshooting
7. References

Code Comments
~~~~~~~~~~~~~

* Commenti in italiano per chiarezza didattica
* Referenzia codice MATLAB originale quando applicabile
* Spiega algoritmi complessi step-by-step

Best Practices
--------------

NumPy
~~~~~

.. code-block:: python

   # ✅ Vectorizzato
   result = np.mean(array, axis=0)

   # ❌ Loop Python
   result = [np.mean(array[i]) for i in range(len(array))]

   # ✅ Broadcasting
   normalized = (array - mean) / std

   # ❌ Loop
   normalized = np.zeros_like(array)
   for i in range(len(array)):
       normalized[i] = (array[i] - mean) / std

SciPy
~~~~~

.. code-block:: python

   # ✅ Usa scipy per filtri
   from scipy.ndimage import uniform_filter
   filtered = uniform_filter(volume, size=7)

   # ❌ Loop manuale
   for z in range(depth):
       for y in range(height):
           for x in range(width):
               filtered[z,y,x] = np.mean(volume[z-3:z+4, y-3:y+4, x-3:x+4])

Matplotlib
~~~~~~~~~~

.. code-block:: python

   # ✅ Salva con alta risoluzione
   plt.savefig('results/plot.png', dpi=150, bbox_inches='tight')

   # ✅ Chiudi figure per liberare memoria
   plt.close('all')

Tools
-----

Linting
~~~~~~~

.. code-block:: bash

   # flake8 per PEP 8
   flake8 src/

   # mypy per type checking
   mypy src/

Formatting
~~~~~~~~~~

.. code-block:: bash

   # black per auto-formatting
   black src/

   # isort per ordinare imports
   isort src/

Prossimi Passi
--------------

* **Architettura**: :doc:`architecture`
* **Contributing**: :doc:`contributing`
* **Rebasing Guide**: :doc:`rebasing-guide`
