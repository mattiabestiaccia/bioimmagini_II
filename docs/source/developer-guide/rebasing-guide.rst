Guida alla Conversione MATLAB → Python
=====================================

Questa pagina fornisce le linee guida per convertire le esercitazioni MATLAB
in Python equivalente.

.. note::
   Questa è una versione sintetica. Per la guida completa, consulta il file
   ``REBASING_GUIDE.md`` nella root del progetto.

Workflow di Conversione
------------------------

1. **Analisi MATLAB**

   * Leggi tutto il codice MATLAB
   * Identifica funzioni chiave e algoritmi
   * Documenta dipendenze e workflow

2. **Setup Python**

   * Crea struttura directory standard
   * Copia file DICOM in ``data/``
   * Inizializza ``requirements.txt``

3. **Conversione Codice**

   * Implementa utilities in ``utils.py``
   * Converti script principali
   * Mantieni nomi variabili simili per tracciabilità

4. **Validazione**

   * Confronta output Python vs MATLAB
   * Verifica equivalenza numerica
   * Testa casi limite

5. **Documentazione**

   * Scrivi README.md
   * Aggiungi docstring NumPy
   * Documenta differenze MATLAB-Python

Equivalenze Comuni
------------------

MATLAB → Python Cheat Sheet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Operazione
     - MATLAB
     - Python
   * - Load DICOM
     - ``dicomread('file.dcm')``
     - ``pydicom.dcmread('file.dcm')``
   * - SD map 2D
     - ``stdfilt(I)``
     - ``scipy.ndimage.generic_filter(I, np.std, size=3)``
   * - Media mobile
     - ``fspecial('average') + imfilter``
     - ``scipy.ndimage.uniform_filter()``
   * - Filtro Gaussiano
     - ``imgaussfilt(I, sigma)``
     - ``scipy.ndimage.gaussian_filter(I, sigma)``
   * - Histogram
     - ``hist(data, bins)``
     - ``np.histogram(data, bins)``
   * - Interpolazione 3D
     - ``interp3(X, Y, Z, V, Xq, Yq, Zq)``
     - ``scipy.ndimage.zoom()``
   * - Random Gaussian
     - ``randn(n, m)``
     - ``np.random.normal(0, 1, (n, m))``
   * - Standard deviation
     - ``std(X)``
     - ``np.std(X, ddof=1)``

.. warning::
   In Python, ``np.std()`` calcola per default la population standard deviation.
   Usa ``ddof=1`` per la sample std (comportamento MATLAB).

Funzioni Non Disponibili in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alcune funzioni MATLAB non hanno equivalente diretto:

**stdfilt3** (SD 3D)

  * MATLAB: Non esiste nativo
  * Python: Implementato custom in ``filters_3d.py``
  * Formula: :math:`\sigma = \sqrt{E[X^2] - E[X]^2}`

**wiener2 3D**

  * MATLAB: Solo 2D
  * Python: Implementato 3D in ``filters_3d.py``
  * Vedi :doc:`../exercises/es_2/theory` per dettagli

Gotchas Comuni
--------------

Indicizzazione
~~~~~~~~~~~~~~

MATLAB è 1-indexed, Python è 0-indexed:

.. code-block:: matlab

   % MATLAB
   I(1, 1)      % Primo elemento

.. code-block:: python

   # Python
   I[0, 0]      # Primo elemento

Slicing
~~~~~~~

.. code-block:: matlab

   % MATLAB - inclusivo
   I(1:10, 1:10)  % Include elemento 10

.. code-block:: python

   # Python - esclusivo
   I[0:10, 0:10]  # Esclude elemento 10

Array Shapes
~~~~~~~~~~~~

MATLAB è column-major, NumPy è row-major:

.. code-block:: matlab

   % MATLAB
   size(I)  % [rows, cols]

.. code-block:: python

   # Python
   I.shape  # (rows, cols)

Ma attenzione con ``meshgrid``:

.. code-block:: matlab

   % MATLAB
   [X, Y] = meshgrid(x, y);  % X è MxN, Y è MxN

.. code-block:: python

   # Python (comportamento diverso!)
   X, Y = np.meshgrid(x, y, indexing='xy')  # Specifica indexing!

Best Practices
--------------

Naming
~~~~~~

* **Mantieni nomi simili** al MATLAB quando possibile
* Converti da camelCase a snake_case
* Commenta equivalenze:

.. code-block:: python

   # Equivalente a MATLAB: calcolo_SD.m:45
   sd_map = compute_sd_map(image, kernel_size=5)

Documentazione
~~~~~~~~~~~~~~

Includi sempre sezione "Confronto MATLAB-Python" nel README:

.. code-block:: markdown

   ## Confronto con MATLAB

   | Feature | MATLAB | Python |
   |---------|--------|--------|
   | SD Map | `stdfilt()` | `scipy.ndimage.generic_filter()` |

Testing
~~~~~~~

Valida equivalenza numerica:

.. code-block:: python

   import numpy as np

   # Load MATLAB output
   matlab_output = loadmat('matlab_result.mat')['result']

   # Compute Python version
   python_output = my_python_function(data)

   # Assert equality (with tolerance)
   assert np.allclose(python_output, matlab_output, rtol=1e-10)

Ottimizzazione
~~~~~~~~~~~~~~

Python può essere più lento di MATLAB compilato. Ottimizza:

* **Vectorizza**: Usa NumPy invece di loop
* **Profila**: ``cProfile`` per identificare bottleneck
* **Numba**: JIT compilation per loop critici

.. code-block:: python

   from numba import jit

   @jit(nopython=True)
   def fast_loop(data):
       result = np.zeros_like(data)
       for i in range(len(data)):
           result[i] = expensive_operation(data[i])
       return result

Checklist Conversione
----------------------

Prima di considerare completa una conversione, verifica:

.. code-block:: text

   ☐ Struttura directory creata
   ☐ File DICOM copiati in data/
   ☐ utils.py implementato
   ☐ Script principali convertiti
   ☐ Docstring NumPy completi
   ☐ Type hints aggiunti
   ☐ README.md scritto
   ☐ Output Python == Output MATLAB (numericamente)
   ☐ Tabella confronto MATLAB-Python
   ☐ requirements.txt aggiornato
   ☐ .gitignore configurato
   ☐ Test scritti
   ☐ Documentazione Sphinx aggiornata

Risorse
-------

* **Guida completa**: ``/home/brusc/Projects/bioimmagini_positano/REBASING_GUIDE.md``
* **NumPy for MATLAB users**: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
* **SciPy docs**: https://docs.scipy.org/doc/scipy/
* **PyDICOM guide**: https://pydicom.github.io/

Prossimi Passi
--------------

* **Architettura**: :doc:`architecture`
* **Convenzioni**: :doc:`conventions`
* **Contribute**: :doc:`contributing`
