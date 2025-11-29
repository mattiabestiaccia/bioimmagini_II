Struttura del Progetto
======================

Questa pagina descrive l'organizzazione delle directory e dei file nel progetto
Bioimmagini Positano.

Panoramica
----------

Il progetto è organizzato in modo modulare con una chiara separazione tra:

* **Esercitazioni originali MATLAB** (sola lettura, per riferimento)
* **Conversioni Python** (codice attivo)
* **Dispense e materiale didattico**
* **Documentazione tecnica** (questa docs)

Struttura Completa
------------------

.. code-block:: text

   bioimmagini_positano/
   │
   ├── esercitazioni/
   │   ├── esercitazioni_matlab/           # ⚠️ SOLO LETTURA - Riferimento
   │   │   ├── es_1__09_03_2022/
   │   │   ├── es_2__16_03_2022_filtraggio/
   │   │   └── ...
   │   │
   │   └── esercitazioni_python/           # ✅ Codice Python attivo
   │       ├── venv/                       # Ambiente virtuale condiviso
   │       ├── activate.sh                 # Script attivazione veloce
   │       ├── es_1__09_03_2022_calcolo_sd/
   │       ├── es_2__16_03_2022_filtraggio/
   │       └── es_3__23_03_2022_clustering/
   │
   ├── dispense/                           # Materiale didattico del corso
   │   ├── cap_1.md
   │   ├── Cap_1 Immagine biomedica.pdf
   │   ├── cap_2.md
   │   └── ...
   │
   ├── esempi_matlab/                      # Script MATLAB di esempio
   │
   ├── docs/                               # 📚 Questa documentazione
   │   ├── source/                         # Sorgenti RST
   │   │   ├── conf.py
   │   │   ├── index.rst
   │   │   ├── exercises/
   │   │   ├── api-reference/
   │   │   └── ...
   │   ├── build/html/                     # HTML generato
   │   ├── Makefile
   │   └── requirements-docs.txt
   │
   ├── REBASING_GUIDE.md                   # Guida conversione MATLAB→Python
   ├── README.md                           # Documentazione principale
   ├── TODO.md                             # Task tracking
   └── .claude/
       └── project_context.md              # Context per AI assistants

Struttura di una Esercitazione Python
--------------------------------------

Ogni esercitazione segue una struttura standardizzata:

.. code-block:: text

   es_N__DATE_TITLE/
   ├── src/                    # 📦 Codice sorgente Python
   │   ├── __init__.py        # Inizializzazione modulo
   │   ├── utils.py           # Funzioni utility condivise
   │   ├── script_1.py        # Script principale 1
   │   ├── script_2.py        # Script principale 2
   │   └── ...
   │
   ├── data/                   # 📂 Dati medici (DICOM, immagini)
   │   ├── phantom.dcm
   │   ├── series_001/
   │   └── README.md          # Documentazione dati
   │
   ├── results/                # 📊 Output generati (grafici, tabelle)
   │
   ├── docs/                   # 📄 Documentazione specifica
   │   ├── esercitazione_XX.pdf
   │   └── ...
   │
   ├── notebooks/              # 📓 Jupyter notebooks (opzionale)
   │
   ├── tests/                  # 🧪 Unit tests
   │   └── test_*.py
   │
   ├── requirements.txt        # Dipendenze Python specifiche
   ├── README.md              # Documentazione esercitazione
   └── .gitignore

Convenzioni di Naming
---------------------

File e Directory
~~~~~~~~~~~~~~~~

* **Esercitazioni**: ``es_N__YYYYMMDD_nome_descrittivo/``
* **Script Python**: ``snake_case.py`` (es. ``calcolo_sd.py``)
* **Moduli**: ``__init__.py`` in ogni directory ``src/``
* **Test**: ``test_*.py`` per unit tests

Codice Python
~~~~~~~~~~~~~

* **Funzioni**: ``snake_case`` (es. ``compute_sd_map``)
* **Classi**: ``PascalCase`` (es. ``ROISelector``)
* **Costanti**: ``UPPER_CASE`` (es. ``RAYLEIGH_FACTOR``)
* **Variabili private**: ``_leading_underscore``

Documentazione
~~~~~~~~~~~~~~

* **Docstring**: Stile NumPy/SciPy
* **Type hints**: Sempre presenti per parametri e return values
* **Comments**: Italiano per chiarezza didattica

File Importanti
---------------

File di Configurazione
~~~~~~~~~~~~~~~~~~~~~~

* ``requirements.txt``: Dipendenze per ogni esercitazione
* ``docs/requirements-docs.txt``: Dipendenze per build documentazione
* ``.gitignore``: Esclude venv, cache, risultati temporanei

Documentazione
~~~~~~~~~~~~~~

* ``README.md`` (root): Entry point del progetto
* ``REBASING_GUIDE.md``: Guida dettagliata conversione MATLAB→Python
* ``TODO.md``: Tracciamento task e progresso conversione
* ``docs/``: Documentazione tecnica completa (questo sito)

Navigazione Rapida
------------------

**Per studenti:**

* Quick start → ``README.md`` nella root
* Esercitazioni → ``esercitazioni/esercitazioni_python/es_N_*/README.md``
* Teoria → PDFs in ``dispense/``

**Per sviluppatori:**

* Linee guida → ``REBASING_GUIDE.md``
* API reference → :doc:`../api-reference/index`
* Architettura → :doc:`../developer-guide/architecture`

**Per eseguire codice:**

.. code-block:: bash

   cd esercitazioni/esercitazioni_python
   source venv/bin/activate
   cd es_1__09_03_2022_calcolo_sd/src
   python calcolo_sd.py

Dimensioni Directory
--------------------

Riferimento dimensioni approssimative (include dati DICOM):

* ``esercitazioni_python/``: ~1.1 GB (include venv)
* ``esercitazioni_matlab/``: ~700 MB
* ``dispense/``: ~87 MB
* ``docs/``: ~2 MB (sorgenti), ~10 MB (build)

.. note::
   La directory ``data/`` in ogni esercitazione contiene file DICOM medici
   che possono essere di grandi dimensioni (centinaia di MB). Questi file
   non sono tracciati in git.

Prossimi Passi
--------------

* **Inizia con il codice**: :doc:`quickstart`
* **Comprendi l'architettura**: :doc:`../developer-guide/architecture`
* **Contribuisci**: :doc:`../developer-guide/contributing`
