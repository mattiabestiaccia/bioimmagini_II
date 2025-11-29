Come Contribuire
================

Grazie per l'interesse a contribuire al progetto Bioimmagini Positano!

Questa guida ti aiuterà a contribuire efficacemente al progetto.

Tipi di Contributi
-------------------

Puoi contribuire in diversi modi:

* **Nuove Esercitazioni**: Converti esercitazioni MATLAB 3-11
* **Bug Fixes**: Correggi errori nel codice esistente
* **Documentazione**: Migliora README, docstring, o questa docs
* **Test**: Aggiungi unit tests
* **Ottimizzazioni**: Migliora performance del codice
* **Features**: Aggiungi nuove funzionalità

Setup Ambiente Sviluppo
------------------------

1. **Clone Repository**

.. code-block:: bash

   git clone <repository-url>
   cd bioimmagini_positano

2. **Crea Branch**

.. code-block:: bash

   git checkout -b feature/nome-feature

3. **Setup Virtual Environment**

.. code-block:: bash

   cd esercitazioni/esercitazioni_python
   python -m venv venv
   source venv/bin/activate

4. **Install Dependencies**

.. code-block:: bash

   pip install -r es_1__09_03_2022_calcolo_sd/requirements.txt
   pip install -r ../../docs/requirements-docs.txt

Workflow Contribuzione
-----------------------

1. **Scegli Task**

   * Consulta ``TODO.md`` per task aperti
   * Crea issue su GitHub per nuove idee

2. **Sviluppa**

   * Segui :doc:`conventions`
   * Scrivi test per nuovo codice
   * Aggiungi docstring NumPy

3. **Testa**

.. code-block:: bash

   # Run tests
   pytest tests/

   # Check type hints
   mypy src/

   # Lint code
   flake8 src/

4. **Documenta**

   * Aggiorna README.md
   * Aggiungi docstring
   * Build documentazione Sphinx

.. code-block:: bash

   cd docs/
   make clean html
   # Verifica build/html/index.html

5. **Commit**

Segui convenzione commit:

.. code-block:: bash

   git add .
   git commit -m "feat: add wiener filter 3D implementation"

6. **Push e Pull Request**

.. code-block:: bash

   git push origin feature/nome-feature

Poi crea Pull Request su GitHub.

Guidelines Codice
-----------------

Style
~~~~~

* **PEP 8** compliant
* **Type hints** per tutti i parametri
* **Docstring NumPy** per tutte le funzioni
* **Commenti in italiano** per chiarezza

Testing
~~~~~~~

Ogni nuova funzione deve avere test:

.. code-block:: python

   # tests/test_new_feature.py
   def test_new_function():
       result = new_function(input_data)
       expected = expected_output
       assert np.allclose(result, expected)

Documentazione
~~~~~~~~~~~~~~

* README.md per ogni esercitazione
* Docstring con esempi
* Note su differenze MATLAB-Python

Review Process
--------------

Pull Request Checklist
~~~~~~~~~~~~~~~~~~~~~~

Prima di submit:

.. code-block:: text

   ☐ Codice segue conventions
   ☐ Test passano
   ☐ Docstring completi
   ☐ README aggiornato
   ☐ Nessun warning da linter
   ☐ Documentazione Sphinx builda senza errori
   ☐ Branch aggiornato con main

Review Criteria
~~~~~~~~~~~~~~~

I reviewer verificheranno:

* **Correttezza**: Codice funziona come atteso
* **Test**: Coverage adeguata
* **Documentazione**: Chiara e completa
* **Style**: Segue conventions
* **Performance**: Nessuna regressione
* **Compatibilità**: Non rompe codice esistente

Aggiungere Nuova Esercitazione
-------------------------------

Step-by-Step
~~~~~~~~~~~~

1. **Analizza MATLAB**

.. code-block:: bash

   # Studia codice originale
   cd esercitazioni/esercitazioni_matlab/ESERCITAZIONE_XX/

2. **Crea Struttura**

.. code-block:: bash

   cd ../esercitazioni_python
   mkdir es_X__YYYYMMDD_nome/
   cd es_X__YYYYMMDD_nome/
   mkdir -p src data results docs tests

3. **Inizializza Files**

.. code-block:: bash

   touch src/__init__.py
   touch src/utils.py
   touch requirements.txt
   touch README.md
   touch .gitignore

4. **Implementa Codice**

Segui :doc:`rebasing-guide` per conversione.

5. **Scrivi Tests**

.. code-block:: python

   # tests/test_utils.py
   import pytest
   from src.utils import my_function

   def test_my_function():
       assert my_function(input) == expected

6. **Documenta**

Crea documentazione Sphinx:

.. code-block:: bash

   cd ../../../../docs/source/exercises
   mkdir es_X
   cd es_X
   # Crea index.rst, theory.rst, usage.rst, api.rst

Template disponibile in ``exercises/es_1/`` da copiare.

7. **Submit PR**

.. code-block:: bash

   git add .
   git commit -m "feat: add esercitazione X (nome)"
   git push origin feature/es-X

Reporting Bugs
--------------

Trovato un bug? Aiutaci a fixarlo!

1. **Verifica** che non sia già reportato
2. **Crea issue** su GitHub con:

   * Descrizione chiara del problema
   * Steps per riprodurre
   * Output atteso vs attuale
   * Versione Python e dipendenze

Esempio Issue:

.. code-block:: text

   **Bug**: wiener_filter_3d fallisce con volumi piccoli

   **To Reproduce**:
   ```python
   volume = np.random.randn(10, 10, 10)
   filtered = wiener_filter_3d(volume, kernel_size=7)
   ```

   **Expected**: Volume filtrato 10x10x10
   **Actual**: ValueError: kernel size too large

   **Environment**:
   - Python 3.12
   - NumPy 2.3.0
   - SciPy 1.16.0

Suggerimenti Features
---------------------

Vuoi proporre una nuova feature?

1. **Apri issue** con label "enhancement"
2. **Descrivi**:

   * Cosa vuoi aggiungere
   * Perché è utile
   * Possibile implementazione

3. **Discussione**: Attendi feedback prima di implementare

Migliorare Documentazione
--------------------------

La documentazione è sempre migliorabile!

* **Typos**: Correggi direttamente e crea PR
* **Esempi**: Aggiungi code examples nei docstring
* **Tutorial**: Crea nuovi tutorial in ``docs/source/``
* **Traduzioni**: Aiuta con traduzioni italiano-inglese

Build Locale Docs
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs/
   make clean html
   firefox build/html/index.html

Domande?
--------

* **Slack/Discord**: [link-to-channel]
* **Email**: bioimmagini@example.com
* **GitHub Discussions**: [link]

Riconoscimenti
--------------

Tutti i contributori verranno riconosciuti nel README.md e nella
documentazione.

Grazie per contribuire! 🎉

Prossimi Passi
--------------

* **Convenzioni**: :doc:`conventions`
* **Architettura**: :doc:`architecture`
* **Rebasing Guide**: :doc:`rebasing-guide`
