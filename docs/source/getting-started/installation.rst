Installazione
=============

Prerequisiti
------------

Prima di iniziare, assicurati di avere installato:

* **Python 3.8 o superiore** (raccomandato: Python 3.12)
* **pip** (package manager Python)
* **git** (per clonare il repository)

Setup Ambiente Virtuale
------------------------

Si raccomanda fortemente di utilizzare un ambiente virtuale Python per isolare
le dipendenze del progetto.

.. code-block:: bash

   # Naviga nella directory del progetto
   cd bioimmagini_positano/esercitazioni/esercitazioni_python

   # Crea un ambiente virtuale
   python -m venv venv

   # Attiva l'ambiente virtuale
   # Su Linux/Mac:
   source venv/bin/activate

   # Su Windows:
   venv\Scripts\activate

Installazione Dipendenze
-------------------------

Il progetto utilizza le seguenti librerie Python principali:

* **NumPy** (≥2.3) - Calcolo numerico e array operations
* **SciPy** (≥1.16) - Algoritmi scientifici e signal processing
* **Matplotlib** (≥3.10) - Visualizzazione dati
* **PyDICOM** (≥3.0) - Lettura file DICOM
* **scikit-image** (≥0.25) - Elaborazione immagini
* **scikit-learn** - Machine learning (clustering, classificazione)

Un file ``requirements.txt`` è disponibile in ogni cartella esercitazione.
Per installare tutte le dipendenze:

.. code-block:: bash

   pip install -r requirements.txt

Verifica Installazione
----------------------

Per verificare che tutto sia installato correttamente, esegui:

.. code-block:: python

   python -c "import numpy, scipy, matplotlib, pydicom; print('OK!')"

Se non ci sono errori, l'installazione è completata con successo.

Struttura Directory
-------------------

Dopo l'installazione, la struttura dovrebbe essere:

.. code-block:: text

   bioimmagini_positano/
   ├── esercitazioni/
   │   └── esercitazioni_python/
   │       ├── venv/                    # Ambiente virtuale
   │       ├── es_1__09_03_2022_calcolo_sd/
   │       ├── es_2__16_03_2022_filtraggio/
   │       └── ...
   ├── docs/                            # Questa documentazione
   └── README.md

Prossimi Passi
--------------

Ora sei pronto per iniziare! Consulta la :doc:`quickstart` per eseguire
il tuo primo script.
