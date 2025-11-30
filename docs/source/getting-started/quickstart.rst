Quick Start
===========

Questa guida ti porterà dall'installazione all'esecuzione del tuo primo script
in meno di 5 minuti.

Setup Rapido
------------

.. code-block:: bash

   # 1. Clona il repository
   git clone <repository-url>
   cd bioimmagini_positano

   # 2. Setup ambiente virtuale
   cd esercitazioni/esercitazioni_python
   python -m venv venv
   source venv/bin/activate  # Linux/Mac

   # 3. Installa dipendenze
   pip install -r es_1__09_03_2022_calcolo_sd/requirements.txt

Primo Script: Analisi Immagine Sintetica
-----------------------------------------

Eseguiamo lo script più semplice dell'Esercizio 1, che analizza il rumore
in un'immagine sintetica.

.. code-block:: bash

   cd es_1__09_03_2022_calcolo_sd/src
   python calcolo_sd.py

Output Atteso
~~~~~~~~~~~~~

Lo script genera:

1. **Immagine sintetica** 512×512 pixel con pattern multipli e rumore gaussiano
2. **Istogramma** dei valori di intensità
3. **Mappa di deviazione standard** (SD map) calcolata con sliding window 5×5
4. **Confronto** tra 4 metodi di stima del rumore:

   * Valore vero (σ = 5.0)
   * Media SD map
   * Mediana SD map
   * Massimo istogramma

I grafici vengono salvati in ``../results/``.

Esempio Output Console
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ========================================
   Metodi di Stima di Sigma
   ========================================
   Valore vero: 5.0000
   Media:       5.1234
   Mediana:     5.0987
   Max histogram: 5.0123

   Errore percentuale:
   Media:       2.47%
   Mediana:     1.97%
   Max histogram: 0.25%

Secondo Script: Fantoccio MRI
------------------------------

Analizziamo ora un'immagine reale di fantoccio MRI con selezione manuale delle ROI.

.. code-block:: bash

   python esempio_calcolo_sd.py --interactive

Questo apre finestre interattive per:

1. Selezionare 3 ROI circolari (olio, acqua, background)
2. Visualizzare le statistiche per ogni ROI
3. Applicare la correzione Rayleigh per il background

.. note::
   La modalità interattiva richiede un display grafico. Se stai lavorando
   su un server remoto senza X11, usa la modalità non-interattiva
   (senza ``--interactive``).

Esplorare il Codice
-------------------

Il codice sorgente è ben documentato con docstring in stile NumPy.
Esplora i moduli:

.. code-block:: python

   from src.utils import compute_sd_map, apply_rayleigh_correction

   # Vedi la documentazione
   help(compute_sd_map)
   help(apply_rayleigh_correction)

Oppure consulta la :doc:`../api-reference/index` per la documentazione
completa delle API.

Prossimi Passi
--------------

* **Esercizio 1 completo**: :doc:`../exercises/es_1/index`
* **Esercizio 2 (Filtraggio 3D)**: :doc:`../exercises/es_2/index`
* **Teoria dettagliata**: :doc:`../exercises/es_1/theory`
* **API Reference**: :doc:`../api-reference/index`
