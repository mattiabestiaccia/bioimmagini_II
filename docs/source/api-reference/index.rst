API Reference
=============

Questa sezione contiene la documentazione completa delle API Python del progetto,
auto-generata dai docstring del codice sorgente.

.. note::
   Tutta la documentazione API è estratta automaticamente dai docstring in stile NumPy
   presenti nel codice sorgente. Le modifiche ai docstring si riflettono automaticamente
   qui alla prossima build della documentazione.

Esercitazioni Documentate
--------------------------

.. toctree::
   :maxdepth: 2

   es_1_modules
   es_2_modules

Panoramica Moduli
-----------------

Esercizio 1 - Analisi Rumore MRI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``src.utils`` - Funzioni utility per calcolo SD map, correzione Rayleigh, statistiche
* ``src.calcolo_sd`` - Analisi immagine sintetica
* ``src.esempio_calcolo_sd`` - Analisi fantoccio MRI reale
* ``src.test_m_sd`` - Test Monte Carlo convergenza ROI

Esercizio 2 - Filtraggio 3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``src.filters_3d`` - Filtri 3D (media mobile, gaussiano, Wiener)
* ``src.metrics`` - Metriche qualità (SNR, CNR, edge sharpness)
* ``src.dicom_utils`` - Caricamento DICOM e interpolazione isotropica
* ``src.interactive_roi_selection`` - Tool interattivo per selezione ROI

Convenzioni
-----------

Tutti i moduli seguono queste convenzioni:

* **Docstring**: Formato NumPy/SciPy
* **Type hints**: Presenti per tutti i parametri e return values
* **Esempi**: Code examples nel docstring quando applicabile
* **References**: Link a paper scientifici quando rilevante

Navigazione
-----------

Utilizza la barra di ricerca in alto per trovare rapidamente funzioni specifiche,
oppure naviga attraverso i moduli usando l'indice a sinistra.

Quick Links
-----------

* :ref:`modindex` - Indice alfabetico di tutti i moduli
* :ref:`genindex` - Indice generale
* :ref:`search` - Ricerca full-text
