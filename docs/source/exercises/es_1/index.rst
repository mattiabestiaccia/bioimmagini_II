Esercizio 1: Analisi Rumore MRI
================================

*Calcolo della Deviazione Standard in Immagini di Risonanza Magnetica*

.. toctree::
   :maxdepth: 2

   theory
   usage
   api
   examples
   matlab-conversion

Panoramica
----------

Questa esercitazione copre tre aspetti fondamentali dell'analisi del rumore
nelle immagini MRI:

1. **Analisi su immagine sintetica** - Validazione metodi stima rumore
2. **Analisi su fantoccio MRI** - Applicazione su dati reali
3. **Test Monte Carlo** - Valutazione convergenza statistica vs dimensione ROI

Obiettivi Didattici
-------------------

* Comprendere la distribuzione del rumore nelle immagini MRI magnitude
* Implementare metodi automatici di stima del rumore (SD map)
* Applicare la correzione Rayleigh per il background
* Valutare l'effetto della dimensione ROI sulla precisione delle stime

Dataset
-------

* **Fantoccio MRI**: ``data/phantom.dcm``
* **Serie cardiaca LGE**: ``data/esempio_LGE/`` (18 file DICOM, ~2.4MB)

(Documentazione completa in fase di sviluppo)
